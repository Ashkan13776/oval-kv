"""C3 cache-level baselines for LLaVA-Video-7B-Qwen2 (ovalkv env), same ports as qwen_c3.py:
InfiniPot-V (block-wise prefill + TaR/VaN compression) and ReKV (frame-block retrieval once per
question). Inputs are built exactly as vision/run_videomme.py (frames, time instruction, stream
split); the LM is run with the unpatched HF Qwen2 forward (as run_videomme --protocol official)
with EXPLICIT position ids, so compressed caches keep true RoPE positions. Unit = one frame =
210 tokens (14 x 14 + 14 newline tokens). --bench lvbench uses the LVBench docs and frame cache.

    python scripts/revision/C/llava_c3.py --method infinipot --block 32 --compress 24 --bench videomme ...
"""
import json, os, socket, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "vision")); sys.path.insert(0, HERE)
import run_videomme as RV
import lvbench as LVB
from qwen_c3 import infinipot_compress, kv, set_kv


def fwd(model, **k):
    from transformers import Qwen2ForCausalLM
    out = Qwen2ForCausalLM.forward(model, use_cache=True, num_logits_to_keep=1, **k)
    return out.logits, out.past_key_values


@torch.no_grad()
def answer(doc, args, tok, model, proc):
    from llava.constants import IMAGE_TOKEN_INDEX
    from transformers import DynamicCache
    ids, pix, hashes = RV.build_inputs(doc, args, tok, model, proc)
    t0 = time.time()
    vt = model.get_vision_tower(); vt.to("cuda:0")
    _, _, _, _, emb, _ = model.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, [pix], modalities=["video"])
    vt.to("cpu"); torch.cuda.empty_cache()
    p = int((ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0])
    n_vis = emb.shape[1] - (ids.shape[1] - 1)
    kctx = RV.build_inputs.n_ctx_text
    split = p + n_vis + kctx
    stream = ids[0, p + 1 + kctx:].tolist()
    dev = emb.device
    F = args.frames
    tpf = n_vis // F
    assert tpf * F == n_vis
    pos_all = torch.arange(split, device=dev)[None]
    extra = {}
    if args.method == "infinipot":
        B, C = args.block, args.compress
        bounds, g = [], 0
        while g < F:
            step = B if g == 0 else B - C
            bounds.append((g, min(F, g + step))); g += step
        pkv, cur = DynamicCache(), 0
        for bi, (g0, g1) in enumerate(bounds):
            a = 0 if bi == 0 else p + g0 * tpf
            b = p + g1 * tpf if bi < len(bounds) - 1 else split
            logits, pkv = fwd(model, inputs_embeds=emb[:, a:b], position_ids=pos_all[:, a:b], past_key_values=pkv)
            cur += g1 - g0
            if bi < len(bounds) - 1 and cur > C:
                infinipot_compress(pkv, p, C, tpf, cur)
                cur = C
        extra = dict(blocks=len(bounds))
    else:                                           # ReKV
        layers = model.model.layers
        cap = {}
        hooks = [l.self_attn.k_proj.register_forward_hook(lambda m, i, o, li=li: cap.__setitem__(li, o[0]))
                 for li, l in enumerate(layers)]
        logits, pkv = fwd(model, inputs_embeds=emb[:, :split], position_ids=pos_all, past_key_values=DynamicCache())
        for h in hooks:
            h.remove()
        nkv = model.config.num_key_value_heads
        hd = model.config.hidden_size // model.config.num_attention_heads
        g = model.config.num_attention_heads // nkv
        rep = {li: cap[li][p:p + n_vis].float().view(F, tpf, nkv, hd).mean(1) for li in cap}
        del cap
        qcap = {}
        hooks = [l.self_attn.q_proj.register_forward_hook(lambda m, i, o, li=li: qcap.__setitem__(li, o[0].float()))
                 for li, l in enumerate(layers)]
        q_ids = tok(doc["question_text"]).input_ids
        fwd(model, input_ids=torch.tensor([q_ids], device=dev),
            position_ids=torch.arange(split, split + len(q_ids), device=dev)[None], past_key_values=pkv)
        for h in hooks:
            h.remove()
        pkv.crop(split)
        picks = []
        for li in range(len(layers)):
            qm = qcap[li].mean(0).view(nkv, g, hd)
            sc = torch.einsum("khd,Fkd->F", qm, rep[li])
            top = sc.topk(min(args.topk, F)).indices.sort().values
            ti = (p + top[:, None] * tpf + torch.arange(tpf, device=dev)[None]).flatten()
            ti = ti[(ti >= args.sink) & (ti < split - args.recent)]
            idx = torch.cat([torch.arange(args.sink, device=dev), ti, torch.arange(split - args.recent, split, device=dev)])
            K, V = kv(pkv, li)
            set_kv(pkv, li, K[:, :, idx], V[:, :, idx])
            picks.append(top.tolist())
        extra = dict(rekv_frames=picks)
    pos = split
    for t in stream:
        logits, pkv = fwd(model, input_ids=torch.tensor([[t]], device=dev),
                          position_ids=torch.tensor([[pos]], device=dev), past_key_values=pkv)
        pos += 1
    top2 = logits[0, -1].float().topk(2)
    eos = (tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>"))
    gen, nxt = [], int(logits[0, -1].argmax())
    for _ in range(args.max_new_tokens):
        gen.append(nxt)
        if nxt in eos:
            break
        logits, pkv = fwd(model, input_ids=torch.tensor([[nxt]], device=dev),
                          position_ids=torch.tensor([[pos]], device=dev), past_key_values=pkv)
        pos += 1
        nxt = int(logits[0, -1].argmax())
    text = tok.decode(gen, skip_special_tokens=True).strip()
    return text, dict(hashes, n_visual=n_vis, prefill_len=split, n_stream=len(stream), n_gen=len(gen),
                      margin=(top2.values[0] - top2.values[1]).item(), sec=time.time() - t0,
                      cache_len=int(pkv.key_cache[0].shape[2]) - len(stream) - len(gen) + 1,
                      peak_gb=torch.cuda.max_memory_allocated() / 2**30, **extra)


def main():
    ap = RV.build_parser()
    for a in ap._actions:
        if a.dest == "protocol":
            a.required, a.default = False, "official"
    ap.add_argument("--bench", default="videomme", choices=["videomme", "lvbench"])
    ap.add_argument("--c3", dest="method_c3", choices=["infinipot", "rekv"], required=True)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--compress", type=int, default=24)
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()
    assert args.sink == 128 and args.recent == 128, "pass --sink 128 --recent 128 (the parser defaults are the reasoning ones)"
    args.method = args.method_c3
    if args.bench == "lvbench":
        RV.CACHE = os.path.join(LVB.LVB_ROOT, "frames")
        RV.doc_to_text = lambda doc: doc["text"]
        df = LVB.load_docs()
        df["question_text"] = df["question"]
    else:
        import pandas as pd
        df = pd.read_parquet(RV.ANN)
        df = df[df.duration.isin(args.durations)].reset_index(drop=True)
        df["group"] = df.duration
        df["question_text"] = [q + "\n" + "\n".join(list(o)) for q, o in zip(df.question, df.options)]
    if args.limit:
        df = df.iloc[:args.limit]
    if args.qids:
        df = df[df.question_id.isin(args.qids)]
    df = df.iloc[args.shard::args.nshards]
    os.makedirs(args.out, exist_ok=True)
    fn = os.path.join(args.out, f"shard{args.shard}of{args.nshards}{args.file_tag}.jsonl")
    done = set(json.loads(l)["question_id"] for l in open(fn) if l.endswith("\n")) if os.path.exists(fn) else set()
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
    tok, model, proc = RV.load_llava(args)
    with open(fn, "a") as f:
        for _, r in df.iterrows():
            if r.question_id in done:
                continue
            doc = r.to_dict()
            if args.bench == "videomme":
                doc["options"] = list(doc["options"])
            while not os.path.exists(os.path.join(RV.CACHE, f"{doc['videoID']}_n{RV.N_CACHE}", "meta.json")):
                time.sleep(30)
            text, info = answer(doc, args, tok, model, proc)
            info["host"], info["gpu"] = socket.gethostname(), torch.cuda.get_device_name()
            pred = RV.extract_letter(text)
            rec = dict(question_id=doc["question_id"], videoID=doc["videoID"], group=doc.get("group"),
                       task_type=doc.get("task_type"), answer=doc["answer"], output=text, pred=pred,
                       score=float(pred.lower() == str(doc["answer"]).lower()),
                       **{k: doc[k] for k in ("time_reference", "duration_minutes") if k in doc}, **info)
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(doc["question_id"], repr(text), doc["answer"], f"{info['sec']:.1f}s", info["cache_len"], flush=True)


if __name__ == "__main__":
    main()
