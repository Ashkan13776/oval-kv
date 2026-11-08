"""C3 cache-level baselines for Qwen2.5-VL-7B, computed in HF (transformers 4.51, ~/qwenvl) on the
stored stream-protocol prefill (qwen_prep.py), then the question streamed and greedy decoding as in
qwen_official.py --mode hfstream (which reproduces the harness's FullKV answers 100/100).
Unit = one Qwen temporal group (2 frames, 405 tokens at our resolution) = one "frame" upstream.

--method infinipot   InfiniPot-V (aiha-lab/InfiniPot-V, kvcache_utils.py, ported):
    block-wise prefill: first block = B groups, later blocks B - C new groups; after every block
    but the last, each layer's cache (system prefix kept) is compressed to C * 405 tokens, per KV
    head: TaR = -(cos sim of post-RoPE keys at the same spatial index between each of the newest
    int(cur_groups * 0.25) "query" groups and every older position block), mean over query groups;
    query groups always kept; top (tar_budget - query_len) TaR picks; tar_ratio 0.5; the rest by
    value norm (TaR picks forced in with max + 1)  (kvcache_utils.py:104-226). Exactly as upstream,
    after the first compression the kept older tokens are reshaped into pseudo-frames of 405.
    The final cache is <= B groups + system + <|vision_end|>. Question excluded from compression.
--method rekv        ReKV (Becomebright/ReKV, kv_cache_manager.py, retrieval step ported):
    one block per temporal group; representative key = mean of the PRE-RoPE keys of the block
    (k_proj output), GQA-expanded and flattened over heads; query = mean of the pre-RoPE queries
    of the question tokens, flattened; per layer one dot product, top-k groups; retrieved once
    per question. Retrieved cache = 128 sink + k groups + the last 128 prefill tokens, keys with
    their ORIGINAL M-RoPE rotation (ReKV re-applies contiguous 1-D RoPE; no M-RoPE equivalent).
    DEVIATION: the question queries come from one dense pass of the question over the full
    cache (then discarded), not from ReKV's layer-by-layer retrieval pass.

    ~/qwenvl/bin/python scripts/revision/C/qwen_c3.py --method infinipot --block 16 --compress 12 ...
"""
import argparse, json, os, socket, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qwen_prep as QP
import qwen_official as QO

VIDEO_TOKEN = 151656


def kv(pkv, l):
    return pkv.key_cache[l], pkv.value_cache[l]


def set_kv(pkv, l, k, v):
    pkv.key_cache[l], pkv.value_cache[l] = k, v


def infinipot_compress(pkv, sys_n, keep_groups, tpf, cur_groups, tar_ratio=0.5, query_ratio=0.25):
    for l in range(len(pkv.key_cache)):
        K, V = kv(pkv, l)
        H = K.shape[1]
        kc, vc = K[:, :, sys_n:], V[:, :, sys_n:]
        n = kc.shape[2]
        budget = keep_groups * tpf
        attn_budget = round((1 - tar_ratio) * budget)
        tar_budget = budget - attn_budget
        qf = int(cur_groups * query_ratio)
        ql = qf * tpf
        assert tar_budget - ql >= 0, "query groups exceed the TaR budget"
        qn = kc[:, :, -ql:].float(); qn = qn / (qn.norm(dim=-1, keepdim=True) + 1e-9)
        kn = kc[:, :, :-ql].float(); kn = kn / (kn.norm(dim=-1, keepdim=True) + 1e-9)
        qn = qn.reshape(1, H, qf, tpf, -1)
        kn = kn.reshape(1, H, -1, tpf, kn.shape[-1])
        score = torch.zeros(1, H, kn.shape[2], tpf, device=K.device)
        for i in range(qf):                                     # mean over query groups, memory-light
            score -= (qn[:, :, i:i + 1] * kn).sum(-1) / qf
        score = score.reshape(1, H, -1)
        sel = score.topk(tar_budget - ql, dim=-1).indices[0]
        recent = torch.arange(n - ql, n, device=K.device).unsqueeze(0).expand(H, -1)
        tar_idx = torch.cat([sel, recent], -1)
        vn = vc.norm(dim=-1)[0].float()                          # [H, n]
        hi = torch.arange(H, device=K.device).unsqueeze(1).expand(-1, tar_idx.shape[1])
        vn[hi, tar_idx] = vn.max() + 1
        idx = vn.topk(budget, dim=-1).indices.sort(-1).values    # [H, budget]
        hh = torch.arange(H, device=K.device).unsqueeze(1).expand(-1, budget)
        nk = kc[0][hh, idx].unsqueeze(0); nv = vc[0][hh, idx].unsqueeze(0)
        set_kv(pkv, l, torch.cat([K[:, :, :sys_n], nk], 2), torch.cat([V[:, :, :sys_n], nv], 2))


@torch.no_grad()
def prefill_infinipot(model, c, args):
    dev = model.device
    emb, pos = c["emb"][None].to(dev), c["pos"][:, None].to(dev).long()
    vis = (c["ids"] == VIDEO_TOKEN).nonzero().flatten()
    sys_n, G = int(vis[0]), int(c["grid_thw"][0])
    tpf = int(len(vis)) // G
    assert tpf * G == len(vis)
    B, C = args.block, args.compress
    bounds, g = [], 0
    while g < G:                                           # group ranges per block
        step = B if g == 0 else B - C
        bounds.append((g, min(G, g + step))); g += step
    pkv, cur = None, 0
    for bi, (g0, g1) in enumerate(bounds):
        a = 0 if bi == 0 else sys_n + g0 * tpf
        b = sys_n + g1 * tpf if bi < len(bounds) - 1 else emb.shape[1]   # last block: + <|vision_end|>
        logits, pkv = QO.lm_forward(model, inputs_embeds=emb[:, a:b], position_ids=pos[:, :, a:b], past_key_values=pkv)
        cur += g1 - g0
        if bi < len(bounds) - 1 and cur > C:
            infinipot_compress(pkv, sys_n, C, tpf, cur)
            cur = C
    return logits, pkv, dict(cache_len=int(pkv.key_cache[0].shape[2]), blocks=len(bounds), tpf=tpf)


@torch.no_grad()
def prefill_rekv(model, c, args, tok, doc):
    dev = model.device
    layers = model.model.layers
    cap = {}
    hooks = [l.self_attn.k_proj.register_forward_hook(lambda m, i, o, li=li: cap.__setitem__(("k", li), o[0]))
             for li, l in enumerate(layers)]
    emb, pos = c["emb"][None].to(dev), c["pos"][:, None].to(dev).long()
    logits, pkv = QO.lm_forward(model, inputs_embeds=emb, position_ids=pos)
    for h in hooks:
        h.remove()
    vis = (c["ids"] == VIDEO_TOKEN).nonzero().flatten()
    v0, G = int(vis[0]), int(c["grid_thw"][0])
    tpf = len(vis) // G
    L = emb.shape[1]
    nkv, hd = model.config.num_key_value_heads, model.config.hidden_size // model.config.num_attention_heads
    g = model.config.num_attention_heads // nkv
    rep = {li: cap[("k", li)][v0:v0 + G * tpf].float().view(G, tpf, nkv, hd).mean(1) for li in range(len(layers))}
    del cap
    # question queries: one dense pass of the question over the full cache, then cropped away
    qcap = {}
    hooks = [l.self_attn.q_proj.register_forward_hook(lambda m, i, o, li=li: qcap.__setitem__(li, o[0].float()))
             for li, l in enumerate(layers)]
    q_ids = tok(doc["question_text"]).input_ids
    p = torch.arange(c["next_pos"], c["next_pos"] + len(q_ids), device=dev).view(1, 1, -1).expand(3, 1, -1)
    QO.lm_forward(model, input_ids=torch.tensor([q_ids], device=dev), position_ids=p, past_key_values=pkv)
    for h in hooks:
        h.remove()
    pkv.crop(L)
    sink, recent, k = args.sink, args.recent, args.topk
    picks = []
    for li in range(len(layers)):
        qm = qcap[li].mean(0).view(nkv, g, hd)                            # mean over question tokens
        sc = torch.einsum("khd,Gkd->G", qm, rep[li])                      # flattened dot product
        top = sc.topk(min(k, G)).indices.sort().values
        tok_idx = (v0 + top[:, None] * tpf + torch.arange(tpf, device=dev)[None]).flatten()
        tok_idx = tok_idx[(tok_idx >= sink) & (tok_idx < L - recent)]
        idx = torch.cat([torch.arange(sink, device=dev), tok_idx, torch.arange(L - recent, L, device=dev)])
        K, V = kv(pkv, li)
        set_kv(pkv, li, K[:, :, idx], V[:, :, idx])
        picks.append(top.tolist())
    return logits, pkv, dict(cache_len=int(pkv.key_cache[0].shape[2]), rekv_groups=picks, tpf=tpf)


@torch.no_grad()
def answer(doc, args, model, tok, eos_ids):
    ep = QP.emb_path(doc["videoID"], args.bench)
    while not os.path.exists(ep):
        time.sleep(30)
    c = torch.load(ep, weights_only=True)
    t0 = time.time()
    if args.method == "infinipot":
        logits, pkv, extra = prefill_infinipot(model, c, args)
    else:
        logits, pkv, extra = prefill_rekv(model, c, args, tok, doc)
    dev = model.device
    pos = c["next_pos"]
    stream = tok(doc["text"] + "<|im_end|>\n<|im_start|>assistant\n").input_ids
    for t in stream:
        p = torch.full((3, 1, 1), pos, device=dev, dtype=torch.long)
        logits, pkv = QO.lm_forward(model, input_ids=torch.tensor([[t]], device=dev), position_ids=p, past_key_values=pkv)
        pos += 1
    top2 = logits[0, -1].float().topk(2)
    gen, nxt = [], int(logits[0, -1].argmax())
    for _ in range(args.max_new_tokens):
        gen.append(nxt)
        if nxt in eos_ids:
            break
        p = torch.full((3, 1, 1), pos, device=dev, dtype=torch.long)
        logits, pkv = QO.lm_forward(model, input_ids=torch.tensor([[nxt]], device=dev), position_ids=p, past_key_values=pkv)
        pos += 1
        nxt = int(logits[0, -1].argmax())
    text = tok.decode(gen, skip_special_tokens=True).strip()
    info = dict(frames_sha1=c["frames_sha1"], pixels_sha1=c["pixels_sha1"], ids_sha1=c["ids_sha1"],
                n_visual=c["n_visual"], prefill_len=c["emb"].shape[0], n_stream=len(stream), n_gen=len(gen),
                margin=(top2.values[0] - top2.values[1]).item(), sec=time.time() - t0,
                peak_gb=torch.cuda.max_memory_allocated() / 2**30, **extra)
    return text, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["infinipot", "rekv"], required=True)
    ap.add_argument("--block", type=int, default=16, help="InfiniPot-V: groups per block (final cache size)")
    ap.add_argument("--compress", type=int, default=12, help="InfiniPot-V: groups kept after each compression")
    ap.add_argument("--topk", type=int, default=4, help="ReKV: retrieved groups")
    ap.add_argument("--sink", type=int, default=128)
    ap.add_argument("--recent", type=int, default=128)
    ap.add_argument("--bench", default="videomme", choices=["videomme", "lvbench"])
    ap.add_argument("--durations", nargs="+", default=["short", "medium", "long"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--qids", nargs="+")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--file_tag", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import importlib.util
    lmms = os.path.join(HERE, "..", "..", "..", "third_party", "lmms-eval")
    spec = importlib.util.spec_from_file_location("mcq", os.path.join(lmms, "lmms_eval", "tasks", "_task_utils", "mcq_extract.py"))
    mcq = importlib.util.module_from_spec(spec); spec.loader.exec_module(mcq)
    import qwen_docs
    df = qwen_docs.load_docs(args)
    # ReKV's retrieval query = the raw question (llava_onevision_rekv.py:44-51): question + options
    df["question_text"] = df["question"] if args.bench == "lvbench" else \
        [q + "\n" + "\n".join(list(o)) for q, o in zip(df.question, df.options)]
    if args.limit:
        df = df.iloc[:args.limit]
    if args.qids:
        df = df[df.question_id.isin(args.qids)]
    df = df.iloc[args.shard::args.nshards]
    os.makedirs(args.out, exist_ok=True)
    fn = os.path.join(args.out, f"shard{args.shard}of{args.nshards}{args.file_tag}.jsonl")
    done = set(json.loads(l)["question_id"] for l in open(fn) if l.endswith("\n")) if os.path.exists(fn) else set()
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
    proc, model = QP.load_model()
    tok = proc.tokenizer
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    with open(fn, "a") as f:
        for _, r in df.iterrows():
            if r.question_id in done:
                continue
            doc = r.to_dict()
            text, info = answer(doc, args, model, tok, eos_ids)
            info["host"], info["gpu"] = socket.gethostname(), torch.cuda.get_device_name()
            pred = mcq.extract_mcq_answer(text, choices=["A", "B", "C", "D"])
            rec = dict(question_id=doc["question_id"], videoID=doc["videoID"], group=doc.get("group"),
                       task_type=doc.get("task_type"), answer=doc["answer"], output=text, pred=pred,
                       score=float(pred.lower() == str(doc["answer"]).lower()),
                       **{k: doc[k] for k in ("time_reference", "duration_minutes") if k in doc}, **info)
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(doc["question_id"], repr(text), doc["answer"], f"{info['sec']:.1f}s", info["cache_len"], flush=True)


if __name__ == "__main__":
    main()
