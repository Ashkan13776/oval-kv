"""Video QA with Qwen2.5-VL-7B-Instruct inside the FreeKV/OVAL harness (C1 / C2).

The vision side runs once per video in the official HF stack (scripts/revision/C/qwen_prep.py,
transformers 4.51): it stores the stream-protocol prefill (system prompt + video, up to
<|vision_end|>) as inputs_embeds plus its 3-D M-RoPE positions. This script (ovalkv env,
transformers 4.45, the harness) loads Qwen2.5-VL's language model -- a stock Qwen2-7B
architecture (28 layers, 28 query / 4 KV heads, head dim 128, q/k/v biases) whose weights
sit under model.* / lm_head in the same checkpoint -- as Qwen2ForCausalLM, runs the
harness's eval.util.load_model_and_tokenizer on it unchanged, and replaces the harness's
RoPE helper (kvc.patch.{llama,tuple_kv_cache}.apply_rope_inplace) with an M-RoPE one:

  prefill  q, k rotated with the stored 3-D positions, sections [16, 24, 24] (x2) of the
           128 channels for (t, h, w) -- HF's apply_multimodal_rotary_pos_emb, bf16 cos/sin.
  decode   text tokens have t = h = w, so M-RoPE equals 1-D RoPE at the Qwen text
           position (next_pos + j), which is not the cache index; applied with flashinfer's
           apply_rope_inplace at that offset, as the harness does for every other model.

The cache therefore holds post-M-RoPE keys, which is what the page bases are built on.
The question text, the end-of-turn and the assistant header stream through the decode path
(query-agnostic), then greedy decoding, max 16 new tokens, lmms-eval's MCQ extraction.

    python scripts/revision/C/run_qwen25vl.py --bench videomme --durations long --out DIR <harness flags>
"""
import argparse, json, os, socket, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..", "..")
FK = os.path.join(REPO, "third_party", "FreeKV", "accuracy")
sys.path.insert(0, FK)
sys.path.insert(0, os.path.join(REPO, "vision"))
sys.path.insert(0, HERE)
import run_videomme as RV                     # lmms-eval MCQ extractor, harness arg parser
import qwen_prep as QP                        # paths, constants (no HF 4.51 import at module level)

MODEL = QP.MODEL
SECTION = [16, 24, 24]


class MRope:
    """State shared by the patched RoPE helper: prefill cos/sin, decode text position."""
    prefill_cos = None        # (L, 128) bf16
    prefill_sin = None
    pos = None                # int, Qwen position of the next decode token
    theta = 1e6


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def mrope_cos_sin(pos3, head_dim=128, theta=1e6, dtype=torch.bfloat16, device="cuda:0"):
    """HF Qwen2_5_VLRotaryEmbedding + the section merge of apply_multimodal_rotary_pos_emb."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64, device=device).float() / head_dim))
    p = pos3.to(device).float()                                           # (3, L)
    freqs = p[:, :, None] * inv_freq[None, None, :]                       # (3, L, 64), fp32
    emb = torch.cat((freqs, freqs), dim=-1)                               # (3, L, 128)
    cos, sin = emb.cos().to(dtype), emb.sin().to(dtype)
    sec = SECTION * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(sec, dim=-1))], dim=-1)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(sec, dim=-1))], dim=-1)
    return cos, sin                                                       # (L, 128)


def make_rope(orig):
    def apply_rope_inplace(config, q, k, offsets, indptr=None):
        bsz, qlen = q.shape[:2]
        if qlen > 1:
            assert MRope.prefill_cos is not None and MRope.prefill_cos.shape[0] == qlen, \
                "M-RoPE prefill positions do not match the prefill length"
            cos, sin = MRope.prefill_cos[None, :, None, :], MRope.prefill_sin[None, :, None, :]
            q.copy_(q * cos + _rotate_half(q) * sin)
            k.copy_(k * cos + _rotate_half(k) * sin)
            return
        off = torch.full_like(offsets, MRope.pos)
        orig(config, q, k, off, indptr)
    return apply_rope_inplace


def load_lm():
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL)
    vc = json.load(open(os.path.join(path, "config.json")))
    cfg = Qwen2Config(vocab_size=vc["vocab_size"], hidden_size=vc["hidden_size"],
                      intermediate_size=vc["intermediate_size"], num_hidden_layers=vc["num_hidden_layers"],
                      num_attention_heads=vc["num_attention_heads"], num_key_value_heads=vc["num_key_value_heads"],
                      hidden_act=vc["hidden_act"], max_position_embeddings=vc["max_position_embeddings"],
                      rms_norm_eps=vc["rms_norm_eps"], rope_theta=vc["rope_theta"], rope_scaling=None,
                      tie_word_embeddings=vc["tie_word_embeddings"], use_sliding_window=False,
                      bos_token_id=vc["bos_token_id"], eos_token_id=vc["eos_token_id"], torch_dtype="bfloat16")
    model = Qwen2ForCausalLM.from_pretrained(path, config=cfg, torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa", device_map="cuda:0")
    model.eval()
    tok = AutoTokenizer.from_pretrained(path)
    MRope.theta = vc["rope_theta"]
    return tok, model, path


def patch_harness(args, tok, model, path):
    from eval import util
    import kvc.patch.llama as KL
    import kvc.patch.tuple_kv_cache as KT
    class _Gen:
        eos_token_id = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    orig = (util.AutoTokenizer.from_pretrained, util.AutoModelForCausalLM.from_pretrained,
            util.GenerationConfig.from_pretrained)
    util.AutoTokenizer.from_pretrained = staticmethod(lambda *a, **k: tok)
    util.AutoModelForCausalLM.from_pretrained = staticmethod(lambda *a, **k: model)
    util.GenerationConfig.from_pretrained = staticmethod(lambda *a, **k: _Gen())
    try:
        model, _, step_updater, eos_ids, config = util.load_model_and_tokenizer(path, args)
    finally:
        (util.AutoTokenizer.from_pretrained, util.AutoModelForCausalLM.from_pretrained,
         util.GenerationConfig.from_pretrained) = orig
    for mod in (KL, KT):
        if hasattr(mod, "apply_rope_inplace"):
            mod.apply_rope_inplace = make_rope(mod.apply_rope_inplace)
    return model, step_updater, eos_ids


def stream_text(tok, doc_text):
    """Tokens after <|vision_end|>: question text, end of user turn, assistant header."""
    return tok(doc_text + "<|im_end|>\n<|im_start|>assistant\n").input_ids


@torch.no_grad()
def answer(doc, args, tok, model, step_updater, eos_ids):
    ep = QP.emb_path(doc["videoID"], args.bench, args.frames)
    while not os.path.exists(ep):          # the vision side (qwen_prep.py) may still be running
        time.sleep(30)
    c = torch.load(ep, weights_only=True)
    emb = c["emb"][None].to("cuda:0")
    L = emb.shape[1]
    MRope.prefill_cos, MRope.prefill_sin = mrope_cos_sin(c["pos"], theta=MRope.theta)
    stream = stream_text(tok, doc["text"])
    t0 = time.time()
    if step_updater is not None:
        step_updater.reset(c["ids"][None].to("cuda:0"))
    out = model(inputs_embeds=emb, past_key_values=None, use_cache=True)
    pkv = out.past_key_values
    MRope.prefill_cos = MRope.prefill_sin = None
    if os.environ.get("QWEN_DUMP_KEYS"):          # unit test (test_mrope.py): cached prefill keys
        torch.save({l: pkv[l][0][:1].cpu() for l in (1, 14, 27)}, os.environ["QWEN_DUMP_KEYS"])
    t_prefill = time.time() - t0
    pos = c["next_pos"]
    for t in stream:
        MRope.pos = pos
        out = model(input_ids=torch.tensor([[t]], device="cuda:0"), past_key_values=pkv, use_cache=True)
        pkv, pos = out.past_key_values, pos + 1
    top2 = out.logits[0, -1].float().topk(2)
    margin = (top2.values[0] - top2.values[1]).item()
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    gen = []
    for _ in range(args.max_new_tokens):
        gen.append(nxt.item())
        if gen[-1] in eos_ids:
            break
        MRope.pos = pos
        out = model(input_ids=nxt, past_key_values=pkv, use_cache=True)
        pkv, pos = out.past_key_values, pos + 1
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    text = tok.decode(gen, skip_special_tokens=True).strip()
    ms = torch.cuda.memory_stats()
    info = dict(frames_sha1=c["frames_sha1"], pixels_sha1=c["pixels_sha1"], ids_sha1=c["ids_sha1"],
                n_visual=c["n_visual"], resized_hw=c["resized_hw"], prefill_len=L, n_stream=len(stream),
                n_gen=len(gen), margin=margin, sec_prefill=t_prefill, sec=time.time() - t0,
                peak_gb=torch.cuda.max_memory_allocated() / 2**30, alloc_retries=ms.get("num_alloc_retries", 0))
    return text, info


def main():
    ap = RV.build_parser()
    for a in ap._actions:
        if a.dest == "protocol":
            a.required, a.default = False, "stream"
    ap.add_argument("--bench", default="videomme", choices=["videomme", "lvbench"])
    args = ap.parse_args()
    import qwen_docs
    df = qwen_docs.load_docs(args)
    if args.limit:
        df = df.iloc[:args.limit]
    if args.qids:
        df = df[df.question_id.isin(args.qids)]
    df = df.iloc[args.shard::args.nshards]
    os.makedirs(args.out, exist_ok=True)
    fn = os.path.join(args.out, f"shard{args.shard}of{args.nshards}{args.file_tag}.jsonl")
    done = set()
    if os.path.exists(fn):
        for l in open(fn):
            if l.endswith("\n"):
                done.add(json.loads(l)["question_id"])
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
    tok, model, path = load_lm()
    model, step_updater, eos_ids = patch_harness(args, tok, model, path)
    with open(fn, "a") as f:
        for _, r in df.iterrows():
            if r.question_id in done:
                continue
            doc = r.to_dict()
            text, info = answer(doc, args, tok, model, step_updater, eos_ids)
            info["host"], info["gpu"] = socket.gethostname(), torch.cuda.get_device_name()
            pred = RV.extract_letter(text)
            rec = dict(question_id=doc["question_id"], videoID=doc["videoID"], group=doc.get("group"),
                       task_type=doc.get("task_type"), answer=doc["answer"], output=text, pred=pred,
                       score=float(pred.lower() == str(doc["answer"]).lower()),
                       **{k: doc[k] for k in ("time_reference", "duration_minutes") if k in doc}, **info)
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(doc["question_id"], repr(text), doc["answer"], f"{info['sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
