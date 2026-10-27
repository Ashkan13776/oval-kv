"""Does the output-aware VALUE basis reduce the error that reaches the model?

Measured on real decode state from a reasoning trace: post-RoPE keys and values
from the cache, the last NQ queries re-rotated at their own positions, and each
layer's o_proj. Keys stay exact, so only the value codec moves the output. The
metric is the relative error of  u = sum_i W_i o_i,  the attention block's
contribution to the residual stream -- the quantity the derivation in
oval/page_basis.py::build_value_summary minimises -- and, for reference, the
plain relative error of the head outputs o_i.

Two attention sets:
  all    every page attended (sink and recent exact)
  top64  the 64 pages with the largest true attention mass per (query, KV head),
         the B=2048 / page 32 budget -- the pages the harness actually rebuilds

Usage:  python analysis/value_codec_check.py [--ranks 8 16 31] [--etas ...]
"""
import argparse, importlib.util, json, os, statistics as st
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from paths import ROOT, FREEKV

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
ap.add_argument("--ngen", type=int, default=4200)
ap.add_argument("--nq", type=int, default=128)
ap.add_argument("--page", type=int, default=32)
ap.add_argument("--sink", type=int, default=512)
ap.add_argument("--recent", type=int, default=512)
ap.add_argument("--topk", type=int, default=64)
ap.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 31])
ap.add_argument("--etas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0],
                help="value-basis eta_V")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default=str(ROOT / "logs/value_codec_check.json"))
A = ap.parse_args()

_s = importlib.util.spec_from_file_location(
    "page_basis", Path(__file__).parent.parent / "oval/page_basis.py")
pb = importlib.util.module_from_spec(_s); _s.loader.exec_module(pb)

torch.manual_seed(A.seed)
tok = AutoTokenizer.from_pretrained(A.model)
model = AutoModelForCausalLM.from_pretrained(
    A.model, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
cfg = model.config
nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
hd = cfg.hidden_size // nh; g = nh // nkv

prob = json.loads(open(FREEKV / "accuracy/eval/reasoning/datasets/math50.jsonl").readline())
msgs = [{"role": "user", "content": prob["problem"] +
         " Please reason step by step, and put your final answer within \\boxed{}.\n"}]
ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
          return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    seq = model.generate(ids, max_new_tokens=A.ngen, min_new_tokens=A.ngen, do_sample=True,
                         temperature=0.6, top_p=0.95, pad_token_id=tok.eos_token_id)
T = seq.shape[-1]
print(f"sequence {T} tokens", flush=True)

# one forward over the whole trace: post-RoPE K/V from the cache, pre-RoPE q
# from q_proj (rotated below at its own positions)
qcap = {}
for li, layer in enumerate(model.model.layers):
    layer.self_attn.q_proj.register_forward_hook(
        lambda m, i, o, li=li: qcap.__setitem__(li, o[:, -A.nq:].detach()))
with torch.no_grad():
    fw = model(seq, use_cache=True)
pkv = fw.past_key_values
pos = torch.arange(T - A.nq, T, device=seq.device)[None]
cos, sin = model.model.rotary_emb(qcap[0], pos)

P = (T - A.sink - A.recent) // A.page
lo, hi = A.sink, A.sink + P * A.page
res = {}
for li, layer in enumerate(model.model.layers):
    Kc, Vc = pkv[li][0][0].float(), pkv[li][1][0].float()        # [nkv,T,hd]
    q = qcap[li][0].view(A.nq, nh, hd).transpose(0, 1).float()     # [nh,nq,hd]
    q, _ = apply_rotary_pos_emb(q[None], q[None], cos, sin)
    q = q[0].view(nkv, g, A.nq, hd)
    W = layer.self_attn.o_proj.weight.float()                      # [D, nh*hd]
    Wh = W.view(W.shape[0], nkv, g, hd)
    Gh = pb.publish_metric(layer.self_attn.o_proj.weight, nkv, hd, g)   # the keys' G^{1/2}
    Ghi = pb.metric_inv_sqrt(Gh)
    # [B=1,P,page,H,d], the harness layout
    Kp = Kc[:, lo:hi].transpose(0, 1).reshape(P, A.page, nkv, hd)[None]
    Vp = Vc[:, lo:hi].transpose(0, 1).reshape(P, A.page, nkv, hd)[None]

    lg = torch.einsum('hgqd,htd->hgqt', q, Kc[:, :T]) / hd ** 0.5
    # causal: query at position T-nq+j sees keys <= its position
    qpos = torch.arange(T - A.nq, T, device=lg.device)[:, None]
    lg = lg.masked_fill(torch.arange(T, device=lg.device)[None] > qpos, float('-inf'))
    p = torch.softmax(lg, -1)                                       # [H,g,nq,T]
    # top-k pages by true attention mass, averaged over the GQA group (avgSM-like)
    pm = p[..., lo:hi].reshape(nkv, g, A.nq, P, A.page).sum(-1).mean(1)   # [H,nq,P]
    keep = torch.zeros_like(pm, dtype=torch.bool)
    keep.scatter_(-1, pm.topk(min(A.topk, P), -1).indices, True)
    tokmask = torch.ones(nkv, A.nq, T, dtype=torch.bool, device=lg.device)
    tokmask[..., lo:hi] = keep.repeat_interleave(A.page, -1)
    p_top = torch.softmax(lg.masked_fill(~tokmask[:, None], float('-inf')), -1)

    def outputs(Vfull, pp):
        o = torch.einsum('hgqt,htd->hgqd', pp, Vfull)               # [H,g,nq,d]
        u = torch.einsum('mhgd,hgqd->qm', Wh, o)                     # [nq,D]
        return o, u
    ref = {k: outputs(Vc, pp) for k, pp in (("all", p), ("top64", p_top))}
    for r in A.ranks:
        for eta in A.etas:
            mu, bs, cf = pb.build_value_summary(Kp, Vp, Gh, Ghi, eta, r)
            Vh = (cf.float() @ bs.float().transpose(-1, -2) + mu.float().unsqueeze(-2))
            Vh = Vh[0].permute(1, 0, 2, 3).reshape(nkv, P * A.page, hd)
            Vx = Vc.clone(); Vx[:, lo:hi] = Vh
            for k, pp in (("all", p), ("top64", p_top)):
                o, u = outputs(Vx, pp)
                o0, u0 = ref[k]
                res.setdefault((k, r, eta), []).append(
                    ((u - u0).norm() / u0.norm()).item() * 100)
                res.setdefault((k + "/o", r, eta), []).append(
                    ((o - o0).norm() / o0.norm()).item() * 100)
    if li % 7 == 0:
        print(f"layer {li} done", flush=True)

print(f"\nrelative error (%), median over {len(model.model.layers)} layers, "
      f"{A.nq} queries, {P} pages; keys exact\n")
for k in ("all", "top64"):
    for metric, tag in ((k, "residual u = sum_i W_i o_i"), (k + "/o", "head outputs o")):
        print(f"[{k}] {tag}")
        print("  rank " + "".join(f"  eta={e:<5}" for e in A.etas))
        for r in A.ranks:
            print(f"  {r:>4} " + "".join(f"  {st.median(res[(metric, r, e)]):>9.3f}"
                                           for e in A.etas))
        print()
Path(A.out).parent.mkdir(parents=True, exist_ok=True)
json.dump({f"{k}|{r}|{e}": v for (k, r, e), v in res.items()}, open(A.out, "w"))
print(f"per-layer values: {A.out}")
