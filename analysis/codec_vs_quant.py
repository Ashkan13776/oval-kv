"""Is the low-rank basis actually a good KEY CODEC?

The honest comparison is not "rank-r vs raw keys" but "rank-r vs just
quantizing the raw keys", because scalar quantization is what a practitioner
would otherwise do and it is a very strong baseline. This measures both on the
SAME captured decode keys, with the same metric (relative error of the
attention OUTPUT, which is what actually matters), across every layer.

Also measures the shipped int4/int8 record packing, so the rank and
quantization axes can be combined rather than treated as alternatives.

Needs a capture from analysis/capture_kv.py.
"""
import argparse, importlib.util, statistics as st, sys
from pathlib import Path
import torch

from paths import ROOT, FREEKV

ap = argparse.ArgumentParser()
ap.add_argument("--capture", default=str(ROOT / "logs/kv_capture.pt"))
ap.add_argument("--page", type=int, default=32)
ap.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16, 32])
A = ap.parse_args()

def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

basis = _load("page_basis", Path(__file__).parent.parent / "oval/page_basis.py")
summary = _load("page_summary", Path(__file__).parent.parent / "oval/page_summary.py")

cap = torch.load(A.capture)
PAGE = A.page
d = next(iter(cap.values()))[0].shape[-1]

def quant(X, bits, per):
    """symmetric absmax quantization of the RAW keys.
    per='tok': one scale per token per head. per='ch': one per channel per page."""
    qmax = 2 ** (bits - 1) - 1
    dim = -1 if per == "tok" else -2
    sc = X.abs().amax(dim=dim, keepdim=True).clamp_min(1e-8) / qmax
    return (torch.round(X / sc).clamp(-qmax - 1, qmax) * sc)

b_raw = PAGE * d * 2
def b_quant(bits, per): return PAGE * d * bits / 8 + (PAGE * 2 if per == "tok" else d * 2)
def b_bf16(r):          return (d * r + PAGE * r + d) * 2
def b_packed(r):        return (d * r / 2 + r * 2) + (PAGE * r + PAGE * 2) + (d + 2)

rows = {}
for li in sorted(cap):
    K, V, q = (t.cuda() for t in cap[li])
    b, T, nkv, hd = K.shape; g = q.shape[1] // nkv
    P = (T - 1024) // PAGE
    if P < 1: continue
    Kp = K[:, 512:512 + P*PAGE].reshape(b, P, PAGE, nkv, hd)
    Vp = V[:, 512:512 + P*PAGE].reshape(b, P, PAGE, nkv, hd)
    qh = q.reshape(b, nkv, g, hd)
    Vf = Vp.permute(0, 3, 1, 2, 4).reshape(b, nkv, P*PAGE, hd)
    Kph = Kp.permute(0, 1, 3, 2, 4)
    def out_of(Ks):
        Kf = Ks.permute(0, 2, 1, 3, 4).reshape(b, nkv, P*PAGE, hd)
        lg = torch.einsum('bhgd,bhtd->bhgt', qh, Kf) / (hd ** 0.5)
        return torch.einsum('bhgt,bhtd->bhgd', torch.softmax(lg, -1), Vf)
    ot = out_of(Kph)
    def rec(k, Kh): rows.setdefault(k, []).append(
        ((out_of(Kh) - ot).norm() / ot.norm()).item() * 100)
    for bits in (8, 4, 3):
        for per in ("tok", "ch"):
            rec((f"int{bits} raw keys ({per})", b_quant(bits, per)), quant(Kph, bits, per))
    for r in A.ranks:
        mu, bs, cf = basis.build_summary(Kp, Vp, None, 0.0, r)
        eff = bs.shape[-1]
        rec((f"rank-{r} bf16", b_bf16(eff)),
            torch.matmul(cf, bs.transpose(-1, -2)) + mu.unsqueeze(-2))
        mr, vr, cr = summary.quantize(mu.bfloat16(), bs.bfloat16(), cf.bfloat16(), eff)
        m2, b2, c2 = (t.float() for t in summary.dequantize(mr, vr, cr, eff))
        rec((f"rank-{r} packed", b_packed(eff)),
            torch.matmul(c2, b2.transpose(-1, -2)) + m2.unsqueeze(-2))
    del K, V, q, Kp, Vp, Vf
    torch.cuda.empty_cache()

n = len(next(iter(rows.values())))
print(f"attention-output relative error, median over {n} layers\n")
print(f"{'representation':<26}{'B/pg/kvh':>10}{'vs raw':>9}{'median err':>12}")
print(f"{'raw bfloat16 keys':<26}{b_raw:>10.0f}{1.0:>8.2f}x{0.0:>11.1f}%")
for (name, nb), v in sorted(rows.items(), key=lambda kv: -kv[0][1]):
    print(f"{name:<26}{nb:>10.0f}{b_raw/nb:>8.2f}x{st.median(v):>11.1f}%")
print("\nRead the two axes together: at matched error, scalar quantization of")
print("the raw keys is a strong baseline that low-rank alone does not beat.")
print("Rank and quantization COMPOSE -- the packed rows are the combination.")
