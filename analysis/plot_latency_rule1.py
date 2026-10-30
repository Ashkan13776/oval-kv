"""Decode latency of eta rule 1 vs the other arms (scripts/latency_rule1.sh).

Reads results/eta_rules/latency/<model>_<arm>[_g<N>].json (per-step decode
times of the timed round, written by FreeKV's source/pred.py --tbt_out) and
plots, per model:
  left   mean ms per decode step (first step excluded, as pred.py's Avg TBT)
  right  ms per step vs decode position, smoothed over 256 steps

  python analysis/plot_latency_rule1.py [--suffix _g256] [--out ...]
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT

ARMS = [("full", "Full KV"), ("arkvale", "ArkVale"), ("raas", "RaaS"), ("freekv", "FreeKV"),
        ("oval_fix", "OVAL fixed eta"), ("oval_r1", "OVAL rule 1")]
NAME = {"ds-r1-qwen-7b": "R1-Distill-Qwen-7B", "ds-r1-llama-8b": "R1-Distill-Llama-8B"}
COL = {"full": "#7f7f7f", "arkvale": "#8c564b", "raas": "#9467bd", "freekv": "#1f77b4",
       "oval_fix": "#ff7f0e", "oval_r1": "#d62728"}

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["ds-r1-qwen-7b", "ds-r1-llama-8b"])
ap.add_argument("--dir", default=str(ROOT / "results/eta_rules/latency"))
ap.add_argument("--suffix", default="", help="e.g. _g256 for a short MAX_GEN run")
ap.add_argument("--smooth", type=int, default=256)
ap.add_argument("--out", default=str(ROOT / "results/eta_rules/figures/latency_rule1"))
A = ap.parse_args()

data = {}
for m in A.models:
    for a, _ in ARMS:
        f = os.path.join(A.dir, f"{m}_{a}{A.suffix}.json")
        if os.path.exists(f):
            data[(m, a)] = json.load(open(f))
if not data:
    sys.exit(f"no latency files in {A.dir}")

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1, 1.4]})
w = 0.8 / len(ARMS)
x = np.arange(len(A.models))
print(f"{'model':<16}" + "".join(f"{lab:>16}" for _, lab in ARMS) + "   (ms / decode step)")
for i, (a, lab) in enumerate(ARMS):
    ys = []
    for m in A.models:
        d = data.get((m, a))
        ys.append(np.mean(d["tbt_ms"][1:]) if d else np.nan)
    bars = ax0.bar(x + (i - (len(ARMS) - 1) / 2) * w, ys, w, color=COL[a], label=lab)
    for b, y in zip(bars, ys):
        if np.isfinite(y):
            ax0.text(b.get_x() + b.get_width() / 2, y, f"{y:.1f}", ha="center", va="bottom", fontsize=7.5)
for m in A.models:
    print(f"{m:<16}" + "".join(f"{np.mean(data[(m, a)]['tbt_ms'][1:]) if (m, a) in data else float('nan'):>16.2f}"
                              for a, _ in ARMS))
ntok = sorted({len(d["tbt_ms"]) for d in data.values()})
ax0.set_xticks(x); ax0.set_xticklabels([NAME.get(m, m) for m in A.models])
ax0.set_ylabel("ms per decode step (mean)")
ax0.set_title(f"Decode latency, AIME24 problem 0, {ntok[-1]} decoded tokens, batch 1\n"
              "budget 2048 tokens (incl. 512 sink + 512 recent), A6000", fontsize=10)
ymax = max(np.mean(d["tbt_ms"][1:]) for d in data.values())
ax0.set_ylim(0, ymax * 1.3)                       # headroom for the legend
ax0.legend(fontsize=8, ncol=3, loc="upper center"); ax0.grid(axis="y", alpha=0.3)

m0 = A.models[0]
for a, lab in ARMS:
    d = data.get((m0, a))
    if not d:
        continue
    t = np.asarray(d["tbt_ms"][1:], float)
    k = min(A.smooth, max(1, len(t) // 4))
    sm = np.convolve(t, np.ones(k) / k, mode="valid")
    ax1.plot(np.arange(len(sm)) + k // 2 + 1, sm, color=COL[a], lw=1.4, label=lab)
ax1.set_xlabel("decode step"); ax1.set_ylabel(f"ms per step ({A.smooth}-step mean)")
ax1.set_title(f"{NAME.get(m0, m0)}: per-step decode time over the generation", fontsize=10)
ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

fig.tight_layout()
os.makedirs(os.path.dirname(A.out), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{A.out}{A.suffix}.{ext}", dpi=150)
print(f"wrote {A.out}{A.suffix}.png/.pdf")
