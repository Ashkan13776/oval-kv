"""Decode latency of eta rule 1 vs the other arms (scripts/latency_rule1.sh).

Reads results/eta_rules/latency/<model>_<arm>[_g<N>].json (per-step decode
times of the timed round, written by FreeKV's source/pred.py --tbt_out) and
plots, per model:
  left   mean ms per decode step (first step excluded, as pred.py's Avg TBT)
  middle ms per step vs decode position, smoothed over 256 steps
  right  the one-time cost of CHOOSING eta, in decoded tokens: the fixed-eta
         arm needs an accuracy sweep over the grid {0, .25, .5, .75, 1} x 3
         tasks x 8 seeds (tokens counted from results/codec when the grid is
         there, else estimated as 5 x the model's rule-1 cells), rule 1 only
         the calibration traces it is fitted on (ncalib x ngen per task, from
         results/eta_rules/<model>_<task>.json), with no labels or scoring

  python analysis/plot_latency_rule1.py [--suffix _g256] [--out ...]
"""
import argparse, glob, json, os, sys
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
ap.add_argument("--codec_dir", default=str(ROOT / "results/codec"))
ap.add_argument("--tasks", nargs="+", default=["MATH50", "AIME24", "GPQA50c"])
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

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 4.8), gridspec_kw={"width_ratios": [1, 1.4, 0.8]})
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


def _toks(files):
    return sum(json.loads(l)["output_len"] for f in files for l in open(f) if l.strip())


GRID = ["0.0", "0.25", "0.5", "0.75", "1.0"]
print(f"{'model':<16}{'sweep tokens':>16}{'rule-1 tokens':>16}   (one-time eta selection)")
cost = {}
for m in A.models:
    d = os.path.join(A.codec_dir, f"{m}-spec_ret")
    sweep, est = 0, False
    for t in A.tasks:
        cells = [glob.glob(f"{d}/{t}-*-oval_r8_eta{e}_q0_recon1-pQ2*.jsonl") for e in GRID]
        if all(len(c) == 8 for c in cells):
            sweep += sum(_toks(c) for c in cells)
        else:                                     # grid not on disk: 5 x the rule-1 cell
            tab = json.load(open(ROOT / f"results/eta_rules/tables/{m}_{t}_r1.json"))
            sweep += len(GRID) * _toks(glob.glob(f"{d}/{t}-*_etab{tab['hash']}_*.jsonl"))
            est = True
    calib = 0
    for t in A.tasks:
        c = json.load(open(ROOT / f"results/eta_rules/{m}_{t}.json"))["config"]
        calib += c["ncalib"] * c["ngen"]
    cost[m] = (sweep, calib, est)
    print(f"{m:<16}{sweep:>16,}{calib:>16,}" + ("   (sweep estimated)" if est else ""))
w2 = 0.38
for j, m in enumerate(A.models):
    sw, ca, est = cost[m]
    for dx, y, a, lab in ((-w2 / 2, sw, "oval_fix", "fixed eta: accuracy sweep"),
                          (w2 / 2, ca, "oval_r1", "rule 1: calibration traces")):
        ax2.bar(j + dx, y, w2, color=COL[a], label=lab if j == 0 else None,
                hatch="//" if (a == "oval_fix" and est) else None, edgecolor="white")
        ax2.text(j + dx, y, f"{y / 1e6:.2f}M" if y >= 1e6 else f"{y / 1e3:.0f}K",
                 ha="center", va="bottom", fontsize=8)
    ax2.text(j, sw * 2.2, f"{sw / ca:.0f}x", ha="center", fontsize=10, fontweight="bold")
ax2.set_yscale("log"); ax2.set_ylim(1e4, max(v[0] for v in cost.values()) * 80)
ax2.set_xticks(range(len(A.models)))
ax2.set_xticklabels([NAME.get(m, m) + ("\n(sweep est.)" if cost[m][2] else "") for m in A.models])
ax2.set_ylabel("decoded tokens (log)")
ax2.set_title("One-time cost of choosing eta (3 tasks)\nsweep: 5 etas x 8 seeds, scored; rule 1: 4 traces/task",
              fontsize=10)
ax2.legend(fontsize=8, loc="upper center", ncol=2); ax2.grid(axis="y", alpha=0.3, which="both")

fig.tight_layout()
os.makedirs(os.path.dirname(A.out), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{A.out}{A.suffix}.{ext}", dpi=150)
print(f"wrote {A.out}{A.suffix}.png/.pdf")
