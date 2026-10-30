"""Decode latency of eta rule 1 vs the other arms (scripts/latency_rule1.sh).

Reads results/eta_rules/latency/<model>_<arm>[_g<N>].json (per-step decode
times of the timed round, written by FreeKV's source/pred.py --tbt_out) and
plots, per model:
  top left   mean ms per decode step (first step excluded, as pred.py's Avg TBT)
  top right  ms per step vs decode position, smoothed over 256 steps
  bottom panel (a) per model at several sequence lengths (prefixes of the
         timed run), with two extra bars for setting eta: the fixed-eta sweep
         (one decode per grid eta {0, .25, .5, .75, 1}, i.e. 5 x its ms/step)
         and rule 1 (one decode, plus the cost of finding eta* from
         time_rule1_find.py -- calibration traces + endpoint errors and the
         closed form -- divided over that sequence's decode steps; it is
         paid once per (model, task), so this is its worst case)

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
ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384],
                help="sequence lengths for the bottom row (prefixes of the timed run)")
ap.add_argument("--find_task", default="AIME24", help="task of the timed eta* search (time_rule1_find.py)")
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

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.3])
ax0, ax1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
axL = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
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


# ---- bottom row: panel (a) at several sequence lengths, per model. A decode
# of N tokens is the first N steps of the timed run (same prompt, greedy,
# --ignore_eos), so every length comes from the same measurement. The two
# ways of setting eta are added: the fixed-eta sweep decodes the sequence once
# per grid eta (5 x the fixed-eta ms/step), rule 1 once plus the cost of
# finding eta* (time_rule1_find.py) divided over the N steps -- paid once per
# (model, task), so this charges it all to one sequence (its worst case).
GRID = ["0.0", "0.25", "0.5", "0.75", "1.0"]
BARS = ARMS[:5] + [("sweep", "OVAL fixed eta: 5-eta sweep"), ("oval_r1", "OVAL rule 1 (+ finding eta*)")]
COL["sweep"] = "#fdae6b"
print(f"\n{'model':<16}{'N':>7}" + "".join(f"{lab[:15]:>17}" for _, lab in BARS) + "  eta*(ms/step)")
for axm, m in zip(axL, A.models):
    ff = os.path.join(A.dir, f"{m}_{A.find_task}_r1_find.json")
    fd = json.load(open(ff)) if os.path.exists(ff) else None
    find_ms = ((fd["gen_s"] or 0.0) + fd["compute_s"] + fd["load_s"]) * 1e3 if fd else 0.0
    xs = np.arange(len(A.lengths)); wb = 0.8 / len(BARS)
    for bi, (a, lab) in enumerate(BARS):
        src = data.get((m, "oval_fix" if a == "sweep" else a))
        if src is None:
            continue
        t = np.asarray(src["tbt_ms"][1:], float)
        dec = np.array([t[:N - 1].mean() for N in A.lengths])
        y = dec * len(GRID) if a == "sweep" else dec
        xb = xs + (bi - (len(BARS) - 1) / 2) * wb
        axm.bar(xb, y, wb, color=COL[a], label=lab)
        top = y
        if a == "oval_r1":
            eta = np.array([find_ms / (N - 1) for N in A.lengths])
            axm.bar(xb, eta, wb, bottom=y, color=COL[a], alpha=0.4, hatch="//", edgecolor="white",
                    label="  finding eta* (spread over this sequence)")
            top = y + eta
        for xx, yy in zip(xb, top):
            axm.text(xx, yy, f"{yy:.0f}", ha="center", va="bottom", fontsize=6.5)
    for N in A.lengths:
        row = []
        for a, _ in BARS:
            src = data.get((m, "oval_fix" if a == "sweep" else a))
            v = np.asarray(src["tbt_ms"][1:N], float).mean() if src else np.nan
            row.append(v * len(GRID) if a == "sweep" else v)
        print(f"{m:<16}{N:>7}" + "".join(f"{v:>17.1f}" for v in row) + f"  {find_ms / (N - 1):.1f}")
    axm.set_xticks(xs); axm.set_xticklabels([f"{N // 1024}K tokens" for N in A.lengths])
    axm.set_ylabel("ms per decode step (mean)")
    axm.set_title(f"{NAME.get(m, m)}: decode time by sequence length\n"
                  "sweep = 5 decodes (one per eta); rule 1 = 1 decode + finding eta*", fontsize=10)
    axm.legend(fontsize=7.5, ncol=2, loc="upper left"); axm.grid(axis="y", alpha=0.3)
    axm.set_ylim(0, axm.get_ylim()[1] * 1.35)

fig.tight_layout()
os.makedirs(os.path.dirname(A.out), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{A.out}{A.suffix}.{ext}", dpi=150)
print(f"wrote {A.out}{A.suffix}.png/.pdf")
