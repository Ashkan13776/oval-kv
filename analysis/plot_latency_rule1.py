"""Decode latency of eta rule 1 vs the other arms (scripts/latency_rule1.sh).

Reads results/eta_rules/latency/<model>_<arm>[_g<N>].json (per-step decode
times of the timed round, written by FreeKV's source/pred.py --tbt_out) and
plots, per model:
  top left   mean ms per decode step (first step excluded, as pred.py's Avg TBT)
             of Full KV, ArkVale, RaaS, FreeKV and the two ways of getting
             OVAL's eta: the oracle (5-eta sweep = 5 x the fixed-eta decode)
             and rule 1 (one decode + computing eta*, spread over the steps)
  top right  the same arms' ms per step vs decode position, smoothed
  bottom     per model, OVAL oracle vs OVAL rule 1 on example sequences
             (results/eta_rules/latency/seq, natural EOS): oracle = sweeping
             the 5 grid etas, i.e. 5 x the measured eta=0 decode; rule 1 = one
             decode + computing eta* (time_rule1_find.py --ncalib 1)

  python analysis/plot_latency_rule1.py [--suffix _g256] [--out ...]
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT

ARMS = [("full", "Full KV"), ("arkvale", "ArkVale"), ("raas", "RaaS"), ("freekv", "FreeKV"),
        ("oval_fix", "OVAL fixed eta"), ("oval_r1", "OVAL rule 1")]
NAME = {"ds-r1-qwen-7b": "R1-Distill-Qwen-7B", "ds-r1-llama-8b": "R1-Distill-Llama-8B"}
# pastel fills; lines use the deeper LINE shade so they stay readable
COL = {"full": "#C8C8C8", "arkvale": "#D9B8A3", "raas": "#CDB9E6", "freekv": "#A9CBEA",
       "oval_fix": "#FFC995", "oval_r1": "#F2A3A8"}
LINE = {"full": "#9A9A9A", "arkvale": "#B98A6E", "raas": "#A58BCB", "freekv": "#6FA3D2",
        "oval_fix": "#F5A25D", "oval_r1": "#E06C75"}
EDGE = "#5A5A5A"
SHORT = {"ds-r1-qwen-7b": "Qwen-7B", "ds-r1-llama-8b": "Llama-8B"}
plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 13,
                     "legend.fontsize": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "hatch.color": "#B05A62", "hatch.linewidth": 0.8})
VAL = 11.5                                        # value labels on bars


def panel_letter(ax, s):
    ax.text(-0.09, 1.02, f"({s})", transform=ax.transAxes, fontsize=17, fontweight="bold", va="bottom")

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["ds-r1-qwen-7b", "ds-r1-llama-8b"])
ap.add_argument("--dir", default=str(ROOT / "results/eta_rules/latency"))
ap.add_argument("--suffix", default="", help="e.g. _g256 for a short MAX_GEN run")
ap.add_argument("--smooth", type=int, default=256)
ap.add_argument("--probs", type=int, nargs="+", default=[0, 8, 1],
                help="AIME24 problems of the bottom row (scripts/latency_oracle_r1.sh)")
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

fig = plt.figure(figsize=(17, 11.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.3])
ax0, ax1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
axL = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
# ---- top row. OVAL with a fixed eta and with rule 1 decode at the same speed
# (eta only enters the page-record build), so the two OVAL arms shown are the
# ways of GETTING eta: the oracle sweeps the 5 grid etas {0, .25, .5, .75, 1},
# i.e. decodes the sequence 5 times (5 x the measured fixed-eta decode); rule 1
# decodes once, plus computing eta* from one trace (time_rule1_find.py
# --ncalib 1), whose total is divided over the sequence's decode steps.
NGRID = 5
TOP = ARMS[:4] + [("oracle", "OVAL oracle (5-$\\eta$ sweep)"), ("oval_r1", "OVAL rule 1 (+ $\\eta^*$)")]
COL["oracle"] = COL["oval_fix"]; LINE["oracle"] = LINE["oval_fix"]


def eta_star_s(m):
    ff = os.path.join(A.dir, f"{m}_{A.find_task}_r1_find_n1.json")
    return (lambda d: d["compute_s"] + d["load_s"])(json.load(open(ff))) if os.path.exists(ff) else 0.0


def steps(m, a):
    """per-step ms of an arm of the top row (None if not measured)."""
    d = data.get((m, "oval_fix" if a == "oracle" else a))
    if d is None:
        return None
    t = np.asarray(d["tbt_ms"][1:], float)
    if a == "oracle":
        return NGRID * t
    if a == "oval_r1":
        return t + eta_star_s(m) * 1e3 / len(t)
    return t


w = 0.8 / len(TOP)
x = np.arange(len(A.models))
print(f"{'model':<16}" + "".join(f"{lab[:15]:>17}" for _, lab in TOP) + "   (ms / decode step)")
for i, (a, lab) in enumerate(TOP):
    ys = [np.mean(steps(m, a)) if steps(m, a) is not None else np.nan for m in A.models]
    xb = x + (i - (len(TOP) - 1) / 2) * w
    if a == "oval_r1":
        dec = [np.mean(data[(m, a)]["tbt_ms"][1:]) for m in A.models]
        ax0.bar(xb, dec, w, color=COL[a], edgecolor=EDGE, lw=0.6, label="OVAL rule 1: decode")
        ax0.bar(xb, np.array(ys) - np.array(dec), w, bottom=dec, color="#FBE1E3", hatch="///",
                edgecolor=EDGE, lw=0.6, label="OVAL rule 1: compute $\\eta^*$")
    else:
        ax0.bar(xb, ys, w, color=COL[a], edgecolor=EDGE, lw=0.6, label=lab)
    for xx, y in zip(xb, ys):
        if np.isfinite(y):
            ax0.text(xx, y + 2, f"{y:.0f}", ha="center", va="bottom", fontsize=VAL)
for m in A.models:
    print(f"{m:<16}" + "".join(f"{np.mean(steps(m, a)) if steps(m, a) is not None else float('nan'):>17.2f}"
                              for a, _ in TOP) + f"   eta* {eta_star_s(m):.0f} s")
ntok = sorted({len(d["tbt_ms"]) for d in data.values()})
ax0.set_xticks(x); ax0.set_xticklabels([NAME.get(m, m) for m in A.models])
ax0.set_ylabel("ms / token")
ymax = max(np.mean(steps(m, a)) for m in A.models for a, _ in TOP if steps(m, a) is not None)
ax0.set_ylim(0, ymax * 1.45)                      # headroom for the legend
ax0.legend(ncol=2, loc="upper left", frameon=False, fontsize=11.5, columnspacing=1.0, handlelength=1.4)
ax0.grid(axis="y", alpha=0.3); panel_letter(ax0, "a")

m0 = A.models[0]
for a, lab in TOP:
    t = steps(m0, a)
    if t is None:
        continue
    k = min(A.smooth, max(1, len(t) // 4))
    sm = np.convolve(t, np.ones(k) / k, mode="valid")
    ax1.plot(np.arange(len(sm)) + k // 2 + 1, sm, color=LINE[a], lw=2.4, label=lab)
ax1.set_xlabel("decode step"); ax1.set_ylabel("ms / token")
ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K" if v else "0"))
ax1.text(0.30, 0.62, NAME.get(m0, m0), transform=ax1.transAxes, ha="left", va="center", fontsize=14, color="#444")
ax1.legend(loc="center right", bbox_to_anchor=(1.0, 0.6), frameon=False); ax1.grid(alpha=0.3); panel_letter(ax1, "b")


# ---- bottom row: OVAL oracle vs OVAL rule 1 on example sequences, per model
# (scripts/latency_oracle_r1.sh). Every run stops at its natural EOS. Oracle =
# sweep the 5 grid etas {0, .25, .5, .75, 1} on the sequence, i.e. 5 decodes;
# eta does not change the per-step cost, so it is 5 x the measured eta=0
# decode over the same number of steps as rule 1's sequence (an eta=0 run that
# stopped early at EOS is re-run with --ignore_eos to that length). Rule 1 = one measured decode with the rule-1 table plus computing
# eta* from one trace (time_rule1_find.py --ncalib 1: endpoint errors, a, b, c,
# closed form).
sd = os.path.join(A.dir, "seq")
print(f"\n{'model':<16}{'prob':>5}{'tokens':>8}{'eta=0':>9}{'oracle':>10}{'r1':>7}{'eta*':>8}{'ratio':>7}   (ms / decode step)")
for axm, m, letter in zip(axL, A.models, "cd"):
    panel_letter(axm, letter)
    eta_s = eta_star_s(m)
    xs, labels = [], []
    for j, pi in enumerate(A.probs):
        load = lambda arm: (lambda f: json.load(open(f)) if os.path.exists(f) else None)(
            os.path.join(sd, f"{m}_{A.find_task}_p{pi}_{arm}.json"))
        r1, fx = load("r1"), load("eta0.0")
        if r1 is None or fx is None:
            continue
        n = len(r1["tbt_ms"])                     # the sweep decodes THIS sequence: same length
        if len(fx["tbt_ms"]) < n:
            print(f"  skip {m} p{pi}: eta=0 run has {len(fx['tbt_ms'])} < {n} steps (rerun with --ignore_eos)")
            continue
        # ms per decode step of the sequence, as in panel (a): total time / n
        fxs, r1s = sum(fx["tbt_ms"][:n]) / n, sum(r1["tbt_ms"]) / n
        eta_ms = eta_s * 1e3 / n
        axm.bar(j - 0.19, NGRID * fxs, 0.36, color=COL["oval_fix"], edgecolor=EDGE, lw=0.6,
                label=f"OVAL oracle ({NGRID}-$\\eta$ sweep)" if not xs else None)
        axm.bar(j + 0.19, r1s, 0.36, color=COL["oval_r1"], edgecolor=EDGE, lw=0.6,
                label="OVAL rule 1: decode" if not xs else None)
        axm.bar(j + 0.19, eta_ms, 0.36, bottom=r1s, color="#FBE1E3", hatch="///", edgecolor=EDGE, lw=0.6,
                label="OVAL rule 1: compute $\\eta^*$" if not xs else None)
        axm.text(j - 0.19, NGRID * fxs + 2, f"{NGRID * fxs:.0f}", ha="center", va="bottom", fontsize=VAL + 1)
        axm.text(j + 0.19, r1s + eta_ms + 2, f"{r1s + eta_ms:.0f}", ha="center", va="bottom", fontsize=VAL + 1)
        xs.append(j); labels.append(f"{n / 1000:.1f}K tokens")
        print(f"{m:<16}{pi:>5}{n:>8}{fxs:>9.1f}{NGRID * fxs:>10.1f}{r1s:>7.1f}{eta_ms:>8.1f}"
              f"{NGRID * fxs / (r1s + eta_ms):>7.2f}")
    axm.set_xticks(xs); axm.set_xticklabels(labels)
    axm.set_ylabel("ms / token")
    axm.set_xlabel(f"{NAME.get(m, m)}, example sequences")
    if xs:
        axm.legend(loc="upper left", frameon=False)
        axm.set_ylim(0, axm.get_ylim()[1] * 1.3)
    axm.grid(axis="y", alpha=0.3)

fig.tight_layout(h_pad=2.5, w_pad=2.5)
os.makedirs(os.path.dirname(A.out), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{A.out}{A.suffix}.{ext}", dpi=200, bbox_inches="tight")
print(f"wrote {A.out}{A.suffix}.png/.pdf")
