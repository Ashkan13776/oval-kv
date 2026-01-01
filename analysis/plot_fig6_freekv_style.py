"""FreeKV Fig. 6 (end-to-end latency) rendered in the paper's own style, with
our output-aware page basis added.

Style is matched to exp_fig/efficiency.pdf, not approximated:
  * colours sampled directly from their PDF's fill operators --
      ArkVale #a21d32, InfiniGen #fbe3b0, ShadowKV #f7b52c,
      FreeKV #898a8b, RaaS #4f5858, RazorAttention #000000
    Every method we share with them keeps ITS OWN colour from their figure;
    ours takes the unused ShadowKV amber slot.
  * 2x2 panel grid: rows = Long Input / Long Generation,
    cols = Qwen-2.5-7B / Llama-3.1-8B
  * grouped bars over Batch Size, black edges, dashed horizontal grid,
    y label "Latency (s)" on the left column only, "Batch Size" beneath each
  * horizontal legend above the grid
  * bold-italic NN.NNx speedup annotation with a double-headed arrow spanning
    ArkVale -> best method, as in theirs

Data comes from results/fig6_latency/*.log (Avg TBT per decode step). Their
y-axis is total latency in seconds, so TBT is multiplied by the scenario's
generated-token count: 512 for long-input, 16384 for long-generation.
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from paths import ROOT
FIG6 = os.path.join(str(ROOT), "results/fig6_latency")
OUT = os.path.join(FIG6, "fig6_freekv_style")

# sampled from their efficiency.pdf
C_ARKVALE = "#a21d32"
C_INFINIGEN = "#fbe3b0"
C_SHADOWKV = "#f7b52c"
C_FREEKV = "#898a8b"
C_RAAS = "#4f5858"
C_RAZOR = "#000000"

# (log key, legend label, colour) -- shared methods keep their paper colour
ARMS = [
    ("arkvale", "ArkVale", C_ARKVALE),
    ("freekv",  "FreeKV",  C_FREEKV),
    ("locks50", "OVAL (ours)", C_SHADOWKV),
]
BATCHES = [1, 2, 4]           # their x-axis; we only have 1 (see note below)
PANELS = [
    ("qwen-2.5-chat-7b",  "longinput", "Qwen-2.5-7B-Instruct (Long Input)",      512),
    ("llama-3.1-chat-8b", "longinput", "Llama-3.1-8B-Instruct (Long Input)",     512),
    ("qwen-2.5-chat-7b",  "longgen",   "Qwen-2.5-7B-Instruct (Long Generation)", 16384),
    ("llama-3.1-chat-8b", "longgen",   "Llama-3.1-8B-Instruct (Long Generation)", 16384),
]


def load():
    """(model, scenario, arm, bs) -> latency seconds."""
    v = {}
    for f in glob.glob(os.path.join(FIG6, "*.log")):
        m = re.search(r"Avg TBT \(decode\) : ([\d.]+)", open(f).read())
        if not m:
            continue
        p = os.path.basename(f)[:-4].split("_")
        model, sc, arm, bs = "_".join(p[:-3]), p[-3], p[-2], int(p[-1][2:])
        v[(model, sc, arm, bs)] = float(m.group(1))
    return v


def main():
    V = load()
    plt.rcParams.update({
        "font.size": 11,
        "axes.linewidth": 1.1,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 5.6))

    for ax, (model, sc, title, ntok) in zip(axes.ravel(), PANELS):
        width = 0.74 / len(ARMS)
        vals_for_ylim = []
        for ai, (key, _lbl, col) in enumerate(ARMS):
            xs, ys = [], []
            for bi, bs in enumerate(BATCHES):
                tbt = V.get((model, sc, key, bs))
                if tbt is None:
                    continue
                xs.append(bi - 0.4 + width * (ai + 0.5))
                ys.append(tbt * ntok / 1000.0)      # ms/token -> s
            if xs:
                ax.bar(xs, ys, width=width, color=col, edgecolor="black",
                       linewidth=0.9, zorder=3)
                vals_for_ylim += ys

        # their bold-italic speedup annotation: ArkVale -> fastest arm
        base = V.get((model, sc, "arkvale", 1))
        ours = V.get((model, sc, "locks50", 1))
        if base and ours:
            lo, hi = ours * ntok / 1000.0, base * ntok / 1000.0
            xa = -0.4 + 0.74 + 0.10        # just right of the batch-1 group
            ax.annotate("", xy=(xa, hi), xytext=(xa, lo),
                        arrowprops=dict(arrowstyle="<->", lw=1.9, color="black"),
                        zorder=5)
            ax.text(xa + 0.07, (hi + lo) / 2, f"{hi/lo:.2f}x", fontsize=12.5,
                    fontweight="bold", style="italic", va="center", zorder=5)

        ax.set_title(title, fontsize=12)
        ax.set_xticks(range(len(BATCHES)))
        ax.set_xticklabels(BATCHES)
        ax.set_xlabel("Batch Size", fontsize=11)
        ax.set_xlim(-0.6, len(BATCHES) - 0.4)
        top = max(vals_for_ylim) * 1.32 if vals_for_ylim else 1
        ax.set_ylim(0, top)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4, steps=[1, 1.5, 3, 5, 10]))
        ax.grid(axis="y", linestyle=":", color="0.75", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        # batch 2 and 4 are absent: upstream eviction-indexing bug, see note
        for bi, bs in enumerate(BATCHES):
            if bs != 1 and not any(V.get((model, sc, k, bs)) for k, _, _ in ARMS):
                ax.text(bi, top * 0.04, "n/a", ha="center", fontsize=9,
                        color="0.45", style="italic")

    axes[0, 0].set_ylabel("Latency (s)", fontsize=11)
    axes[1, 0].set_ylabel("Latency (s)", fontsize=11)

    handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.9, label=l)
               for _k, l, c in ARMS]
    fig.legend(handles=handles, loc="upper center", ncol=len(ARMS),
               frameon=False, fontsize=12, bbox_to_anchor=(0.5, 1.005))
    fig.text(0.5, 0.005,
             "Batch 2 and 4 absent: upstream eviction-indexing bug (hits their own "
             "ArkVale baseline too). InfiniGen / ShadowKV / RazorAttention are not "
             "in the released perf engine. RTX PRO 6000 Blackwell; paper used A100 40GB.",
             ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=[0, 0.035, 1, 0.94])
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=170, bbox_inches="tight")
        print("wrote", f"{OUT}.{ext}")


if __name__ == "__main__":
    main()
