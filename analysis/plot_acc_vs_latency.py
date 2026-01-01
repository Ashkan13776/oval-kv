"""Accuracy vs latency trade-off, in the FreeKV paper's visual style.

The question: if OVAL is a little slower, is it a little stronger?

Both axes are measured in OUR harness for the same model, Llama-3.1-8B:

  x = decode latency, ms/token, from the Fig-6 long-input scenario
      (32K prompt / 512 generated, B=2048, batch 1)   results/fig6_latency/*.log
  y = InfiniteBench accuracy, official scorers, 10 tasks
      $FREEKV_DIR/accuracy/eval/InfiniteBench/

Nothing is hard-coded: arms appear on the plot exactly when BOTH coordinates
exist on disk, so re-running this after a baseline finishes fills the frontier
in. Arms with only one coordinate are reported as such rather than guessed at.

kv_retrieval is EXCLUDED from the mean: FullKV scores 0.0 on it with degenerate
output on this model, so the task has no headroom and cannot rank selection
quality (notes/finding-kv-retr.md).
"""
import glob
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paths import ROOT, FREEKV, use_infinitebench_scorers
use_infinitebench_scorers()
from compute_scores import get_score_one  # noqa: E402

FIG6 = ROOT / "results/fig6_latency"
# Two roots: runs predating the --out_root_dir fix landed in
# results/, later ones in res/. Scan both or half the arms vanish.
from paths import IB_DIRS as IB_ROOTS
OUT = FIG6 / "acc_vs_latency"
MODEL = "llama-3.1-chat-8b"
N_GEN = 512                      # long-input generated tokens

# colours sampled from their exp_fig/efficiency.pdf -- shared methods keep the
# colour the paper gave them; ours takes the unused ShadowKV amber slot.
C = {"ArkVale": "#a21d32", "FreeKV": "#898a8b", "RaaS": "#4f5858",
     "OVAL (ours)": "#f7b52c", "FullKV": "#000000"}

# display name -> (fig6 log key, InfiniteBench result SUBDIR, filename fragment,
#                   fragment that must NOT appear)
#
# Match on the subdir, not just the filename: their out_dir is
# "{model}-{method}", so --method arkv / raas / full each get their own
# directory. Matching on the filename alone is unsafe -- an earlier version
# used "-a" for RaaS, which globbed straight into the spec_ret files and put a
# fabricated RaaS accuracy on the plot.
ARMS = [
    ("ArkVale",     "arkvale", "arkv",     "",                ""),
    ("RaaS",        "raas",    "raas",     "",                ""),
    ("FreeKV",      "freekv",  "spec_ret", "avgSM-pQ2",       "oval"),
    ("OVAL (ours)", "oval50", "spec_ret", "oval_r8_eta0.25", ""),
]
# Dense reference: no page selection, so it has no Fig-6 latency point. Drawn
# as a ceiling line rather than a plotted method.
FULLKV = ("full", "", "")
# All 11 tasks. kv_retrieval was previously excluded on the grounds that FullKV
# scored 0.0 on it; that was the tuple-KV cache bug (notes/freekv-port.md).
# With the cache fixed FullKV scores 56.0 there -- the largest headroom on the
# benchmark -- so excluding it was dropping the one cell that separates methods.
TASKS = ["passkey", "number_string", "kv_retrieval", "longdialogue_qa_eng",
         "longbook_sum_eng", "longbook_choice_eng", "longbook_qa_eng",
         "longbook_qa_chn", "math_find", "code_run", "code_debug"]


def _pat(frag, task):
    """Glob for one arm's task file. An empty fragment must not produce '**',
    which pathlib rejects as a partial path component."""
    return f"*{frag}*-{task}.jsonl" if frag else f"*-{task}.jsonl"


def latency(key):
    """ms/token for one arm, from the Fig-6 long-input batch-1 log."""
    f = FIG6 / f"{MODEL}_longinput_{key}_bs1.log"
    if not f.exists():
        return None
    m = re.search(r"Avg TBT \(decode\) : ([\d.]+)", f.read_text())
    return float(m.group(1)) if m else None


def accuracy(method, frag, exclude):
    """InfiniteBench 10-task mean for one arm, official scorers.

    Returns (mean, n_tasks) or (None, n) when the arm is incomplete. An
    ambiguous match is a hard error, not a silent pick of hits[0]: two files
    matching one arm means the fragment is too loose and the number would be
    someone else's.
    """
    dirs = [r / f"{MODEL}-{method}" for r in IB_ROOTS if (r / f"{MODEL}-{method}").is_dir()]
    if not dirs:
        return None, 0
    scores = []
    for task in TASKS:
        hits = [h for d in dirs for h in d.glob(_pat(frag, task))
                if not (exclude and exclude in h.name)]
        if len(hits) > 1:
            raise SystemExit(f"ambiguous match for {method}/{frag} on {task}: "
                             + ", ".join(h.name for h in hits))
        if not hits:
            return None, len(scores)
        rows = [json.loads(l) for l in open(hits[0]) if l.strip()]
        if not rows:
            return None, len(scores)
        s = [get_score_one(r["prediction"], r["ground_truth"], task, MODEL)
             for r in rows]
        scores.append(100.0 * float(np.mean(s)))
    return float(np.mean(scores)), len(scores)


def main():
    pts, partial = [], []
    for name, key, method, frag, excl in ARMS:
        lat = latency(key)
        acc, nt = accuracy(method, frag, excl)
        if lat is not None and acc is not None:
            pts.append((name, lat, acc))
        else:
            partial.append((name, lat, acc, nt))
    for name, lat, acc, nt in partial:
        have = []
        if lat is not None:
            have.append(f"latency {lat:.2f} ms")
        have.append(f"accuracy {nt}/{len(TASKS)} tasks")
        print(f"  [pending] {name}: {', '.join(have)} -- not plotted",
              file=sys.stderr)
    if len(pts) < 2:
        raise SystemExit("need >=2 fully-measured arms")

    plt.rcParams.update({
        "font.size": 11, "axes.linewidth": 1.1,
        "font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(7.4, 5.2))

    # Pareto frontier: an arm is dominated if another is both faster and better
    pts.sort(key=lambda p: p[1])
    front, best = [], -1e9
    for name, lat, acc in pts:
        if acc > best:
            front.append((lat, acc)); best = acc
    if len(front) > 1:
        fx, fy = zip(*front)
        ax.plot(fx, fy, color="0.55", lw=1.6, linestyle=(0, (5, 3)),
                zorder=2, label="_")
        ax.fill_between(fx, fy, ax.get_ylim()[0], color="0.92", zorder=0)

    # Three of the four arms sit within 6 ms of each other, so per-point text
    # boxes overlap and hide the markers underneath. Use a legend instead.
    for name, lat, acc in pts:
        dominated = (lat, acc) not in front
        ax.scatter(lat, acc, s=320, color=C.get(name, "0.5"),
                   edgecolor="black", linewidth=1.3, zorder=5,
                   marker="o" if not dominated else "X",
                   label=f"{name}  —  {acc:.1f}  @  {lat:.1f} ms"
                         + ("" if not dominated else "   (dominated)"))
    leg = ax.legend(loc="upper left", frameon=True, fontsize=10.5,
                    borderpad=0.7, labelspacing=0.65, handletextpad=0.8,
                    framealpha=0.95, edgecolor="0.8")
    leg.set_zorder(8)

    full_acc, full_n = accuracy(*FULLKV)
    if full_acc is not None and full_n == len(TASKS):
        ax.axhline(full_acc, color="black", lw=1.5, linestyle=(0, (6, 3)),
                   zorder=3)
        ax.text(ax.get_xlim()[1], full_acc, f" FullKV {full_acc:.1f} ",
                ha="right", va="bottom", fontsize=10.5, fontweight="bold",
                color="black")

    xs = [p[1] for p in pts]; ys = [p[2] for p in pts] + (
        [full_acc] if full_acc is not None else [])
    xpad = max(0.12 * (max(xs) - min(xs)), 2.0)
    ypad = max(0.45 * (max(ys) - min(ys)), 0.6)
    ax.set_xlim(min(xs) - xpad, max(xs) + xpad * 1.3)
    ax.set_ylim(min(ys) - ypad, max(ys) + ypad * 1.6)

    # Keep the axis labels short enough to fit the axes extent: a label taller
    # than the axes escapes the canvas and gets clipped by bbox_inches="tight".
    ax.set_xlabel("Decode latency (ms / token)$\\,\\downarrow$", fontsize=11.5)
    ax.set_ylabel("InfiniteBench accuracy, 11 tasks$\\,\\uparrow$", fontsize=11.5)
    ax.set_title("Accuracy vs. latency — Llama-3.1-8B-Instruct, $B$ = 2048, "
                 "32K input", fontsize=12.5, pad=12)
    ax.grid(linestyle=":", color="0.75", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    note = ("Both axes measured in one harness, all 11 InfiniteBench tasks. "
            "Latency: 32K input / 512 generated, batch 1, RTX PRO 6000 "
            "Blackwell.\nAccuracy: 50 seeded-random records per task, greedy. "
            "FullKV is dense (no page selection), so it has no latency point.")
    if partial:
        note += ("\nNot plotted (one axis still running): "
                 + ", ".join(p[0] for p in partial) + ".")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.255)
    fig.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=8.3,
             color="0.35", linespacing=1.5)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")
        print("wrote", f"{OUT}.{ext}")
    print("\n  arm                 latency(ms)   acc")
    for name, lat, acc in pts:
        print(f"  {name:<18} {lat:>10.2f} {acc:>7.2f}")


if __name__ == "__main__":
    main()
