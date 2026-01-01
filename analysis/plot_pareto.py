"""Pareto front on MATH50 at the published config (B=2048, sink/recent 512).

Axes: KV memory held per KV head per layer, against avg@8 accuracy.

Scope. Only arms that decide WHAT TO KEEP from a given budget are plotted:
retrieval (full cache, re-chooses what to attend each step) and eviction (the
cache shrinks). Fixed-window arms (StreamingLLM, sliding window) and the
PyramidKV budget schedule are excluded -- the windows make no selection at all,
and a budget schedule is orthogonal, layerable onto any scorer here. So is
OVAL Retv. no-spec., which differs from OVAL Retv. only in FreeKV's
speculative retrieval (docs/speculative-retrieval.md), a latency mechanism
rather than a scoring rule. All of their results stay on disk;
analysis/front.py prints the full table.

Eviction arms retain only sink+recent+budget, so they are charged 96 pages
rather than the 110 a retrieval arm holds. Those implemented here WITHOUT
eviction (H2O, TOVA, SnapKV keep the cache and merely stop attending, so they
can re-attend a page the published method would have destroyed) carry a dagger
and are kept off the frontier: an upper bound is not an achievable point.

Labels are packed vertically within each memory column, so nothing collides.

Usage:  python analysis/plot_pareto.py [--data-root DIR] [--budget 2048]
"""
import argparse, glob, json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

D = "results/freekv/codec/ds-r1-qwen-7b"
PAT = r"\\boxed{((?:[^{}]|\{[^{}]*\})*)}"

# ---------------------------------------------------------------- memory model
# ONE definition, applied to every arm: the KV state resident for ONE KV head in
# ONE layer, on a reference sequence, in bytes. Nothing else is counted.
#
#   HD      head dimension
#   BPE     bytes per element (fp16)
#   PG      page size, tokens
#   L_REF   reference sequence length, tokens. FIXED across arms on purpose:
#           a weaker method generates longer traces (dense 3745, FreeKV 3847,
#           OVAL 3869 measured here), and charging each its own length would
#           fold accuracy into the memory axis. 3745 = the dense mean.
#   SINK/RECENT  always-exact regions: never paged, never summarised, so their
#           KEYS stay exact even for arms that compress the paged keys.
HD, BPE, PG = 128, 2, 32
L_REF, SINK, RECENT = 3745, 512, 512
TOK = HD * BPE                      # bytes for one token, one tensor (K or V)
N_EXACT = SINK + RECENT             # tokens held exact by every arm
N_PAGED = (L_REF - N_EXACT) // PG   # whole pages that can be SELECTED
N_DIGEST = (L_REF - SINK) // PG     # pages that have a digest written: the
                                    # summary is built when a page completes,
                                    # which includes pages still inside the
                                    # recent window
B_K = PG * TOK                      # exact keys, one page
B_V = B_K

#  tag: (label, family, digest bytes/page, keeps exact keys, glob)
ARMS = {
 "q4key":    ("4-bit keys",     "ret",   2560, False, "{D}-quest/MATH50-*b{B}-avgSM-q4key*"),
 "ours":     ("OVAL Retv.+Comp.", "ours",  2816, False, "{D}-spec_ret/MATH50-*b{B}-avgSM-oval_r8_eta0.0_q0_recon1-*"),
 "ours_sel": ("OVAL Retv.",       "ours",  2816, True,  "{D}-spec_ret/MATH50-*b{B}-avgSM-oval_r8_eta0.0_q0-pQ2*"),
 "loki":     ("Loki r16",       "ret",   1024, True,  "{D}-quest/MATH50-*b{B}-avgSM-loki*"),
 "centroid": ("Centroid",       "ret",   256,  True,  "{D}-quest/MATH50-*b{B}-avgSM-centroid*"),
 "shadowkv": ("ShadowKV",       "ret",   256,  True,  "{D}-quest/MATH50-*b{B}-avgSM-shadowkv*"),
 "rocketkv": ("RocketKV",       "ret",   256,  True,  "{D}-quest/MATH50-*b{B}-avgSM-rocketkv*"),
 "infllm":   ("InfLLM",         "ret",   8,    True,  "{D}-quest/MATH50-*b{B}-avgSM-infllm*"),
 "sparq":    ("SparQ",          "ret",   0,    True,  "{D}-quest/MATH50-*b{B}-avgSM-sparq*"),
 "quest":    ("Quest",          "ret",   512,  True,  "{D}-quest/MATH50-*b{B}-avgSM-skl1*"),
 "freekv":   ("FreeKV",         "ret",   512,  True,  "{D}-spec_ret/MATH50-*b{B}-avgSM-pQ2-*"),
 "arkvale":  ("ArkVale",        "ret",   512,  True,  "{D}-spec_ret/MATH50-*b{B}-avgSM-arkv-*"),
 "expatt":   ("ExpectedAttn",   "ret",   4,    True,  "{D}-quest/MATH50-*b{B}-avgSM-expatt*"),
 "knorm":    ("K-norm",         "ret",   2,    True,  "{D}-quest/MATH50-*b{B}-avgSM-knorm*"),
 "random":   ("Random",         "ctrl",  0,    True,  "{D}-quest/MATH50-*b{B}-avgSM-random*"),
 "tova":     ("TOVA \u2020",    "evict", 4,    False, "{D}-quest/MATH50-*b{B}-avgSM-tova*"),
 "h2o":      ("H2O \u2020",     "evict", 4,    False, "{D}-quest/MATH50-*b{B}-avgSM-h2o*"),
 "raas":     ("RaaS",           "evict", 516,  False, "{D}-raas/MATH50-*b{B}*"),
 "full":     ("Dense",          "dense", 0,    True,  "{D}-full/MATH50-*"),
 "oracle":   ("Oracle",         "ceil",  0,    True,  "{D}-quest/MATH50-*b{B}-avgSM-oracle*"),
 "oracle_prev": ("Oracle q(t\u22121)", "ceil", 0, True,
                 "{D}-spec_ret/MATH50-*b{B}-avgSM-oracle*"),
}
UB = {"h2o", "tova"}                      # implemented without eviction
STYLE = {                                 # family: (colour, marker, size)
 "ours":  ("#c2410c", "D", 115),
 "ret":   ("#334155", "o",  66),
 "evict": ("#b45309", "v",  78),
 "ctrl":  ("#0891b2", "s",  72),
 "ceil":  ("#7c3aed", "*", 250),
 "dense": ("#111827", "P", 110),
}

def load(pat):
    by, files = {}, glob.glob(pat)
    if not files and "oval_" in pat:
        files = glob.glob(pat.replace("oval_", "oab_"))     # pre-rename results
    for f in sorted(files):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        if len(rows) < 50:
            continue
        s = re.search(r"-seed(\d+)", f)
        by[int(s.group(1)) if s else 42] = {
            e["qid"]: int(bool(re.search(PAT, e["pred"]) and
                re.search(PAT, e["pred"]).group(1).replace(" ", "")
                == str(e["answer"]).replace(" ", ""))) for e in rows}
    return by

# Fixed per-KV-head, per-layer overhead that is NOT per-page. Expected
# Attention models the query distribution per QUERY head -- the published rule
# -- so it holds a mean and a full covariance for each of the g heads sharing a
# KV head: g * (d + d*d) floats = 451 KiB. That is a quarter of this arm's
# footprint, and it is the honest price of the method rather than a rounding.
GQA_G = 7                           # query heads per KV head, ds-r1-qwen-7b
FIXED = {"expatt": GQA_G * (HD + HD * HD) * 4}


def memory(tag, fam, exact, digest, budget):
    """Resident KV bytes for one KV head in one layer, as KiB.

    dense      : K and V for every token.
    eviction   : K and V for the survivors only -- sink + recent + budget --
                 because the cache genuinely shrinks. Capped at L_REF.
    retrieval  : the whole cache stays resident, so K and V for every token,
                 plus the per-page digest it stores to decide what to attend.
    codec      : as retrieval, except the PAGED keys are replaced by the record
                 (`exact=False`). The sink and recent keys are not paged and
                 stay exact -- charging zero for them would be a discount the
                 method does not earn.
    """
    if fam == "evict":
        kept = min(L_REF, N_EXACT + budget)
        # an eviction arm still holds a digest for each page it retains
        return (kept * TOK * 2 + (budget // PG) * digest) / 1024
    v_all = L_REF * TOK                                   # values, always exact
    if tag == "full":
        return (L_REF * TOK + v_all) / 1024
    k_exact = N_EXACT * TOK                               # sink + recent keys
    k_paged = N_PAGED * (B_K if exact else 0)             # exact, or replaced
    return (k_exact + k_paged + v_all + N_DIGEST * digest
            + FIXED.get(tag, 0)) / 1024

def pack(vals, lo, hi, gap):
    """Spread values to at least `gap` apart inside [lo, hi], moving each as
    little as possible. Stacks labels in one memory column without overlap."""
    order = np.argsort(vals)
    v = np.asarray(vals, float)[order]
    for i in range(1, len(v)):
        v[i] = max(v[i], v[i - 1] + gap)
    over = v[-1] - hi
    if over > 0:
        v -= over
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], v[i + 1] - gap)
    v = np.maximum(v, lo)
    out = np.empty_like(v)
    out[order] = v
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--out", default="results/figures/pareto_math50_b2048")
    ap.add_argument("--root", default=".")
    ap.add_argument("--data-root", default=None)
    a = ap.parse_args()
    droot = a.data_root or a.root

    pts = {}
    for tag, (label, fam, dig, exact, pat) in ARMS.items():
        by = load(pat.format(D=f"{droot}/{D}", B=a.budget))
        if not by:
            continue
        seeds = sorted(by)
        qs = sorted(set.intersection(*[set(by[s]) for s in seeds]))
        M = np.array([[by[s][q] for q in qs] for s in seeds])
        pts[tag] = dict(tag=tag, label=label, fam=fam, M=M, acc=M.mean() * 100,
                        exact=exact, mem=memory(tag, fam, exact, dig, a.budget))
    if "freekv" not in pts:
        sys.exit("no FreeKV reference for this budget")

    achievable = [p for p in pts.values()
                  if p["fam"] != "ceil" and p["tag"] not in UB]
    front, best = [], -1e9
    for p in sorted(achievable, key=lambda z: (z["mem"], -z["acc"])):
        if p["acc"] > best:
            front.append(p); best = p["acc"]
    fset = {p["tag"] for p in front}

    # The exact-key arms sit inside ~50 KiB of one another. No font size
    # separates thirteen labels in that box, so the box is drawn once and the
    # arms are named in the ranked panel instead of with thirteen leader lines.
    CLUST_LO, CLUST_HI = 1860, 1935
    pile = [p for p in pts.values() if CLUST_LO <= p["mem"] <= CLUST_HI]
    loose = [p for p in pts.values() if p not in pile]

    # RaaS is ~17 points below everything else and would otherwise compress the
    # live range into the top fifth of the axis; break the axis instead.
    BRK = 65.0
    low = [p for p in pts.values() if p["acc"] < BRK]
    hi_acc = [p["acc"] for p in pts.values() if p["acc"] >= BRK]

    plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(17.0, 9.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.30, 1.0],
                          height_ratios=[7.0, 1.0] if low else [1, 0.001],
                          hspace=0.07, wspace=0.26,
                          left=.055, right=.985, top=.885, bottom=.085)
    axT = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0], sharex=axT) if low else None
    axR = fig.add_subplot(gs[:, 1])

    xs = [p["mem"] for p in pts.values()]
    xlo, xhi = min(xs) - 135, max(xs) + 145
    axT.set_xlim(xlo, xhi)
    axT.set_ylim(min(hi_acc) - 1.3, max(hi_acc) + 1.5)

    # frontier
    if len(front) > 1:
        axT.step([p["mem"] for p in front], [p["acc"] for p in front],
                 where="post", color="#c2410c", lw=2.4, alpha=.40, zorder=2)
    axT.scatter([p["mem"] for p in front if p["acc"] >= BRK],
                [p["acc"] for p in front if p["acc"] >= BRK],
                s=560, facecolors="none", edgecolors="#c2410c", lw=2.6, zorder=9)

    def draw(ax, group):
        for p in group:
            c, mk, sz = STYLE[p["fam"]]
            ax.scatter(p["mem"], p["acc"], s=sz * 1.5, c=c, marker=mk, zorder=7,
                       edgecolors="white", linewidths=1.3)

    draw(axT, [p for p in pts.values() if p["acc"] >= BRK])

    # the exact-key cluster: one box, one caption
    if pile:
        px = [p["mem"] for p in pile]; py = [p["acc"] for p in pile]
        axT.add_patch(mpatches.Rectangle(
            (min(px) - 34, min(py) - .30), (max(px) - min(px)) + 68,
            (max(py) - min(py)) + .60,
            fill=False, ec="#94a3b8", lw=1.3, ls=(0, (5, 4)), zorder=5))
        axT.annotate(f"{len(pile)} methods that store the keys exactly\n"
                     f"(all named at right)",
                     xy=(float(np.mean(px)), min(py) - .55),
                     xytext=(0, -16), textcoords="offset points",
                     ha="center", va="top", fontsize=13.5, color="#64748b",
                     linespacing=1.4)

    # label only the arms outside the box -- these are well separated
    OFF = {"ours": (0, 26), "ours_sel": (0, 24), "q4key": (0, -26),
           "tova": (16, 0), "h2o": (16, 0), "loki": (16, 0),
           "expatt": (0, 22), "raas": (16, 0)}
    for p in loose:
        if p["acc"] < BRK:
            continue
        dx, dy = OFF.get(p["tag"], (0, 20))
        axT.annotate(p["label"], (p["mem"], p["acc"]), textcoords="offset points",
                     xytext=(dx, dy),
                     ha="right" if dx < 0 else ("left" if dx > 0 else "center"),
                     va="center", fontsize=14.5,
                     fontweight="bold" if p["fam"] == "ours" else "normal",
                     color=STYLE[p["fam"]][0])

    axT.grid(alpha=.17, lw=.8)
    axT.set_axisbelow(True)
    for sp in ("top", "right"):
        axT.spines[sp].set_visible(False)
    axT.set_ylabel("avg@8 accuracy   (%)", fontsize=16.5, labelpad=9)

    if axB is not None:
        draw(axB, low)
        axB.scatter([p["mem"] for p in front if p["acc"] < BRK],
                    [p["acc"] for p in front if p["acc"] < BRK],
                    s=560, facecolors="none", edgecolors="#c2410c", lw=2.6, zorder=9)
        for p in low:
            axB.annotate(p["label"], (p["mem"], p["acc"]), textcoords="offset points",
                         xytext=(16, 0), ha="left", va="center", fontsize=14.5,
                         color=STYLE[p["fam"]][0])
        lo_acc = [p["acc"] for p in low]
        axB.set_ylim(min(lo_acc) - 1.6, max(lo_acc) + 1.6)
        axB.set_yticks([round(min(lo_acc))])
        axB.grid(alpha=.17, lw=.8); axB.set_axisbelow(True)
        for sp in ("top", "right"):
            axB.spines[sp].set_visible(False)
        axT.spines["bottom"].set_visible(False)
        axT.tick_params(labelbottom=False, bottom=False)
        # break marks
        kw = dict(marker=[(-1, -.6), (1, .6)], markersize=11, lw=0,
                  color="#475569", mec="#475569", mew=1.5, clip_on=False)
        axT.plot([0], [0], transform=axT.transAxes, **kw)
        axB.plot([0], [1], transform=axB.transAxes, **kw)
        xax = axB
    else:
        xax = axT
    xax.set_xlabel("KV memory retained   (KiB per KV head per layer)",
                   fontsize=16.5, labelpad=9)
    xax.tick_params(labelsize=14)
    axT.tick_params(labelsize=14)

    # ---------------- right: every arm, ranked, named ----------------
    order = sorted(pts.values(), key=lambda z: z["acc"])
    yy = np.arange(len(order))
    xr_lo = min(p["acc"] for p in order) - 1.5
    xr_hi = max(p["acc"] for p in order) + 1.0
    axR.set_xlim(xr_lo, xr_hi)
    for y, p in zip(yy, order):
        c = STYLE[p["fam"]][0]
        axR.plot([xr_lo, p["acc"]], [y, y], color=c, lw=1.2, alpha=.20, zorder=1)
        axR.scatter(p["acc"], y, s=STYLE[p["fam"]][2] * 1.25, c=c,
                    marker=STYLE[p["fam"]][1], zorder=3,
                    edgecolors="white", linewidths=1.0)
        axR.annotate(f"{p['acc']:.2f}", (p["acc"], y), textcoords="offset points",
                     xytext=(13, 0), ha="left", va="center", fontsize=12.5,
                     color="#475569",
                     fontweight="bold" if p["fam"] == "ours" else "normal")
    axR.set_yticks(yy)
    axR.set_yticklabels([p["label"] + ("  ★" if p["tag"] in fset else "")
                         for p in order], fontsize=14.5)
    for t, p in zip(axR.get_yticklabels(), order):
        t.set_color(STYLE[p["fam"]][0])
        if p["fam"] == "ours":
            t.set_fontweight("bold")
    axR.set_ylim(-.8, len(order) - .2)
    axR.set_xlabel("avg@8 accuracy   (%)", fontsize=16.5, labelpad=9)
    axR.tick_params(axis="x", labelsize=14)
    axR.tick_params(axis="y", length=0)
    axR.grid(axis="x", alpha=.17, lw=.8); axR.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        axR.spines[sp].set_visible(False)

    # ---------------- one legend strip across the top ----------------
    handles = [
        Line2D([], [], ls="", marker="D", ms=12, color="#c2410c",
               mec="white", mew=1.2, label="OVAL (ours)"),
        Line2D([], [], ls="", marker="o", ms=10, color="#334155",
               mec="white", mew=1.2, label="retrieval baseline"),
        Line2D([], [], ls="", marker="v", ms=11, color="#b45309",
               mec="white", mew=1.2, label="eviction   († upper bound)"),
        Line2D([], [], ls="", marker="s", ms=10, color="#0891b2",
               mec="white", mew=1.2, label="random control"),
        Line2D([], [], ls="", marker="*", ms=18, color="#7c3aed",
               mec="white", mew=1.0, label="oracle ceiling"),
        Line2D([], [], ls="", marker="P", ms=12, color="#111827",
               mec="white", mew=1.2, label="dense attention"),
        Line2D([], [], ls="", marker="o", ms=17, mfc="none", mec="#c2410c",
               mew=2.4, label="Pareto-optimal  ★"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .99),
               ncol=7, frameon=False, fontsize=13.5,
               columnspacing=1.5, handletextpad=.6)

    os.makedirs(os.path.dirname(f"{a.root}/{a.out}"), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.root}/{a.out}.{ext}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    print(f"wrote {a.out}.pdf/.png   {len(pts)} arms   "
          f"({len(pile)} in the exact-key cluster)")
    print("Pareto frontier (achievable arms only):")
    for p in front:
        print(f"  {p['tag']:<12}{p['mem']:8.0f} KiB{p['acc']:8.2f}%")


if __name__ == "__main__":
    main()
