"""Pareto front on MATH50 at the published config (B=2048, sink/recent 512).

Axes: KV memory held per KV head per layer, against avg@8 accuracy.

Scope. Only arms that decide WHAT TO KEEP from a given budget are plotted:
retrieval (full cache, re-chooses what to attend each step) and eviction (the
cache shrinks). Fixed-window arms (StreamingLLM, sliding window) and the
PyramidKV budget schedule are excluded -- the windows make no selection at all,
and a budget schedule is orthogonal, layerable onto any scorer here. Their
results stay on disk; analysis/front.py prints the full table.

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
HD, PG, NPAGE = 128, 32, 110
B_K = HD * PG * 2
B_V = B_K

#  tag: (label, family, digest bytes/page, keeps exact keys, glob)
ARMS = {
 "q4key":    ("4-bit keys",     "ret",   2560, False, "{D}-quest/MATH50-*b{B}-avgSM-q4key*"),
 "ours":     ("OVAL + codec",   "ours",  2816, False, "{D}-spec_ret/MATH50-*b{B}-avgSM-oval_r8_eta0.0_q0_recon1-*"),
 "ours_sel": ("OVAL index",     "ours",  2816, True,  "{D}-spec_ret/MATH50-*b{B}-avgSM-oval_r8_eta0.0_q0-pQ2*"),
 "ours_q":   ("OVAL no-spec",   "ours",  2816, True,  "{D}-quest/MATH50-*b{B}-avgSM-oval_r8_eta0.0_q0-*"),
 "loki":     ("Loki",           "ret",   512,  True,  "{D}-quest/MATH50-*b{B}-avgSM-loki*"),
 "centroid": ("Centroid",       "ret",   256,  True,  "{D}-quest/MATH50-*b{B}-avgSM-centroid*"),
 "shadowkv": ("ShadowKV",       "ret",   256,  True,  "{D}-quest/MATH50-*b{B}-avgSM-shadowkv*"),
 "rocketkv": ("RocketKV",       "ret",   256,  True,  "{D}-quest/MATH50-*b{B}-avgSM-rocketkv*"),
 "infllm":   ("InfLLM",         "ret",   1024, True,  "{D}-quest/MATH50-*b{B}-avgSM-infllm*"),
 "sparq":    ("SparQ",          "ret",   0,    True,  "{D}-quest/MATH50-*b{B}-avgSM-sparq*"),
 "quest":    ("Quest",          "ret",   512,  True,  "{D}-quest/MATH50-*b{B}-avgSM-skl1*"),
 "freekv":   ("FreeKV",         "ret",   512,  True,  "{D}-spec_ret/MATH50-*b{B}-avgSM-pQ2-*"),
 "arkvale":  ("ArkVale",        "ret",   512,  True,  "{D}-spec_ret/MATH50-*b{B}-avgSM-arkv-*"),
 "expatt":   ("ExpectedAttn",   "ret",   0,    True,  "{D}-quest/MATH50-*b{B}-avgSM-expatt*"),
 "knorm":    ("K-norm",         "ret",   2,    True,  "{D}-quest/MATH50-*b{B}-avgSM-knorm*"),
 "random":   ("Random",         "ctrl",  0,    True,  "{D}-quest/MATH50-*b{B}-avgSM-random*"),
 "tova":     ("TOVA \u2020",    "evict", 4,    False, "{D}-quest/MATH50-*b{B}-avgSM-tova*"),
 "h2o":      ("H2O \u2020",     "evict", 4,    False, "{D}-quest/MATH50-*b{B}-avgSM-h2o*"),
 "snapkv":   ("SnapKV \u2020",  "evict", 0,    False, "{D}-quest/MATH50-*b{B}-avgSM-snapkv*"),
 "raas":     ("RaaS",           "evict", 0,    False, "{D}-raas/MATH50-*b{B}*"),
 "full":     ("Dense",          "dense", 0,    True,  "{D}-full/MATH50-*"),
 "oracle":   ("Oracle",         "ceil",  0,    True,  "{D}-quest/MATH50-*b{B}-avgSM-oracle*"),
 "oracle_prev": ("Oracle q(t\u22121)", "ceil", 0, True,
                 "{D}-spec_ret/MATH50-*b{B}-avgSM-oracle*"),
}
UB = {"h2o", "tova", "snapkv"}            # implemented without eviction
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

def memory(tag, fam, exact, digest, budget):
    """KiB per KV head per layer on a mean-length sequence."""
    if fam == "evict":                 # the cache shrinks to what survives
        return (1024 + budget) / PG * (B_K + B_V) / 1024
    if tag == "full":
        return NPAGE * (B_K + B_V) / 1024
    return NPAGE * ((B_K if exact else 0) + B_V + digest) / 1024

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
                        mem=memory(tag, fam, exact, dig, a.budget))
    if "freekv" not in pts:
        sys.exit("no FreeKV reference for this budget")

    ref, fkv = pts["freekv"]["M"], pts["freekv"]["acc"]
    rng = np.random.default_rng(0)
    for p in pts.values():
        n = min(p["M"].shape[1], ref.shape[1]); k = min(p["M"].shape[0], ref.shape[0])
        d = p["M"][:k, :n].mean(0) - ref[:k, :n].mean(0)
        b = d[rng.integers(0, n, (8000, n))].mean(1) * 100
        p["lo"], p["hi"] = np.percentile(b, [2.5, 97.5])
        p["tie"] = p["lo"] <= 0 <= p["hi"]
    band = (min(p["acc"] for p in pts.values() if p["tie"]),
            max(p["acc"] for p in pts.values() if p["tie"]))

    achievable = [p for p in pts.values()
                  if p["fam"] != "ceil" and p["tag"] not in UB]
    front, best = [], -1e9
    for p in sorted(achievable, key=lambda z: (z["mem"], -z["acc"])):
        if p["acc"] > best:
            front.append(p); best = p["acc"]

    plt.rcParams.update({"font.size": 14})
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(16.4, 8.8), gridspec_kw=dict(width_ratios=[1.0, 1.0]))

    # ---------- left: the Pareto structure ----------
    xs = [p["mem"] for p in pts.values()]
    ys = [p["acc"] for p in pts.values()]
    xlo, xhi = min(xs) - 170, max(xs) + 170
    ylo, yhi = min(ys) - 2.5, max(ys) + 2.5
    axL.set_xlim(xlo, xhi); axL.set_ylim(ylo, yhi)
    axL.axhspan(*band, color="#64748b", alpha=.10, zorder=0, lw=0)
    if len(front) > 1:
        axL.step([p["mem"] for p in front], [p["acc"] for p in front],
                 where="post", color="#c2410c", lw=2.0, alpha=.45, zorder=2)
    axL.scatter([p["mem"] for p in front], [p["acc"] for p in front],
                s=470, facecolors="none", edgecolors="#c2410c", lw=2.3, zorder=8)
    for p in pts.values():
        c, mk, sz = STYLE[p["fam"]]
        axL.errorbar(p["mem"], p["acc"],
                     yerr=[[max(0., p["acc"] - (fkv + p["lo"]))],
                           [max(0., (fkv + p["hi"]) - p["acc"])]],
                     fmt="none", ecolor=c, elinewidth=1.2, alpha=.25, capsize=0, zorder=3)
        axL.scatter(p["mem"], p["acc"], s=sz, c=c, marker=mk, zorder=6,
                    edgecolors="white", linewidths=1.0)
    # label only what does NOT sit in the exact-key pile; the pile is named in
    # full on the right, where there is room for it
    pile = [p for p in pts.values() if 1740 <= p["mem"] <= 1900]
    side = {"q4key": (0, 17), "ours": (0, -24), "ours_sel": (0, 17),
            "ours_q": (0, -24), "raas": (0, 17),
            "tova": (-13, 6), "h2o": (-13, 2), "snapkv": (-13, -8)}
    for p in pts.values():
        if p in pile:
            continue
        dx, dy = side.get(p["tag"], (0, 16))
        axL.annotate(p["label"], (p["mem"], p["acc"]), textcoords="offset points",
                     xytext=(dx, dy), ha="right" if dx < 0 else "center",
                     va="center", fontsize=14,
                     fontweight="bold" if p["fam"] in ("ours", "ceil") else "normal",
                     color=STYLE[p["fam"]][0])
    if pile:
        px = [p["mem"] for p in pile]; py = [p["acc"] for p in pile]
        axL.add_patch(mpatches.Rectangle(
            (min(px) - 34, min(py) - .7), (max(px) - min(px)) + 68,
            (max(py) - min(py)) + 1.4, fill=False, ec="#94a3b8", lw=1.1,
            ls=(0, (4, 3)), zorder=7))
        axL.annotate(f"{len(pile)} methods that keep the keys exact "
                     f"\u2014 named at right",
                     xy=(float(np.mean(px)), min(py) - .8), xytext=(0, -13),
                     textcoords="offset points", ha="center", va="top",
                     fontsize=13, color="#64748b")
    axL.set_xlabel("KV memory retained   (KiB per KV head per layer)", fontsize=15.5)
    axL.set_ylabel("avg@8 accuracy   (%)", fontsize=15.5)
    axL.tick_params(labelsize=13.5)
    axL.grid(alpha=.15, lw=.7)
    for sp in ("top", "right"):
        axL.spines[sp].set_visible(False)
    axL.legend(handles=[
        Line2D([], [], ls="", marker="D", ms=10, color="#c2410c", label="OVAL (ours)"),
        Line2D([], [], ls="", marker="o", ms=8,  color="#334155", label="retrieval baseline"),
        Line2D([], [], ls="", marker="v", ms=9,  color="#b45309",
               label="eviction  (\u2020 no-evict upper bound)"),
        Line2D([], [], ls="", marker="s", ms=8,  color="#0891b2", label="random control"),
        Line2D([], [], ls="", marker="*", ms=16, color="#7c3aed", label="oracle ceiling"),
        Line2D([], [], ls="", marker="P", ms=10, color="#111827", label="dense attention"),
        Line2D([], [], ls="", marker="o", ms=15, mfc="none", mec="#c2410c", mew=2.2,
               label="Pareto-optimal"),
        mpatches.Patch(fc="#64748b", alpha=.18, ec="none",
                       label="tie band: paired 95% CI vs FreeKV spans 0"),
    ], loc="lower left", frameon=False, fontsize=12.5, labelspacing=.72,
       borderpad=.8, handletextpad=.8)

    # ---------- right: every method, named, with its interval ----------
    order = sorted(pts.values(), key=lambda z: z["acc"])
    yy = np.arange(len(order))
    axR.axvspan(*band, color="#64748b", alpha=.10, zorder=0, lw=0)
    axR.axvline(fkv, color="#64748b", lw=1.0, ls="--", zorder=1)
    for y, p in zip(yy, order):
        c = STYLE[p["fam"]][0]
        axR.plot([fkv + p["lo"], fkv + p["hi"]], [y, y], color=c, lw=2.6,
                 alpha=.38, solid_capstyle="round", zorder=2)
        axR.scatter(p["acc"], y, s=STYLE[p["fam"]][2] * .9, c=c,
                    marker=STYLE[p["fam"]][1], zorder=3,
                    edgecolors="white", linewidths=.9)
    axR.set_yticks(yy)
    axR.set_yticklabels([p["label"] for p in order], fontsize=14)
    for t, p in zip(axR.get_yticklabels(), order):
        t.set_color(STYLE[p["fam"]][0])
        if p["fam"] in ("ours", "ceil"):
            t.set_fontweight("bold")
    axR.set_ylim(-.8, len(order) - .2)
    axR.set_xlabel("avg@8 accuracy   (%)      bar = paired 95% CI vs FreeKV", fontsize=15.5)
    axR.tick_params(axis="x", labelsize=13.5)
    axR.tick_params(axis="y", length=0)
    axR.grid(axis="x", alpha=.15, lw=.7)
    for sp in ("top", "right", "left"):
        axR.spines[sp].set_visible(False)

    fig.tight_layout(w_pad=2.2)
    os.makedirs(os.path.dirname(f"{a.root}/{a.out}"), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.root}/{a.out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}.pdf/.png   {len(pts)} arms   "
          f"tie band {band[0]:.2f}-{band[1]:.2f}%")
    print("Pareto frontier (achievable arms only):")
    for p in front:
        print(f"  {p['tag']:<10}{p['mem']:8.0f} KiB{p['acc']:8.2f}%")

if __name__ == "__main__":
    main()
