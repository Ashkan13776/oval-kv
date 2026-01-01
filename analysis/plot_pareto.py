"""Pareto front on MATH50 at the published config (B=2048, sink/recent 512).

Two axes: KV memory held per KV head per layer, and avg@8 accuracy.

The figure's main job is to NOT overclaim. Both oracles -- which score with the
true attention distribution and with q_{t-1} -- land inside the same cluster as
Quest and FreeKV, so at this budget there is no headroom for any scoring rule
to win. The tie band drawn across the plot is the span of arms whose paired
95% CI against FreeKV contains zero; nothing inside it is separable, and the
plot says so rather than inviting the reader to rank points within it.

Error bars are the paired bootstrap CI against FreeKV, centred on each arm's
own accuracy. Pairing matters: every arm sees the same 50 problems under the
same 8 seeds, so an unpaired interval would be dominated by problem difficulty
spread (about +/-12 points) rather than by the comparison (about +/-2).

Usage:  python analysis/plot_pareto.py [--out FIG] [--budget 2048]
"""
import argparse, glob, json, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

D = "results/freekv/codec/ds-r1-qwen-7b"
PAT = r"\\boxed{((?:[^{}]|\{[^{}]*\})*)}"
HD, PG, NPAGE = 128, 32, 110          # head dim, page size, mean paged pages
B_K = HD * PG * 2                     # exact keys per page per head, fp16
B_V = B_K

#   key: (label, family, digest bytes/page, keeps exact keys, glob)
ARMS = {
 "q4key":    ("4-bit keys",      "baseline", 2560, False, "{D}-quest/MATH50-*b{B}-avgSM-q4key*"),
 "ours":     ("OVAL r8 + codec", "ours",     2816, False, "{D}-spec_ret/MATH50-*b{B}-avgSM-oval_r8_eta0.0_q0_recon1-*"),
 "ours_sel": ("OVAL r8 index",   "ours",     2816, True,  "{D}-spec_ret/MATH50-*b{B}-avgSM-oval_r8_eta0.0_q0-pQ2*"),
 "ours_q":   ("OVAL r8 no-spec", "ours",     2816, True,  "{D}-quest/MATH50-*b{B}-avgSM-oval_r8_eta0.0_q0-*"),
 "loki":     ("Loki (shared)",   "baseline", 512,  True,  "{D}-quest/MATH50-*b{B}-avgSM-loki*"),
 "centroid": ("Centroid",        "baseline", 256,  True,  "{D}-quest/MATH50-*b{B}-avgSM-centroid*"),
 "shadowkv": ("ShadowKV",        "baseline", 256,  True,  "{D}-quest/MATH50-*b{B}-avgSM-shadowkv*"),
 "rocketkv": ("RocketKV",        "baseline", 256,  True,  "{D}-quest/MATH50-*b{B}-avgSM-rocketkv*"),
 "infllm":   ("InfLLM",          "baseline", 1024, True,  "{D}-quest/MATH50-*b{B}-avgSM-infllm*"),
 "sparq":    ("SparQ",           "baseline", 0,    True,  "{D}-quest/MATH50-*b{B}-avgSM-sparq*"),
 "quest":    ("Quest",           "baseline", 512,  True,  "{D}-quest/MATH50-*b{B}-avgSM-skl1*"),
 "freekv":   ("FreeKV",          "baseline", 512,  True,  "{D}-spec_ret/MATH50-*b{B}-avgSM-pQ2-*"),
 "arkvale":  ("ArkVale",         "baseline", 512,  True,  "{D}-spec_ret/MATH50-*b{B}-avgSM-arkv-*"),
 "pyramid":  ("PyramidKV",       "sched",    512,  True,  "{D}-spec_ret/MATH50-*b{B}-avgSM-pyr*"),
 "tova":     ("TOVA \u2020",        "evict",    4,    False, "{D}-quest/MATH50-*b{B}-avgSM-tova*"),
 "h2o":      ("H2O \u2020",         "evict",    4,    False, "{D}-quest/MATH50-*b{B}-avgSM-h2o*"),
 "snapkv":   ("SnapKV \u2020",      "evict",    0,    False, "{D}-quest/MATH50-*b{B}-avgSM-snapkv*"),
 "expatt":   ("ExpectedAttn",    "baseline", 0,    True,  "{D}-quest/MATH50-*b{B}-avgSM-expatt*"),
 "knorm":    ("K-norm",          "baseline", 2,    True,  "{D}-quest/MATH50-*b{B}-avgSM-knorm*"),
 "raas":     ("RaaS",            "evict",    0,    False, "{D}-raas/MATH50-*b{B}*"),
 "random":   ("Random",          "control",  0,    True,  "{D}-quest/MATH50-*b{B}-avgSM-random*"),
 "sliding":  ("Sliding window",  "control",  0,    False, "{D}-quest/MATH50-s512-r2560-*-b0-avgSM-skl1*"),
 "stream":   ("StreamingLLM",    "control",  0,    False, "{D}-quest/MATH50-s512-r512-*-b0-avgSM-skl1*"),
 "full":     ("Dense",           "dense",    0,    True,  "{D}-full/MATH50-*"),
 "oracle":   ("Oracle",          "ceiling",  0,    True,  "{D}-quest/MATH50-*b{B}-avgSM-oracle*"),
 "oracle_prev": ("Oracle q(t-1)","ceiling",  0,    True,  "{D}-spec_ret/MATH50-*b{B}-avgSM-oracle*"),
}
STYLE = {                      # family -> (colour, marker, size, z)
 "ours":     ("#c2410c", "D", 86, 5),
 "baseline": ("#475569", "o", 46, 3),
 "control":  ("#0891b2", "s", 52, 4),
 "ceiling":  ("#7c3aed", "*", 165, 6),
 "evict":    ("#b45309", "v", 54, 4),
 "sched":    ("#475569", "o", 46, 3),
 "dense":    ("#111827", "P", 78, 6),
}

def load(pat):
    by = {}
    files = glob.glob(pat)
    if not files and "oval_" in pat:
        files = glob.glob(pat.replace("oval_", "oab_"))   # pre-rename results
    for f in sorted(files):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        if len(rows) < 50:
            continue                       # mid-write file, not a result yet
        s = re.search(r"-seed(\d+)", f)
        by[int(s.group(1)) if s else 42] = {
            e["qid"]: int(bool(re.search(PAT, e["pred"]) and
                re.search(PAT, e["pred"]).group(1).replace(" ", "")
                == str(e["answer"]).replace(" ", ""))) for e in rows}
    return by

def memory(tag, exact, digest, budget):
    """KiB per KV head per layer on a mean-length sequence."""
    if tag == "full":
        tot = NPAGE * (B_K + B_V)
    elif tag in ("raas", "h2o", "tova", "snapkv"):
        # eviction: the cache SHRINKS to what survives. Charging these the full
        # cache (as a retrieval arm pays) overstates their cost by ~15% here.
        tot = (1024 + budget) / PG * (B_K + B_V)
    elif tag in ("stream", "sliding"):                  # only a window survives
        tot = 1024 / PG * (B_K + B_V)
    else:
        tot = NPAGE * ((B_K if exact else 0) + B_V + digest)
    return tot / 1024

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--out", default="results/figures/pareto_math50_b2048")
    ap.add_argument("--root", default=".")
    ap.add_argument("--data-root", default=None,
                    help="where the results live, if not alongside the figure")
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
        pts[tag] = dict(tag=tag, label=label, fam=fam, k=len(seeds), M=M,
                        acc=M.mean() * 100, mem=memory(tag, exact, dig, a.budget))
    if "freekv" not in pts:
        sys.exit("no FreeKV reference on disk for this budget")

    ref, fkv = pts["freekv"]["M"], pts["freekv"]["acc"]
    rng = np.random.default_rng(0)
    for p_ in pts.values():
        n = min(p_["M"].shape[1], ref.shape[1]); k = min(p_["M"].shape[0], ref.shape[0])
        d = p_["M"][:k, :n].mean(0) - ref[:k, :n].mean(0)
        b = d[rng.integers(0, n, (8000, n))].mean(1) * 100
        p_["lo"], p_["hi"] = np.percentile(b, [2.5, 97.5])
        p_["tie"] = p_["lo"] <= 0 <= p_["hi"]
    band = (min(p_["acc"] for p_ in pts.values() if p_["tie"]),
            max(p_["acc"] for p_ in pts.values() if p_["tie"]))

    # Pareto frontier over everything deployable: ceilings are not deployable,
    # but the window controls are real methods and belong on the front.
    # h2o/tova/snapkv are implemented here WITHOUT eviction -- they keep the
    # cache and merely stop attending -- so their accuracy is an upper bound on
    # the published methods, which can never recover a discarded token. An
    # upper bound cannot sit on a frontier of achievable points.
    UB = {"h2o", "tova", "snapkv"}
    dep = [p_ for p_ in pts.values()
           if p_["fam"] != "ceiling" and p_["tag"] not in UB]
    front, best = [], -1e9
    for p_ in sorted(dep, key=lambda z: (z["mem"], -z["acc"])):
        if p_["acc"] > best:
            front.append(p_); best = p_["acc"]
    fset = {p_["tag"] for p_ in front}

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(14.2, 7.6), gridspec_kw=dict(width_ratios=[1.08, 1.0]))

    # ---------------- left: the Pareto view ----------------
    axL.axhspan(*band, color="#94a3b8", alpha=.15, zorder=0)
    axL.step([p_["mem"] for p_ in front], [p_["acc"] for p_ in front],
             where="post", color="#c2410c", lw=1.6, alpha=.7, zorder=2)
    axL.scatter([p_["mem"] for p_ in front], [p_["acc"] for p_ in front],
                s=230, facecolors="none", edgecolors="#c2410c", lw=1.6, zorder=7)
    # the big cluster is annotated once instead of 15 overlapping labels
    clu = [p_ for p_ in pts.values() if 1700 <= p_["mem"] <= 1900]
    for p_ in pts.values():
        c, mk, sz, z = STYLE[p_["fam"]]
        lo_abs, hi_abs = fkv + p_["lo"], fkv + p_["hi"]
        axL.errorbar(p_["mem"], p_["acc"],
                     yerr=[[max(0., p_["acc"] - lo_abs)], [max(0., hi_abs - p_["acc"])]],
                     fmt="none", ecolor=c, elinewidth=.9, alpha=.3, capsize=2, zorder=z - 1)
        axL.scatter(p_["mem"], p_["acc"], s=sz, c=c, marker=mk, zorder=z,
                    edgecolors="white", linewidths=.7)
        if p_ not in clu or p_["fam"] in ("ours", "ceiling", "dense"):
            # the ceilings and dense share an x with the exact-key cluster, so
            # fan their labels out instead of stacking them on one point
            off = {"oracle": (-46, 20), "oracle_prev": (46, 34),
                   "full": (-30, -17), "ours": (0, 12)}.get(p_["tag"], (0, 9))
            axL.annotate(p_["label"], (p_["mem"], p_["acc"]),
                         textcoords="offset points", xytext=off,
                         ha="center", fontsize=8.6,
                         fontweight="bold" if p_["fam"] in ("ours", "ceiling") else "normal",
                         color="#1f2937",
                         arrowprops=dict(arrowstyle="-", color="#cbd5e1", lw=.7,
                                         shrinkA=0, shrinkB=3)
                                     if p_["tag"] in ("oracle","oracle_prev","full") else None)
    if clu:
        axL.annotate(f"{len(clu)} methods that keep the keys exact\n"
                     f"({min(p_['mem'] for p_ in clu):.0f}-{max(p_['mem'] for p_ in clu):.0f} KiB, "
                     f"{min(p_['acc'] for p_ in clu):.1f}-{max(p_['acc'] for p_ in clu):.1f}%)"
                     "  \u2192 see right panel",
                     xy=(np.mean([p_["mem"] for p_ in clu]), min(p_["acc"] for p_ in clu)),
                     xytext=(-30, -52), textcoords="offset points", ha="center",
                     fontsize=8.4, color="#475569",
                     arrowprops=dict(arrowstyle="-", color="#94a3b8", lw=.8))
    axL.set_xlabel("KV memory retained  (KiB per KV head per layer)")
    axL.set_ylabel("avg@8 accuracy  (%)")
    axL.set_title("Pareto view: accuracy vs memory", fontsize=11, loc="left")
    axL.grid(alpha=.18, lw=.6)
    for sp in ("top", "right"): axL.spines[sp].set_visible(False)
    axL.legend(handles=[Line2D([], [], ls="", marker=STYLE[f][1],
                               ms=np.sqrt(STYLE[f][2]), color=STYLE[f][0], label=l)
                        for f, l in [("ours", "OVAL (ours)"), ("baseline", "retrieval baseline"),
                                     ("evict", "eviction (\u2020 = no-evict upper bound)"),
                                     ("control", "fixed window"), ("ceiling", "ceiling (oracle)"),
                                     ("dense", "dense attention")]]
               + [Line2D([], [], ls="", marker="o", mfc="none", mec="#c2410c",
                         ms=11, label="Pareto-optimal")],
               loc="lower right", frameon=False, fontsize=8.6)

    # ---------------- right: the ranking that the left panel cannot show ------
    order = sorted(pts.values(), key=lambda z: z["acc"])
    ys = np.arange(len(order))
    axR.axvspan(*band, color="#94a3b8", alpha=.15, zorder=0)
    axR.axvline(fkv, color="#64748b", lw=.9, ls="--", zorder=1)
    for y, p_ in zip(ys, order):
        c = STYLE[p_["fam"]][0]
        axR.plot([fkv + p_["lo"], fkv + p_["hi"]], [y, y], color=c, lw=2.1,
                 alpha=.42, solid_capstyle="round", zorder=2)
        axR.scatter(p_["acc"], y, s=44, c=c, marker=STYLE[p_["fam"]][1],
                    zorder=3, edgecolors="white", linewidths=.6)
    axR.set_yticks(ys)
    STAR = "  \u2605"
    axR.set_yticklabels(
        [p_["label"] + (STAR if p_["tag"] in fset else "") for p_ in order],
        fontsize=8.6)
    for t_, p_ in zip(axR.get_yticklabels(), order):
        t_.set_color(STYLE[p_["fam"]][0])
        if p_["fam"] in ("ours", "ceiling"): t_.set_fontweight("bold")
    axR.set_xlabel("avg@8 accuracy  (%)   —   bar = paired 95% CI vs FreeKV")
    axR.set_title("Ranking with paired intervals  (\u2605 = Pareto-optimal)",
                  fontsize=11, loc="left")
    axR.grid(axis="x", alpha=.18, lw=.6)
    for sp in ("top", "right", "left"): axR.spines[sp].set_visible(False)
    axR.tick_params(axis="y", length=0)

    fig.suptitle("KV-cache retrieval on MATH50 — 26 methods at matched budget   "
                 f"(DS-R1-Qwen-7B, budget {a.budget}, sink 512, recent 512, 8 seeds)",
                 fontsize=12.5, x=.012, ha="left", y=.985)
    fig.text(.012, .013,
             "Both oracles sit inside the tie band: even with perfect knowledge of the true attention distribution there is nothing to win at this budget, because "
             "only ~20% of a\nreasoning trace is ever compressed (onset at token 3072, mean sequence 3869). No ordering within the band is a result \u2014 the separable "
             "findings are the five arms below it.",
             fontsize=8.5, color="#334155", va="bottom")
    fig.tight_layout(rect=[0, .052, 1, .965])
    import os
    os.makedirs(os.path.dirname(f"{a.root}/{a.out}"), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.root}/{a.out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}.pdf/.png  ({len(pts)} arms, tie band {band[0]:.2f}-{band[1]:.2f}%)")
    print("Pareto frontier:")
    for p_ in front:
        print(f"  {p_['tag']:<12}{p_['mem']:8.0f} KiB{p_['acc']:8.2f}%")

if __name__ == "__main__":
    main()
