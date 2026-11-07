"""Video-MME Long split: accuracy vs KV memory -> results/figures/videomme_long_memory.pdf

Reads the per-sample logs directly (never a stored summary), so the figure cannot
drift from the numbers in results/vision/summary.md.

  python vision/plot_longtable.py                 # the compression rows, FullKV-64 reference
  python vision/plot_longtable.py --with-frames   # also FullKV at 44 and 32 frames

Error bars are the PAIRED 95% CI of the difference vs FullKV-64 (cluster bootstrap
over videos, 10,000 draws), placed around each row's absolute accuracy; FullKV-64 is
the reference and carries no bar. Runs with any python that has numpy + matplotlib.
"""
import argparse, os, sys
from collections import defaultdict

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import memory as MEM
from compare import load

REF = "FullKV-64"
# (label, run dir, is an OVAL arm)
ROWS = [("FullKV-64", "row1_full_n64", False),
        ("4-bit keys", "row6_q4key_n64", False),
        ("FreeKV", "row4_freekv_n64", False),
        ("Quest", "row5_quest_n64", False),
        ("SnapKV$^\\dagger$", "row8_snapkv_n64", False),
        ("Sliding window", "row9_sliding_n64", False),
        ("ArkVale", "row10_arkvale_n64", False),
        ("RaaS", "row11_raas_n64", False),
        ("OVAL (keys)", "row7_ovalkeys_n64", True),
        ("OVAL (full)", "row3_ovalfull_n64", True)]
FRAME_ROWS = [("FullKV-44", "row2b_full_n44", False), ("FullKV-32", "row2_full_n32", False)]

OVAL_C, BASE_C, REF_C = "#E8806F", "#5F6B7D", "#1A1A1A"   # same pair as vision/plot_spectra.py
# label offsets in points, tuned so the 100%-memory cluster does not collide
OFF = {"FullKV-64": (-26, 4), "4-bit keys": (-30, -2), "FreeKV": (9, -13), "Quest": (9, 6),
       "SnapKV$^\\dagger$": (0, -16), "OVAL (keys)": (26, -3), "OVAL (full)": (0, -16),
       "FullKV-44": (0, 10), "FullKV-32": (0, 10), "Sliding window": (0, 10),
       "ArkVale": (0, 10), "RaaS": (0, -16)}


def boot(dv, n=10000, seed=0):
    vids = list(dv)
    s = np.array([sum(dv[v]) for v in vids]); c = np.array([len(dv[v]) for v in vids])
    idx = np.random.default_rng(seed).integers(0, len(vids), (n, len(vids)))
    st = s[idx].sum(1) / c[idx].sum(1)
    return s.sum() / c.sum(), np.quantile(st, .025), np.quantile(st, .975)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-frames", action="store_true", help="also plot FullKV at 44 and 32 frames")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "figures", "videomme_long_memory.pdf"))
    a = ap.parse_args()

    rows, wanted = {}, ROWS + (FRAME_ROWS if a.with_frames else [])
    for lab, d, is_oval in wanted:
        p = os.path.join(HERE, "runs", d)
        if not os.path.exists(os.path.join(p, "args.json")):
            print(f"skip {lab}: no run yet"); continue
        r = {q: v for q, v in load(p).items() if v["duration"] == "long"}
        if len(r) < 900:
            print(f"skip {lab}: Long split incomplete ({len(r)}/900)"); continue
        rows[lab] = (r, p, is_oval)
    if REF not in rows:
        sys.exit(f"{REF} is required as the reference")

    common = set.intersection(*[set(r) for r, _, _ in rows.values()])
    ref_layer = MEM.row(rows[REF][1])[3]
    pts = []
    for lab, (r, p, is_oval) in rows.items():
        acc = 100 * np.mean([r[q]["score"] for q in common])
        mem = 100 * MEM.row(p)[3] / ref_layer
        if lab == REF:
            dn = up = 0.0
        else:
            dv = defaultdict(list)
            for q in common:
                dv[r[q]["videoID"]].append(100 * (r[q]["score"] - rows[REF][0][q]["score"]))
            m, lo, hi = boot(dv)
            # the CI is on the DIFFERENCE (point estimate m); carry its half-widths
            # onto the absolute accuracy so the bar is centred on the plotted point
            dn, up = m - lo, hi - m
        pts.append((lab, mem, acc, acc - dn, acc + up, is_oval))

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ref_acc = next(p[2] for p in pts if p[0] == REF)
    ax.axhline(ref_acc, color=REF_C, lw=1, ls="--", alpha=0.5, zorder=1)
    for lab, mem, acc, lo, hi, is_oval in pts:
        c = REF_C if lab == REF else (OVAL_C if is_oval else BASE_C)
        m = "D" if lab == REF else ("o" if is_oval else "s")
        if lab != REF:
            ax.errorbar(mem, acc, yerr=[[acc - lo], [hi - acc]], fmt="none",
                        ecolor=c, elinewidth=1.4, capsize=3, alpha=0.75, zorder=2)
        ax.scatter(mem, acc, marker=m, s=46 if lab == REF else 40, color=c,
                   edgecolors="white", linewidths=0.8, zorder=3)
        ax.annotate(lab, (mem, acc), textcoords="offset points", xytext=OFF.get(lab, (0, 9)),
                    fontsize=8, color="#333333", ha="center")
    ax.set_xlabel("KV memory, % of FullKV-64 (per layer)")
    ax.set_ylabel("Video-MME Long, accuracy (%)")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    xs = [p[1] for p in pts]; ys = [p[3] for p in pts] + [p[4] for p in pts]
    ax.set_xlim(min(xs) - 14, max(xs) + 14)
    ax.set_ylim(min(ys) - 1.0, max(ys) + 1.4)
    h = [plt.Line2D([], [], ls="", marker="D", color=REF_C, label="FullKV-64 (reference)"),
         plt.Line2D([], [], ls="", marker="s", color=BASE_C, label="baselines"),
         plt.Line2D([], [], ls="", marker="o", color=OVAL_C, label="OVAL")]
    ax.legend(handles=h, fontsize=8, frameon=False, loc="lower right")
    ax.set_title(f"Long split, {len(common)} questions; bars = paired 95% CI vs FullKV-64",
                 fontsize=9, color="#333333")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out)
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=200)
    print("wrote", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
