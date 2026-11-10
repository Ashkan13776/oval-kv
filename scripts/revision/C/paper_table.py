"""One combined LaTeX table for the paper: overall accuracy, 2 models x 2 benchmarks, all C1/C2/C3 arms.
Cell = accuracy (%); superscript * = paired 95% CI vs FullKV (N) on the same questions excludes 0
(cluster bootstrap over videos, 10k, seed 0, as summarize_c.py). Arms with fewer questions than the
benchmark total are printed as "--" (incomplete), never filled in.

    python scripts/revision/C/paper_table.py   # -> results/revision/C/latex/paper_table_c.tex (+ .md, .csv)
"""
import csv, os
from collections import defaultdict

from summarize_c import boot, load, TOTAL

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results", "revision", "C")
RUNS = os.path.join(R, "runs")
COLS = [  # (label, bench, roots searched in order)
    ("Qwen2.5-VL, Video-MME", "videomme", ["c3_videomme_qwen25vl", "c1_qwen25vl"]),
    ("LLaVA-Video, Video-MME", "videomme", ["c3_videomme_llava", "tab5_llava_videomme"]),
    ("Qwen2.5-VL, LVBench", "lvbench", ["c3_lvbench_qwen25vl", "c2_lvbench_qwen25vl"]),
    ("LLaVA-Video, LVBench", "lvbench", ["c3_lvbench_llava", "c2_lvbench_llava"]),
]
ROWS = [  # (arm dir, LaTeX label, tokens column, group)
    ("full_n64", "FullKV ($N$ frames)", "100\\%", 0),
    ("full_n32", "FullKV ($N/2$ frames)", "50\\%", 0),
    ("vz50_full", "VisionZip$^\\ddagger$", "50\\%", 1),
    ("infinipot_mem", "InfiniPot-V$^\\ddagger$", "50\\%", 1),
    ("vz25_full", "VisionZip$^\\ddagger$", "25\\%", 1),
    ("infinipot_2k", "InfiniPot-V$^\\ddagger$", "$\\approx$15\\%", 2),
    ("rekv", "ReKV$^\\ddagger$", "$\\approx$15\\%", 2),
    ("snapkv_n64", "SnapKV$^\\dagger$", "$\\approx$15\\%", 2),
    ("quest_n64", "Quest", "$\\approx$15\\%", 2),
    ("freekv_n64", "FreeKV", "$\\approx$15\\%", 2),
    ("q4key_n64", "4-bit keys", "$\\approx$15\\%", 2),
    ("oval_n64", "\\method{}", "$\\approx$15\\%", 2),
]


def get(roots, d):
    for r in roots:
        x = load(os.path.join(RUNS, r, d))
        if x:
            return x
    return {}


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--drop", nargs="*", default=[], help="arm dirs to leave out")
    ap.add_argument("--name", default="paper_table_c"); a = ap.parse_args()
    global ROWS
    ROWS = [r for r in ROWS if r[0] not in a.drop]
    cell, out_csv = {}, []
    for ci, (lab, bench, roots) in enumerate(COLS):
        ref = get(roots, "full_n64")
        for d, *_ in ROWS:
            r = get(roots, d)
            if len(r) < TOTAL[bench] or len(ref) < TOTAL[bench]:
                cell[d, ci] = None; continue
            qs = sorted(set(r) & set(ref))
            dv, dd = defaultdict(list), defaultdict(list)
            for q in qs:
                dv[r[q]["videoID"]].append(100 * r[q]["score"])
                dd[r[q]["videoID"]].append(100 * (r[q]["score"] - ref[q]["score"]))
            m, lo, hi = boot(dv); dm, dlo, dhi = boot(dd)
            sig = d != "full_n64" and (dlo > 0 or dhi < 0)
            cell[d, ci] = (m, sig)
            out_csv.append([lab, d, len(qs), f"{m:.3f}", f"{lo:.3f}", f"{hi:.3f}", f"{dm:.3f}", f"{dlo:.3f}", f"{dhi:.3f}", sig])
    tex = ["\\begin{tabular}{llcccc}", "\\toprule",
           " & & \\multicolumn{2}{c}{Video-MME} & \\multicolumn{2}{c}{LVBench} \\\\",
           "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
           "\\textbf{Method} & \\textbf{Budget} & Qwen2.5-VL & LLaVA-Video & Qwen2.5-VL & LLaVA-Video \\\\",
           "\\midrule"]
    md = ["| Method | Budget | " + " | ".join(c[0] for c in COLS) + " |", "|---|---|" + "---|" * len(COLS)]
    g0 = 0
    for d, lab, tok, g in ROWS:
        if g != g0:
            tex.append("\\midrule"); g0 = g
        cs = []
        for ci in range(len(COLS)):
            c = cell[d, ci]
            cs.append("--" if c is None else f"{c[0]:.2f}" + ("$^{*}$" if c[1] else ""))
        tex.append(f"{lab} & {tok} & " + " & ".join(cs) + " \\\\")
        md.append(f"| {lab} | {tok} | " + " | ".join(x.replace('$^{*}$', '*') for x in cs) + " |")
    tex += ["\\bottomrule", "\\end{tabular}"]
    os.makedirs(os.path.join(R, "latex"), exist_ok=True)
    open(os.path.join(R, "latex", a.name + ".tex"), "w").write("\n".join(tex) + "\n")
    open(os.path.join(R, "tables", a.name + ".md"), "w").write("\n".join(md) + "\n")
    with open(os.path.join(R, "tables", a.name + ".csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["column", "arm", "n", "acc", "lo", "hi", "diff_vs_fullN", "dlo", "dhi", "sig"]); w.writerows(out_csv)
    print("\n".join(md)); print(); print("\n".join(tex))




ABL_ROWS = [  # LLaVA-Video-7B, Video-MME: every Tab. 5 run, by duration (companion to the merged table)
    ("full_n64", "FullKV ($N$ frames)", "100\\%"),
    ("full_n44", "FullKV ($0.69N$ frames)", "69\\%"),
    ("full_n32", "FullKV ($N/2$ frames)", "50\\%"),
    ("sliding_n64", "Sliding window", "$\\approx$15\\%"),
    ("snapkv_n64", "SnapKV$^\\dagger$", "$\\approx$15\\%"),
    ("quest_n64", "Quest", "$\\approx$15\\%"),
    ("freekv_n64", "FreeKV", "$\\approx$15\\%"),
    ("q4key_n64", "4-bit keys", "$\\approx$15\\%"),
    ("ovalkeys_n64", "\\method{} (keys only)", "$\\approx$15\\%"),
    ("oval_n64", "\\method{}", "$\\approx$15\\%"),
]


def ablation():
    """LLaVA-Video Video-MME, all Tab. 5 runs, overall + short / medium / long; * vs FullKV (N)."""
    root = os.path.join(RUNS, "tab5_llava_videomme")
    dur = lambda x: x.get("group") or x["duration"]
    ref = load(os.path.join(root, "full_n64"))
    G = ["overall", "short", "medium", "long"]
    tex = ["\\begin{tabular}{llcccc}", "\\toprule",
           "\\textbf{Method} & \\textbf{Budget} & Overall & Short & Medium & Long \\\\", "\\midrule"]
    md = ["| Method | Budget | " + " | ".join(G) + " |", "|---|---|" + "---|" * len(G)]
    rows_csv = []
    for d, lab, tok in ABL_ROWS:
        r = load(os.path.join(root, d))
        cs = []
        for g in G:
            if len(r) < TOTAL["videomme"]:
                cs.append("--"); continue
            dv, dd = defaultdict(list), defaultdict(list)
            for q in r:
                if g == "overall" or dur(r[q]) == g:
                    dv[r[q]["videoID"]].append(100 * r[q]["score"])
                    dd[r[q]["videoID"]].append(100 * (r[q]["score"] - ref[q]["score"]))
            (m, lo, hi), (dm, dlo, dhi) = boot(dv), boot(dd)
            sig = d != "full_n64" and (dlo > 0 or dhi < 0)
            cs.append(f"{m:.2f}" + ("$^{*}$" if sig else ""))
            rows_csv.append([d, g, f"{m:.3f}", f"{lo:.3f}", f"{hi:.3f}", f"{dm:.3f}", f"{dlo:.3f}", f"{dhi:.3f}", sig])
        tex.append(f"{lab} & {tok} & " + " & ".join(cs) + " \\\\")
        md.append(f"| {lab} | {tok} | " + " | ".join(x.replace('$^{*}$', '*') for x in cs) + " |")
        if d in ("full_n32",):
            tex.append("\\midrule")
    tex += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(R, "latex", "paper_table_c_llava_vmme.tex"), "w").write("\n".join(tex) + "\n")
    open(os.path.join(R, "tables", "paper_table_c_llava_vmme.md"), "w").write("\n".join(md) + "\n")
    with open(os.path.join(R, "tables", "paper_table_c_llava_vmme.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "group", "acc", "lo", "hi", "diff_vs_fullN", "dlo", "dhi", "sig"]); w.writerows(rows_csv)
    print("\n".join(md)); print(); print("\n".join(tex))


if __name__ == "__main__":
    import sys
    if "--ablation" in sys.argv:
        ablation()
    else:
        main()
