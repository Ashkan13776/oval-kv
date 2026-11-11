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


UNRANKED = {"full_n64", "full_n32", "full_n44", "snapkv_n64"}   # references and SnapKV (daggered) are never ranked


def ranks(vals):
    """vals: {arm: value rounded to 2 dp}; competition ranking (ties share a rank)."""
    v = sorted(set(vals.values()), reverse=True)
    out, r = {}, 1
    for x in v:
        tied = [d for d in vals if vals[d] == x]
        for d in tied:
            out[d] = r
        r += len(tied)
    return out


def mark(txt, rank):
    return {1: f"\\topone{{{txt}}}", 2: f"\\toptwo{{{txt}}}", 3: f"\\topthree{{{txt}}}"}.get(rank, txt)


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
    rk = {}
    for ci in range(len(COLS)):
        vals = {d: round(cell[d, ci][0], 2) for d, *_ in ROWS if d not in UNRANKED and cell[d, ci] is not None}
        for d, r in ranks(vals).items():
            rk[d, ci] = r
    g0 = 0
    for d, lab, tok, g in ROWS:
        if g != g0:
            tex.append("\\midrule"); g0 = g
        cs = []
        for ci in range(len(COLS)):
            c = cell[d, ci]
            cs.append("--" if c is None else mark(f"{c[0]:.2f}" + ("$^{*}$" if c[1] else ""), rk.get((d, ci))))
        tex.append(f"{lab} & {tok} & {mem.get(d, '--')} & " + " & ".join(cs) + " \\\\")
        md.append(f"| {lab} | {tok} | {mem.get(d, '--')} | " + " | ".join(x.replace('$^{*}$', '*') for x in cs) + " |")
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


def ablation(drop=()):
    """LLaVA-Video Video-MME, all Tab. 5 runs, overall + short / medium / long; * vs FullKV (N);
    top-3 per column among the ranked arms marked with \\topone / \\toptwo / \\topthree."""
    root = os.path.join(RUNS, "tab5_llava_videomme")
    dur = lambda x: x.get("group") or x["duration"]
    ref = load(os.path.join(root, "full_n64"))
    G = ["overall", "short", "medium", "long"]
    rows = [r for r in ABL_ROWS if r[0] not in drop]
    cell, rows_csv = {}, []
    for d, lab, tok in rows:
        r = load(os.path.join(root, d))
        for g in G:
            if len(r) < TOTAL["videomme"]:
                cell[d, g] = None; continue
            dv, dd = defaultdict(list), defaultdict(list)
            for q in r:
                if g == "overall" or dur(r[q]) == g:
                    dv[r[q]["videoID"]].append(100 * r[q]["score"])
                    dd[r[q]["videoID"]].append(100 * (r[q]["score"] - ref[q]["score"]))
            (m, lo, hi), (dm, dlo, dhi) = boot(dv), boot(dd)
            sig = d != "full_n64" and (dlo > 0 or dhi < 0)
            cell[d, g] = (m, sig)
            rows_csv.append([d, g, f"{m:.3f}", f"{lo:.3f}", f"{hi:.3f}", f"{dm:.3f}", f"{dlo:.3f}", f"{dhi:.3f}", sig])
    rk = {}
    for g in G:
        vals = {d: round(cell[d, g][0], 2) for d, *_ in rows if d not in UNRANKED and cell[d, g] is not None}
        for d, r in ranks(vals).items():
            rk[d, g] = r
    tex = ["\\begin{tabular}{llcccc}", "\\toprule",
           "\\textbf{Method} & \\textbf{Budget} & Overall & Short & Medium & Long \\\\", "\\midrule"]
    md = ["| Method | Budget | " + " | ".join(G) + " |", "|---|---|" + "---|" * len(G)]
    for d, lab, tok in rows:
        cs = []
        for g in G:
            c = cell[d, g]
            cs.append("--" if c is None else mark(f"{c[0]:.2f}" + ("$^{*}$" if c[1] else ""), rk.get((d, g))))
        tex.append(f"{lab} & {tok} & {mem.get(d, '--')} & " + " & ".join(cs) + " \\\\")
        md.append(f"| {lab} | {tok} | {mem.get(d, '--')} | " + " | ".join(x.replace('$^{*}$', '*') for x in cs) + " |")
        if d == "full_n32":
            tex.append("\\midrule")
    tex += ["\\bottomrule", "\\end{tabular}"]
    name = "paper_table_c_llava_vmme" + ("_paper" if "ovalkeys_n64" in drop else "_no4bit" if "q4key_n64" in drop else "")
    open(os.path.join(R, "latex", name + ".tex"), "w").write("\n".join(tex) + "\n")
    open(os.path.join(R, "tables", name + ".md"), "w").write("\n".join(md) + "\n")
    with open(os.path.join(R, "tables", name + ".csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "group", "acc", "lo", "hi", "diff_vs_fullN", "dlo", "dhi", "sig"]); w.writerows(rows_csv)
    print("\n".join(md)); print(); print("\n".join(tex))


# ---------------------------------------------------------------------------------------------
# Merged paper table (2026-11-10): Video-MME by duration for both models + LVBench overall; all rows of
# the former main and duration tables. Top-3 marks only within the matched ~15% block (SnapKV excluded).
MCOLS = [  # (bench, model label, roots, split)
    *[("videomme", "Qwen2.5-VL", ["c3_videomme_qwen25vl", "c1_qwen25vl"], g) for g in ("short", "medium", "long", "overall")],
    *[("videomme", "LLaVA-Video", ["c3_videomme_llava", "tab5_llava_videomme"], g) for g in ("short", "medium", "long", "overall")],
    ("lvbench", "Qwen2.5-VL", ["c3_lvbench_qwen25vl", "c2_lvbench_qwen25vl"], "overall"),
    ("lvbench", "LLaVA-Video", ["c3_lvbench_llava", "c2_lvbench_llava"], "overall"),
]
MROWS = [  # (arm, label, budget, block); block 2 = matched ~15% (ranked)
    ("full_n64", "FullKV ($N$ frames)", "100\\%", 0),
    ("full_n44", "FullKV ($0.69N$ frames)", "69\\%", 0),
    ("full_n32", "FullKV ($N/2$ frames)", "50\\%", 0),
    ("vz50_full", "VisionZip$^\\ddagger$", "50\\%", 1),
    ("infinipot_mem", "InfiniPot-V$^\\ddagger$", "50\\%", 1),
    ("vz25_full", "VisionZip$^\\ddagger$", "25\\%", 1),
    ("sliding_n64", "Sliding window", "$\\approx$15\\%", 2),
    ("infinipot_2k", "InfiniPot-V$^\\ddagger$", "$\\approx$15\\%", 2),
    ("rekv", "ReKV$^\\ddagger$", "$\\approx$15\\%", 2),
    ("snapkv_n64", "SnapKV$^\\dagger$", "$\\approx$15\\%", 2),
    ("quest_n64", "Quest", "$\\approx$15\\%", 2),
    ("freekv_n64", "FreeKV", "$\\approx$15\\%", 2),
    ("oval_n64", "\\method{}", "$\\approx$15\\%", 2),
]


def memcol():
    """KV-memory column: stored KV cache per question (MiB, whole model = memory_c's per-KV-head bytes x 4 KV heads),
    "Qwen2.5-VL / LLaVA-Video" on Video-MME (LVBench differs by at most 2.4%, asserted < 2.5%), from memory_c.csv.
    Also writes tables/video_pareto.csv (every setting x arm: MiB, overall acc)."""
    p = os.path.join(R, "tables", "memory_c.csv")
    if not os.path.exists(p):
        return {}
    mib = {}
    for r in csv.DictReader(open(p)):
        if r["kv_MiB_mean"]:
            mib[r["bench"], r["model"], r["arm"]] = 4 * float(r["kv_MiB_mean"])
    for (b, m, a), x in mib.items():
        y = mib.get(("lvbench" if b == "videomme" else "videomme", m, a))
        assert y is None or abs(x - y) / y < 0.025, (b, m, a, x, y)
    out = {}
    for a in {k[2] for k in mib}:
        q, l = mib.get(("videomme", "Qwen2.5-VL", a)), mib.get(("videomme", "LLaVA-Video", a))
        f = lambda v: "--" if v is None else f"{v:.0f}"
        out[a] = f"{f(q)}\\,/\\,{f(l)}" + ("$^{\\S}$" if a == "rekv" else "")
    return out


def merged():
    dur = lambda x: x.get("group") or x.get("duration")
    cell, rows_csv = {}, []
    for ci, (bench, mlab, roots, split) in enumerate(MCOLS):
        ref = get(roots, "full_n64")
        for d, *_ in MROWS:
            r = get(roots, d)
            if len(r) < TOTAL[bench] or len(ref) < TOTAL[bench]:
                cell[d, ci] = None; continue
            dv, dd = defaultdict(list), defaultdict(list)
            for q in set(r) & set(ref):
                if split == "overall" or dur(r[q]) == split:
                    dv[r[q]["videoID"]].append(100 * r[q]["score"])
                    dd[r[q]["videoID"]].append(100 * (r[q]["score"] - ref[q]["score"]))
            (m, lo, hi), (dm, dlo, dhi) = boot(dv), boot(dd)
            sig = d != "full_n64" and (dlo > 0 or dhi < 0)
            cell[d, ci] = (m, sig)
            rows_csv.append([bench, mlab, split, d, sum(map(len, dv.values())), f"{m:.3f}", f"{lo:.3f}", f"{hi:.3f}",
                             f"{dm:.3f}", f"{dlo:.3f}", f"{dhi:.3f}", sig])
    rk = {}
    for ci in range(len(MCOLS)):
        vals = {d: round(cell[d, ci][0], 2) for d, _, _, blk in MROWS
                if blk == 2 and d not in UNRANKED and cell[d, ci] is not None}
        for d, r in ranks(vals).items():
            rk[d, ci] = r
    S = {"short": "S", "medium": "M", "long": "L", "overall": "All"}
    mem = memcol()
    tex = ["\\begin{tabular}{@{}lll" + "cccc" * 2 + "cc@{}}", "\\toprule",
           " & & & \\multicolumn{8}{c}{Video-MME} & \\multicolumn{2}{c}{LVBench} \\\\",
           "\\cmidrule(lr){4-11}\\cmidrule(l){12-13}",
           " & & & \\multicolumn{4}{c}{Qwen2.5-VL} & \\multicolumn{4}{c}{LLaVA-Video} & Qwen2.5-VL & LLaVA-Video \\\\",
           "\\cmidrule(lr){4-7}\\cmidrule(lr){8-11}",
           "\\textbf{Method} & \\textbf{Budget} & \\textbf{KV (MiB)} & " + " & ".join(S[c[3]] for c in MCOLS[:8]) + " & All & All \\\\",
           "\\midrule"]
    md = ["| Method | Budget | KV mem. | " + " | ".join(f"{c[0]} {c[1]} {c[3]}" for c in MCOLS) + " |", "|---|---|---|" + "---|" * len(MCOLS)]
    b0 = 0
    for d, lab, tok, blk in MROWS:
        if blk != b0:
            tex.append("\\midrule"); b0 = blk
        cs = []
        for ci in range(len(MCOLS)):
            c = cell[d, ci]
            cs.append("--" if c is None else mark(f"{c[0]:.2f}" + ("$^{*}$" if c[1] else ""), rk.get((d, ci))))
        tex.append(f"{lab} & {tok} & {mem.get(d, '--')} & " + " & ".join(cs) + " \\\\")
        md.append(f"| {lab} | {tok} | {mem.get(d, '--')} | " + " | ".join(x.replace('$^{*}$', '*') for x in cs) + " |")
    tex += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(R, "latex", "paper_table_c_merged.tex"), "w").write("\n".join(tex) + "\n")
    open(os.path.join(R, "tables", "paper_table_c_merged.md"), "w").write("\n".join(md) + "\n")
    with open(os.path.join(R, "tables", "paper_table_c_merged.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["bench", "model", "split", "arm", "n", "acc", "lo", "hi", "diff_vs_fullN", "dlo", "dhi", "sig"])
        w.writerows(rows_csv)
    print("\n".join(tex))


if __name__ == "__main__":
    import sys
    if "--merged" in sys.argv:
        merged()
    elif "--ablation" in sys.argv:
        # --paper: the version in the paper (no 4-bit keys, no OVAL keys-only; author decision 2026-11-10)
        ablation(drop=("q4key_n64", "ovalkeys_n64") if "--paper" in sys.argv else
                 ("q4key_n64",) if "--no4bit" in sys.argv else ())
    else:
        main()
