"""LIBERO-Recall table: accuracy (%) by question type and overall, both models, every arm, plus KV memory.
Same statistics as the video table: cluster bootstrap over sessions (10k, seed 0); * = paired 95% CI vs
FullKV (N) excludes 0; ranking (\\topone/\\toptwo/\\topthree) within the matched ~15% block, SnapKV excluded.
Arms with fewer than 600 answered questions print "--" (incomplete). Also writes OVAL - X paired differences.

    python scripts/revision/C/lr_table.py  ->  results/revision/C/{latex/lr_table{,_main}.tex, tables/lr_table.{md,csv}, tables/lr_pairs.csv, tables/lr_pareto.csv}
KV memory: stored KV-cache MiB per question, whole model (memory_c.model_bytes x 4 KV heads), at the final length.
Pareto front (lr_pareto.csv): arms not dominated in (KV MiB, overall acc.), SnapKV (upper bound) excluded.
"""
import csv, os
from collections import defaultdict

from summarize_c import boot, load
from paper_table import ranks, mark, R, RUNS, UNRANKED
from memory_c import model_bytes

TOTAL = 600
MODELS = [("Qwen2.5-VL", "lr_qwen25vl"), ("LLaVA-Video", "lr_llava")]
TYPES = ["present", "absent", "first", "last", "after", "before", "overall"]
TLAB = {"present": "Pres.", "absent": "Abs.", "first": "First", "last": "Last", "after": "After", "before": "Before", "overall": "All"}
ROWS = [  # (arm, label, budget, block)
    ("full_n64", "FullKV ($N$)", "100\\%", 0),
    ("full_n32", "FullKV ($N/2$)", "50\\%", 0),
    ("vz50_full", "VisionZip$^\\ddagger$", "50\\%", 1),
    ("infinipot_mem", "InfiniPot-V$^\\ddagger$", "50\\%", 1),
    ("vz25_full", "VisionZip$^\\ddagger$", "25\\%", 1),
    ("infinipot_2k", "InfiniPot-V$^\\ddagger$", "$\\approx$15\\%", 2),
    ("rekv", "ReKV$^\\ddagger$", "$\\approx$15\\%", 2),
    ("snapkv_n64", "SnapKV$^\\dagger$", "$\\approx$15\\%", 2),
    ("quest_n64", "Quest", "$\\approx$15\\%", 2),
    ("freekv_n64", "FreeKV", "$\\approx$15\\%", 2),
    ("oval_n64", "\\method{}", "$\\approx$15\\%", 2),
    ("vz50_oval", "VisionZip$^\\ddagger$ + \\method{}", "50\\% / $\\approx$15\\%", 3),
]
SHOW_COMBO = False   # VisionZip-50% + OVAL: run, but not in the paper table/figure yet
if not SHOW_COMBO:
    ROWS = [r for r in ROWS if r[0] != "vz50_oval"]
NOFRONT = {"snapkv_n64"}   # run without eviction -> accuracy is an upper bound; kept off the Pareto front
KV_HEADS = 4   # Qwen2.5-VL-7B and LLaVA-Video-7B-Qwen2: 28 layers x 4 KV heads; model_bytes is per KV head


def qtype(r):
    return r["task_type"].split("|")[0]


def mem_mib(root, arm):
    """Mean stored KV bytes per question, whole model, in MiB (None if the arm is incomplete)."""
    import json
    d = os.path.join(RUNS, root, arm)
    rs = list(load(d).values())
    if len(rs) < TOTAL:
        return None
    a = json.load(open(os.path.join(d, "args.json")))
    return KV_HEADS * sum(model_bytes(arm, a, r)[0] for r in rs) / len(rs) / 2**20


def front(pts):
    """Arms not dominated in (memory down, accuracy up). pts: {arm: (mib, acc)}."""
    return {a for a, (m, x) in pts.items()
            if not any(m2 <= m and x2 >= x and (m2 < m or x2 > x) for b, (m2, x2) in pts.items() if b != a)}


def main():
    import json
    cell, mem, pairs, rows_csv = {}, {}, [], []
    for mlab, root in MODELS:
        ref = load(os.path.join(RUNS, root, "full_n64"))
        ok_ref = len(ref) >= TOTAL
        arms = {d: load(os.path.join(RUNS, root, d)) for d, *_ in ROWS}
        for d, *_ in ROWS:
            r = arms[d]
            mem[mlab, d] = mem_mib(root, d)
            for t in TYPES:
                if len(r) < TOTAL or not ok_ref:
                    cell[mlab, t, d] = None; continue
                dv, dd = defaultdict(list), defaultdict(list)
                for q in set(r) & set(ref):
                    if t == "overall" or qtype(r[q]) == t:
                        dv[r[q]["videoID"]].append(100 * r[q]["score"])
                        dd[r[q]["videoID"]].append(100 * (r[q]["score"] - ref[q]["score"]))
                (m, lo, hi), (dm, dlo, dhi) = boot(dv), boot(dd)
                sig = d != "full_n64" and (dlo > 0 or dhi < 0)
                cell[mlab, t, d] = (m, sig)
                rows_csv.append([mlab, t, d, sum(map(len, dv.values())), f"{m:.3f}", f"{lo:.3f}", f"{hi:.3f}",
                                 f"{dm:.3f}", f"{dlo:.3f}", f"{dhi:.3f}", sig])
        o = arms["oval_n64"]
        for d, *_ in ROWS:
            x = arms[d]
            if d == "oval_n64" or len(o) < TOTAL or len(x) < TOTAL:
                continue
            dd = defaultdict(list)
            for q in set(o) & set(x):
                dd[o[q]["videoID"]].append(100 * (o[q]["score"] - x[q]["score"]))
            dm, dlo, dhi = boot(dd)
            pairs.append([mlab, f"OVAL - {d}", f"{dm:.3f}", f"{dlo:.3f}", f"{dhi:.3f}", dlo > 0 or dhi < 0])
    rk = {}
    for mlab, _ in MODELS:
        for t in TYPES:
            vals = {d: round(cell[mlab, t, d][0], 2) for d, _, _, b in ROWS
                    if b == 2 and d not in UNRANKED and cell[mlab, t, d] is not None}
            for d, x in ranks(vals).items():
                rk[mlab, t, d] = x

    def memtxt(m, d):
        v = mem.get((m, d))
        return "--" if v is None else f"{v:.0f}" + ("$^{\\S}$" if d == "rekv" else "")

    def acc(m, t, d):
        c = cell[m, t, d]
        return "--" if c is None else mark(f"{c[0]:.2f}" + ("$^{*}$" if c[1] else ""), rk.get((m, t, d)))

    # Pareto front per model (overall accuracy vs stored KV MiB); SnapKV (upper bound) excluded
    par = []
    for m, _ in MODELS:
        pts = {d: (mem[m, d], cell[m, "overall", d][0]) for d, *_ in ROWS
               if d not in NOFRONT and mem.get((m, d)) is not None and cell[m, "overall", d] is not None}
        f = front(pts)
        for d, *_ in ROWS:
            c = cell[m, "overall", d]
            if c is None or mem.get((m, d)) is None:
                continue
            lo, hi = next((float(x[5]), float(x[6])) for x in rows_csv if x[0] == m and x[1] == "overall" and x[2] == d)
            par.append([m, d, f"{mem[m, d]:.1f}", f"{c[0]:.2f}", f"{lo:.2f}", f"{hi:.2f}", d in f, d in NOFRONT])

    # compact main-paper table: per model, stored KV (MiB) and overall accuracy
    comp = ["\\begin{tabular}{@{}ll" + "rc" * len(MODELS) + "@{}}", "\\toprule",
            " & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{m}}}" for m, _ in MODELS) + " \\\\",
            "\\cmidrule(lr){3-4}\\cmidrule(l){5-6}",
            "\\textbf{Method} & \\textbf{Budget} & " + " & ".join("KV (MiB) & Acc." for _ in MODELS) + " \\\\", "\\midrule"]
    n = len(TYPES)
    tex = ["\\begin{tabular}{@{}ll" + "c" * (2 * n) + "@{}}", "\\toprule",
           " & & " + " & ".join(f"\\multicolumn{{{n}}}{{c}}{{{m}}}" for m, _ in MODELS) + " \\\\",
           f"\\cmidrule(lr){{3-{2 + n}}}\\cmidrule(l){{{3 + n}-{2 + 2 * n}}}",
           "\\textbf{Method} & \\textbf{Budget} & " + " & ".join(TLAB[t] for _ in MODELS for t in TYPES) + " \\\\",
           "\\midrule"]
    md = ["| Method | Budget | " + " | ".join(f"{m} KV MiB | {m} " + " | ".join(f"{m} {t}" for t in TYPES) for m, _ in MODELS) + " |",
          "|---|---|" + "---|" * (2 * (n + 1))]
    b0 = 0
    for d, lab, bud, blk in ROWS:
        if blk != b0:
            tex.append("\\midrule"); comp.append("\\midrule"); b0 = blk
        tex.append(f"{lab} & {bud} & " + " & ".join(acc(m, t, d) for m, _ in MODELS for t in TYPES) + " \\\\")
        comp.append(f"{lab} & {bud} & " + " & ".join(f"{memtxt(m, d)} & {acc(m, 'overall', d)}" for m, _ in MODELS) + " \\\\")
        md.append(f"| {lab} | {bud} | " + " | ".join(memtxt(m, d) + " | " + " | ".join(acc(m, t, d) for t in TYPES) for m, _ in MODELS).replace("$^{*}$", "*") + " |")
    tex += ["\\bottomrule", "\\end{tabular}"]
    comp += ["\\bottomrule", "\\end{tabular}"]
    os.makedirs(os.path.join(R, "latex"), exist_ok=True)
    open(os.path.join(R, "latex", "lr_table_main.tex"), "w").write("\n".join(comp) + "\n")
    with open(os.path.join(R, "tables", "lr_pareto.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "arm", "kv_MiB", "acc", "lo", "hi", "on_front", "upper_bound"]); w.writerows(par)
    open(os.path.join(R, "latex", "lr_table.tex"), "w").write("\n".join(tex) + "\n")
    open(os.path.join(R, "tables", "lr_table.md"), "w").write("\n".join(md) + "\n")
    with open(os.path.join(R, "tables", "lr_table.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "qtype", "arm", "n", "acc", "lo", "hi", "diff_vs_fullN", "dlo", "dhi", "sig"]); w.writerows(rows_csv)
    with open(os.path.join(R, "tables", "lr_pairs.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "pair", "diff", "lo", "hi", "sig"]); w.writerows(pairs)
    print("\n".join(md))
    for p in par:
        print(*p)
    for p in pairs:
        print(*p)


if __name__ == "__main__":
    main()
