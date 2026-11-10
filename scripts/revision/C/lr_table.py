"""LIBERO-Recall table: accuracy (%) by question type and overall, both models, every arm, plus KV memory.
Same statistics as the video table: cluster bootstrap over sessions (10k, seed 0); * = paired 95% CI vs
FullKV (N) excludes 0; ranking (\\topone/\\toptwo/\\topthree) within the matched ~15% block, SnapKV excluded.
Arms with fewer than 600 answered questions print "--" (incomplete). Also writes OVAL - X paired differences.

    python scripts/revision/C/lr_table.py  ->  results/revision/C/{latex/lr_table.tex, tables/lr_table.{md,csv}, tables/lr_pairs.csv}
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
]


def qtype(r):
    return r["task_type"].split("|")[0]


def mem_pct(root, arm, ref_bytes):
    import json
    d = os.path.join(RUNS, root, arm)
    rs = list(load(d).values())
    if len(rs) < TOTAL:
        return None
    a = json.load(open(os.path.join(d, "args.json")))
    return 100 * sum(model_bytes(arm, a, r)[0] for r in rs) / len(rs) / ref_bytes


def main():
    import json
    cell, mem, pairs, rows_csv = {}, {}, [], []
    for mlab, root in MODELS:
        ref = load(os.path.join(RUNS, root, "full_n64"))
        ok_ref = len(ref) >= TOTAL
        refb = (sum(model_bytes("full_n64", json.load(open(os.path.join(RUNS, root, "full_n64", "args.json"))), r)[0]
                    for r in ref.values()) / len(ref)) if ok_ref else None
        arms = {d: load(os.path.join(RUNS, root, d)) for d, *_ in ROWS}
        for d, *_ in ROWS:
            r = arms[d]
            mem[mlab, d] = mem_pct(root, d, refb) if ok_ref else None
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

    def memtxt(d):
        v = [mem[m, d] for m, _ in MODELS if mem.get((m, d)) is not None]
        if not v:
            return "--"
        a, b = round(min(v)), round(max(v))
        s = f"{a}\\%" if a == b else f"{a}--{b}\\%"
        return s + ("$^{\\S}$" if d == "rekv" else "")

    n = len(TYPES)
    tex = ["\\begin{tabular}{@{}lll" + "c" * (2 * n) + "@{}}", "\\toprule",
           " & & & " + " & ".join(f"\\multicolumn{{{n}}}{{c}}{{{m}}}" for m, _ in MODELS) + " \\\\",
           f"\\cmidrule(lr){{4-{3 + n}}}\\cmidrule(l){{{4 + n}-{3 + 2 * n}}}",
           "\\textbf{Method} & \\textbf{Budget} & \\textbf{KV mem.} & " + " & ".join(TLAB[t] for _ in MODELS for t in TYPES) + " \\\\",
           "\\midrule"]
    md = ["| Method | Budget | KV mem. | " + " | ".join(f"{m} {t}" for m, _ in MODELS for t in TYPES) + " |",
          "|---|---|---|" + "---|" * (2 * n)]
    b0 = 0
    for d, lab, bud, blk in ROWS:
        if blk != b0:
            tex.append("\\midrule"); b0 = blk
        cs = []
        for m, _ in MODELS:
            for t in TYPES:
                c = cell[m, t, d]
                cs.append("--" if c is None else mark(f"{c[0]:.2f}" + ("$^{*}$" if c[1] else ""), rk.get((m, t, d))))
        tex.append(f"{lab} & {bud} & {memtxt(d)} & " + " & ".join(cs) + " \\\\")
        md.append(f"| {lab} | {bud} | {memtxt(d)} | " + " | ".join(x.replace('$^{*}$', '*') for x in cs) + " |")
    tex += ["\\bottomrule", "\\end{tabular}"]
    os.makedirs(os.path.join(R, "latex"), exist_ok=True)
    open(os.path.join(R, "latex", "lr_table.tex"), "w").write("\n".join(tex) + "\n")
    open(os.path.join(R, "tables", "lr_table.md"), "w").write("\n".join(md) + "\n")
    with open(os.path.join(R, "tables", "lr_table.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "qtype", "arm", "n", "acc", "lo", "hi", "diff_vs_fullN", "dlo", "dhi", "sig"]); w.writerows(rows_csv)
    with open(os.path.join(R, "tables", "lr_pairs.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "pair", "diff", "lo", "hi", "sig"]); w.writerows(pairs)
    print("\n".join(md))
    for p in pairs:
        print(*p)


if __name__ == "__main__":
    main()
