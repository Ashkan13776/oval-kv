"""Video-MME table, paired CIs and LaTeX, from the per-sample logs.

Rows (vision/runs/<dir>): accuracy by duration and overall (on the questions every
row has), KV memory as % of FullKV-N (vision/memory.py), visual tokens, frames.
Paired 95% CIs: per-question differences (same question, same inputs), bootstrap
over VIDEOS (Video-MME asks 3 questions per video, so questions are not
independent; lmms-eval clusters on videoID too), 10,000 draws.

    python vision/summarize.py          # -> results/vision/summary.md, results/vision/table.tex
"""
import json, os, sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import memory as MEM
from compare import load

RUNS = [  # (label, dir, latex label)
    ("FullKV, N frames", "row1_full_n64", "FullKV ($N$)"),
    ("FullKV, N/2 frames", "row2_full_n32", "FullKV ($N/2$)"),
    ("4-bit keys", "row6_q4key_n64", "4-bit keys"),
    ("FreeKV", "row4_freekv_n64", "FreeKV"),
    ("Quest", "row5_quest_n64", "Quest"),
    ("OVAL keys-as-basis", "row7_ovalkeys_n64", "\\method{} (keys)"),
    ("OVAL full", "row3_ovalfull_n64", "\\method{}"),
]
PAIRS = [("OVAL full", "FullKV, N frames"), ("OVAL full", "FullKV, N/2 frames"), ("OVAL full", "FreeKV")]
DUR = ["short", "medium", "long"]
OUT = os.path.join(HERE, "..", "results", "vision")


def boot(diff_by_video, n=10000, seed=0):
    vids = list(diff_by_video)
    sums = np.array([sum(diff_by_video[v]) for v in vids]); cnts = np.array([len(diff_by_video[v]) for v in vids])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vids), (n, len(vids)))
    stats = sums[idx].sum(1) / cnts[idx].sum(1)
    return sums.sum() / cnts.sum(), np.quantile(stats, 0.025), np.quantile(stats, 0.975)


def main():
    rows = {}
    for lab, d, tex in RUNS:
        p = os.path.join(HERE, "runs", d)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "args.json")):
            rows[lab] = (load(p), p, tex)
    common = set.intersection(*[set(r[0]) for r in rows.values()]) if rows else set()
    ref = None
    lines = ["# Video-MME (no subtitles), LLaVA-Video-7B-Qwen2, stream protocol", "",
             f"Questions scored in every row: {len(common)} of 2700.", "",
             "| Row | Frames | Visual tokens | Short | Medium | Long | Overall | KV memory (% FullKV-N, per layer) | model |",
             "|---|---|---|---|---|---|---|---|---|"]
    table = {}
    for lab, (r, p, tex) in rows.items():
        acc = {k: 100 * np.mean([r[q]["score"] for q in common if r[q]["duration"] == k] or [np.nan]) for k in DUR}
        acc["overall"] = 100 * np.mean([r[q]["score"] for q in common])
        rule, fr, L, layer, model = MEM.row(p)
        if ref is None:
            ref = (layer, model)
        vt = int(np.median([r[q]["n_visual"] for q in common]))
        mem, memm = 100 * layer / ref[0], 100 * model / ref[1]
        table[lab] = (acc, mem, fr, vt, tex)
        lines.append(f"| {lab} | {fr} | {vt} | {acc['short']:.1f} | {acc['medium']:.1f} | {acc['long']:.1f} | "
                     f"{acc['overall']:.1f} | {mem:.1f} | {memm:.1f} |")
    lines += ["", "## Paired differences (percentage points), 95% CI, bootstrap over videos", "",
              "| Pair | Short | Medium | Long | Overall |", "|---|---|---|---|---|"]
    for a, b in PAIRS:
        if a not in rows or b not in rows:
            continue
        ra, rb = rows[a][0], rows[b][0]
        cells = []
        for k in DUR + ["overall"]:
            dv = defaultdict(list)
            for q in common:
                if k == "overall" or ra[q]["duration"] == k:
                    dv[ra[q]["videoID"]].append(100 * (ra[q]["score"] - rb[q]["score"]))
            m, lo, hi = boot(dv)
            cells.append(f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]{'*' if lo > 0 or hi < 0 else ''}")
        lines.append(f"| {a} - {b} | " + " | ".join(cells) + " |")
    lines += ["", "*: CI excludes 0."]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    # LaTeX, Table 2 style: FullKV-N is the reference and not ranked; medals over the rest
    keys = ["short", "medium", "long", "overall"]
    ranked = [l for l in table if l != "FullKV, N frames"]
    def fmt(l, k):
        v = round(table[l][0][k], 1)
        if l == "FullKV, N frames":
            return f"{v:.1f}"
        vals = sorted({round(table[x][0][k], 1) for x in ranked}, reverse=True)
        if v == vals[0]:
            return f"\\best{{{v:.1f}}}"
        if len(vals) > 1 and v == vals[1]:
            return f"\\second{{{v:.1f}}}"
        return f"{v:.1f}"
    tex = ["\\begin{tabular}{l|cccc|c}", "\\toprule",
           "\\textbf{Method} & Short & Medium & Long & Overall & KV memory \\\\", "\\midrule"]
    for l in table:
        tex.append(f"{table[l][4]} & " + " & ".join(fmt(l, k) for k in keys) + f" & {table[l][1]:.0f}\\% \\\\")
        if l == "FullKV, N frames":
            tex.append("\\midrule")
    tex += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "table.tex"), "w").write("\n".join(tex) + "\n")


if __name__ == "__main__":
    main()
