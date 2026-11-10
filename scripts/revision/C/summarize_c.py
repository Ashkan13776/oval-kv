"""Block C tables (C1 Video-MME, C2 LVBench) from the per-question shard files.

    python scripts/revision/C/summarize_c.py --run_root results/revision/C/runs/c1_qwen25vl \
        --bench videomme --model "Qwen2.5-VL-7B" --out results/revision/C/c1_qwen25vl

Scored set: questions present in EVERY arm that is shown (timing files `*_timing20*` are
excluded; they are reruns of questions inside the full runs). An arm with fewer questions
than the benchmark total is marked INCOMPLETE and the table says on how many questions it is.

CIs: 95%, cluster bootstrap over VIDEOS (several questions per video), 10,000 draws, seed 0.
Paired CIs: per-question differences arm - reference on the same questions, same bootstrap.
Every arm is paired against both FullKV references (64 and 32 frames). SnapKV is daggered
(reported, never ranked), as the brief requires.

Breakdowns: Video-MME by duration; LVBench by question_type (a question with several types
counts in each, as LVBench reports) and by video type.
Outputs: <out>.md, <out>_acc.csv, <out>_paired.csv (use --out results/revision/C/tables/<name>),
and ../latex/<name>.tex next to the tables dir. Two decimals, as in the paper; no medals.
"""
import argparse, csv, glob, json, os
from collections import defaultdict

import numpy as np

ARMS = [  # (dir, label, tex)
    ("full_n64", "FullKV-64", "FullKV ($N$)"),
    ("full_n32", "FullKV-32", "FullKV ($N/2$)"),
    ("q4key_n64", "4-bit keys", "4-bit keys"),
    ("quest_n64", "Quest", "Quest"),
    ("freekv_n64", "FreeKV", "FreeKV"),
    ("snapkv_n64", "SnapKV†", "SnapKV$^\\dagger$"),
    ("oval_n64", "OVAL", "\\method{}"),
    # C3 (video baselines; "(our port)" in the report)
    ("hf_full", "FullKV-64, HF path", "FullKV ($N$, HF path)"),
    ("vz50_full", "VisionZip 50%", "VisionZip 50\\%"),
    ("vz25_full", "VisionZip 25%", "VisionZip 25\\%"),
    ("vz50_oval", "VisionZip 50% + OVAL", "VisionZip 50\\% + \\method{}"),
    ("infinipot_mem", "InfiniPot-V (mem-matched)", "InfiniPot-V (mem.)"),
    ("infinipot_2k", "InfiniPot-V (2k)", "InfiniPot-V (2k)"),
    ("rekv", "ReKV", "ReKV"),
]
REFS = ["full_n64", "full_n32", "hf_full"]
TOTAL = {"videomme": 2700, "lvbench": 1549}


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "shard*.jsonl"))):
        if "_timing" in f or "_pilot" in f:
            continue
        for l in open(f):
            if l.endswith("\n"):
                r = json.loads(l); out[r["question_id"]] = r
    return out


def boot(dv, n=10000, seed=0):
    vids = list(dv)
    s = np.array([sum(dv[v]) for v in vids], float); c = np.array([len(dv[v]) for v in vids], float)
    idx = np.random.default_rng(seed).integers(0, len(vids), (n, len(vids)))
    st = s[idx].sum(1) / c[idx].sum(1)
    return s.sum() / c.sum(), np.quantile(st, 0.025), np.quantile(st, 0.975)


def groups(r, bench):
    if bench == "videomme":
        return [("duration", r.get("group") or r["duration"])]   # Tab. 5 run files use "duration"
    g = [("video type", r["group"])]
    g += [("question type", t) for t in (r.get("task_type") or "").split("|") if t]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True, nargs="+",
                    help="one or more run dirs; each arm is read from the first root that has it")
    ap.add_argument("--bench", required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", default=[a[0] for a in ARMS])
    a = ap.parse_args()
    meta = {d: (lab, tex) for d, lab, tex in ARMS}
    def find(d):
        for root in a.run_root:
            if glob.glob(os.path.join(root, d, "shard*.jsonl")):
                return os.path.join(root, d)
        return os.path.join(a.run_root[0], d)
    rows = {d: load(find(d)) for d in a.arms}
    rows = {d: r for d, r in rows.items() if r}
    if not rows:
        print("no runs"); return
    common = sorted(set.intersection(*[set(r) for r in rows.values()]))
    any_r = next(iter(rows.values()))
    ORD = {"short": 0, "medium": 1, "long": 2}
    keys = [("all", "overall")] + sorted({g for q in common for g in groups(any_r[q], a.bench)},
                                         key=lambda g: (g[0], ORD.get(g[1], 9), g[1]))
    def sel(r, k):
        return [q for q in common if k == ("all", "overall") or k in groups(r[q], a.bench)]
    acc = {}
    for d, r in rows.items():
        for k in keys:
            dv = defaultdict(list)
            for q in sel(r, k):
                dv[r[q]["videoID"]].append(100 * r[q]["score"])
            acc[d, k] = (boot(dv), sum(len(v) for v in dv.values()))
    paired = {}
    for ref in REFS:
        if ref not in rows:
            continue
        for d in rows:
            if d == ref:
                continue
            for k in keys:
                dv = defaultdict(list)
                for q in sel(rows[ref], k):
                    dv[rows[d][q]["videoID"]].append(100 * (rows[d][q]["score"] - rows[ref][q]["score"]))
                paired[d, ref, k] = boot(dv)
    T = TOTAL[a.bench]
    L = [f"# {a.model} on {'Video-MME (no subtitles)' if a.bench == 'videomme' else 'LVBench'}, "
         "query-agnostic stream protocol", "",
         f"Scored on the {len(common)} questions present in every arm below (benchmark total {T}).", ""]
    L += ["| arm | questions run | status |", "|---|---|---|"]
    for d, r in rows.items():
        L.append(f"| {meta[d][0]} | {len(r)} | {'complete' if len(r) >= T else 'INCOMPLETE'} |")
    for kind in ["overall"] + sorted({k[0] for k in keys if k[0] != "all"}):
        ks = [k for k in keys if (k[0] == "all") == (kind == "overall") and (kind == "overall" or k[0] == kind)]
        L += ["", f"## Accuracy (%), {kind}, 95% CI (bootstrap over videos)", "",
              "| arm | " + " | ".join(f"{k[1]} (n={acc[next(iter(rows)), k][1]})" for k in ks) + " |",
              "|---|" + "---|" * len(ks)]
        for d in rows:
            L.append(f"| {meta[d][0]} | " + " | ".join(
                f"{acc[d, k][0][0]:.2f} [{acc[d, k][0][1]:.2f}, {acc[d, k][0][2]:.2f}]" for k in ks) + " |")
        for ref in REFS:
            if ref not in rows:
                continue
            L += ["", f"Paired difference vs {meta[ref][0]} (pp), 95% CI; * = CI excludes 0", "",
                  "| arm | " + " | ".join(k[1] for k in ks) + " |", "|---|" + "---|" * len(ks)]
            for d in rows:
                if d == ref:
                    continue
                L.append(f"| {meta[d][0]} | " + " | ".join(
                    "{:+.2f} [{:+.2f}, {:+.2f}]{}".format(*paired[d, ref, k],
                    "*" if paired[d, ref, k][1] > 0 or paired[d, ref, k][2] < 0 else "") for k in ks) + " |")
    L += ["", "†: SnapKV is reported, never ranked."]
    if any(d in rows for d in ("vz50_full", "vz25_full", "vz50_oval", "infinipot_mem", "infinipot_2k", "rekv")):
        L += ["VisionZip, InfiniPot-V and ReKV are our ports (see REPORT_C.md, C3)."]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    with open(a.out + "_acc.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "bench", "arm", "breakdown", "group", "n", "acc", "lo", "hi", "complete"])
        for (d, k), ((m, lo, hi), n) in acc.items():
            w.writerow([a.model, a.bench, meta[d][0], k[0], k[1], n, f"{m:.3f}", f"{lo:.3f}", f"{hi:.3f}", len(rows[d]) >= T])
    with open(a.out + "_paired.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "bench", "arm", "reference", "breakdown", "group", "diff", "lo", "hi"])
        for (d, ref, k), (m, lo, hi) in paired.items():
            w.writerow([a.model, a.bench, meta[d][0], meta[ref][0], k[0], k[1], f"{m:.3f}", f"{lo:.3f}", f"{hi:.3f}"])
    # LaTeX: overall + first breakdown; FullKV rows are references, SnapKV daggered; no medals
    ks = [k for k in keys if k[0] != "question type"]
    tex = ["\\begin{tabular}{l|" + "c" * len(ks) + "}", "\\toprule",
           "\\textbf{Method} & " + " & ".join(k[1].capitalize() for k in ks) + " \\\\", "\\midrule"]
    for d in rows:
        tex.append(meta[d][1] + " & " + " & ".join(f"{acc[d, k][0][0]:.2f}" for k in ks) + " \\\\")
        if d in REFS and not any(x in REFS for x in list(rows)[list(rows).index(d) + 1:]):
            tex.append("\\midrule")
    tex += ["\\bottomrule", "\\end{tabular}"]
    td = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.out))), "latex")
    os.makedirs(td, exist_ok=True)
    open(os.path.join(td, os.path.basename(a.out) + ".tex"), "w").write("\n".join(tex) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
