"""Video-MME accuracy by task type (existing run files only) and paired OVAL - X differences per task type.

Task type: the `task_type` field of the Video-MME annotation (lmms-lab/Video-MME parquet, 12 types),
joined by question_id (not read from the answer lines, so every run is labelled the same way).
Scored set per model: questions present in every arm listed (an arm with < 2,700 is flagged).
Paired CIs: per-question OVAL - X on the same questions, cluster bootstrap over videos, 10,000 draws,
seed 0 (summarize_c.boot). 12 types x 4 pairs x 2 models = 96 intervals: ~5 would exclude 0 by chance.

    python scripts/revision/C/tasktype_c.py  ->  results/revision/C/tables/tasktype_c.{md,csv}
"""
import csv, os, sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_c import boot, load

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results", "revision", "C")
RUNS = os.path.join(R, "runs")
ANN = os.path.join(os.environ.get("VMME_ROOT", os.path.expanduser("~/vmme")), "videomme/test-00000-of-00001.parquet")
MODELS = {  # label -> arm -> run dir
    "Qwen2.5-VL-7B": {a: f"c1_qwen25vl/{a}" for a in ("full_n64", "full_n32", "oval_n64", "quest_n64", "freekv_n64")}
    | {a: f"c3_videomme_qwen25vl/{a}" for a in ("vz50_full", "vz25_full", "infinipot_mem", "infinipot_2k", "rekv")},
    "LLaVA-Video-7B": {a: f"tab5_llava_videomme/{a}" for a in ("full_n64", "full_n32", "oval_n64", "quest_n64", "freekv_n64")}
    | {a: f"c3_videomme_llava/{a}" for a in ("vz50_full", "vz25_full", "infinipot_mem", "infinipot_2k", "rekv")},
}
LAB = {"full_n64": "FullKV-64", "full_n32": "FullKV-32", "oval_n64": "OVAL", "quest_n64": "Quest", "freekv_n64": "FreeKV",
       "vz50_full": "VZ50", "vz25_full": "VZ25", "infinipot_mem": "InfP-mem", "infinipot_2k": "InfP-2k", "rekv": "ReKV"}
PAIRS = ["vz50_full", "vz25_full", "infinipot_mem", "full_n32"]


def main():
    ann = pd.read_parquet(ANN).set_index("question_id")
    tt = ann.task_type.to_dict()
    md, rows = [], []
    for model, arms in MODELS.items():
        D = {a: load(os.path.join(RUNS, d)) for a, d in arms.items()}
        qs = sorted(set.intersection(*(set(x) for x in D.values())))
        short = {a: len(x) for a, x in D.items() if len(x) < 2700}
        types = sorted({tt[q] for q in qs}, key=lambda t: -sum(tt[q] == t for q in qs))
        md += [f"## {model} (n = {len(qs)} questions in every arm" + (f"; INCOMPLETE arms {short}" if short else "") + ")", "",
               "Accuracy (%) by task type:", "",
               "| task type | n | " + " | ".join(LAB[a] for a in arms) + " |", "|---|---|" + "---|" * len(arms)]
        for t in types + ["ALL"]:
            sel = [q for q in qs if t == "ALL" or tt[q] == t]
            acc = {a: 100 * sum(D[a][q]["score"] for q in sel) / len(sel) for a in arms}
            md.append(f"| {t} | {len(sel)} | " + " | ".join(f"{acc[a]:.1f}" for a in arms) + " |")
            rows += [[model, "acc", t, len(sel), a, f"{acc[a]:.2f}", "", ""] for a in arms]
        md += ["", "Paired OVAL - X (pp, 95% CI, cluster bootstrap over videos; * = CI excludes 0):", "",
               "| task type | n | " + " | ".join(f"OVAL - {LAB[b]}" for b in PAIRS) + " |", "|---|---|" + "---|" * len(PAIRS)]
        for t in types + ["ALL"]:
            sel = [q for q in qs if t == "ALL" or tt[q] == t]
            cells = []
            for b in PAIRS:
                dv = defaultdict(list)
                for q in sel:
                    dv[D["oval_n64"][q]["videoID"]].append(100 * (D["oval_n64"][q]["score"] - D[b][q]["score"]))
                m, lo, hi = boot(dv)
                sig = "*" if lo > 0 or hi < 0 else ""
                cells.append(f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]{sig}")
                rows.append([model, "oval_minus", t, len(sel), b, f"{m:.2f}", f"{lo:.2f}", f"{hi:.2f}"])
            md.append(f"| {t} | {len(sel)} | " + " | ".join(cells) + " |")
        md.append("")
    os.makedirs(os.path.join(R, "tables"), exist_ok=True)
    open(os.path.join(R, "tables", "tasktype_c.md"), "w").write("\n".join(md) + "\n")
    with open(os.path.join(R, "tables", "tasktype_c.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "kind", "task_type", "n", "arm", "value", "ci_lo", "ci_hi"]); w.writerows(rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()
