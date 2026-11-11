"""Iso-memory (128 frames), Qwen2.5-VL-7B, Video-MME Long: stored KV memory and accuracy.

Memory: memory_c.py's model (vision/memory.py per_layer, bf16, d 128, page 32, layer 0 dense for harness
rules), per question at L = prefill + streamed question + answer, mean over questions, MiB for the whole
model (28 layers x 4 KV heads). Accuracy: Long questions answered by every arm shown; paired CIs as
summarize_c.py (cluster bootstrap over videos, 10k, seed 0). An arm with < 900 Long answers is INCOMPLETE.

    python scripts/revision/C/iso128_summary.py [--root results/revision/C/runs/iso128_qwen25vl]
"""
import argparse, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory_c import model_bytes
from summarize_c import boot, load
import json

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results", "revision", "C", "runs")
REF = {"full_n64": "c1_qwen25vl/full_n64", "full_n32": "c1_qwen25vl/full_n32", "oval_n64": "c1_qwen25vl/oval_n64",
       "vz50_full": "c3_videomme_qwen25vl/vz50_full"}
NEW = ["full_n128", "oval_n128", "vz50_full_n128"]
PAIRS = [("oval_n128", "full_n64"), ("vz50_full_n128", "full_n64"), ("oval_n128", "vz50_full_n128"),
         ("full_n128", "full_n64"), ("oval_n128", "oval_n64"), ("vz50_full_n128", "vz50_full"), ("oval_n128", "full_n128")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.join(RUNS, "iso128_qwen25vl"))
    a = ap.parse_args()
    dirs = {k: os.path.join(RUNS, v) for k, v in REF.items()} | {k: os.path.join(a.root, k) for k in NEW}
    D, A = {}, {}
    for k, d in dirs.items():
        if os.path.exists(os.path.join(d, "args.json")):
            D[k] = {q: r for q, r in load(d).items() if (r.get("group") or r.get("duration")) == "long"}
            A[k] = json.load(open(os.path.join(d, "args.json")))
    qs = sorted(set.intersection(*(set(x) for x in D.values() if x)))
    print(f"common Long questions: {len(qs)}" + ("" if len(qs) == 900 else "  (INCOMPLETE: < 900)"))
    ref = sum(model_bytes("full_n64", A["full_n64"], D["full_n64"][q])[0] for q in qs) / len(qs)
    print("| arm | Long answered | stored KV MiB / question (mean over common q) | % of FullKV-64 | mean L | acc (common q) |")
    print("|---|---|---|---|---|---|")
    for k in D:
        m = sum(model_bytes(k, A[k], D[k][q])[0] for q in qs) / len(qs)
        L = sum(D[k][q]["prefill_len"] + D[k][q]["n_stream"] + D[k][q]["n_gen"] for q in qs) / len(qs)
        acc = 100 * sum(D[k][q]["score"] for q in qs) / len(qs)
        print(f"| {k} | {len(D[k])} | {m / 2**20:.1f} | {100 * m / ref:.1f} | {L:.0f} | {acc:.2f} |")
    print("\npaired (pp, 95% CI over videos):")
    for x, y in PAIRS:
        if x in D and y in D:
            dv = defaultdict(list)
            for q in qs:
                dv[D[x][q]["videoID"]].append(100 * (D[x][q]["score"] - D[y][q]["score"]))
            m, lo, hi = boot(dv)
            print(f"  {x} - {y}: {m:+.2f} [{lo:+.2f}, {hi:+.2f}]{'*' if lo > 0 or hi < 0 else ''}  (n={len(qs)})")


if __name__ == "__main__":
    main()
