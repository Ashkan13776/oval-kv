"""Compare per-sample logs of two or more runs: accuracy, answer agreement, input hashes."""
import json, sys, glob
def load(d):
    r = {}
    for f in sorted(glob.glob(f"{d}/*.jsonl")):
        for l in open(f):
            if l.endswith("\n"):
                x = json.loads(l); r[x["question_id"]] = x
    return r
if __name__ == "__main__":
    runs = {d: load(d) for d in sys.argv[1:]}
    ref_name, ref = next(iter(runs.items()))
    for d, r in runs.items():
        common = sorted(set(r) & set(ref))
        acc = sum(r[q]["score"] for q in r) / max(len(r), 1)
        agree = sum(r[q]["pred"] == ref[q]["pred"] for q in common)
        same_out = sum(r[q]["output"] == ref[q]["output"] for q in common)
        hashes = all(r[q][k] == ref[q][k] for q in common for k in ("frames_sha1", "pixels_sha1", "ids_sha1"))
        print(f"{d}: n={len(r)} acc={acc:.4f} | vs {ref_name}: same letter {agree}/{len(common)}, "
              f"same text {same_out}/{len(common)}, input hashes identical: {hashes}")
