"""KV memory of every row of the merged video table (tab:videomme), as % of FullKV (N frames).

Same accounting as vision/memory.py (the paper's memory model, Sec. S5): bf16, d = 128, page 32, per KV head
per layer, at each question's final length L = prefill + streamed question + answer, mean over questions,
whole model (28 layers x 4 KV heads for both LLMs, so per-head/layer bytes scale identically).

Harness rules (FullKV, Quest, FreeKV, SnapKV, sliding window, OVAL) use vision/memory.per_layer; the harness
keeps layer 0 dense, so layer 0 is counted as FullKV for every harness rule (as vision/memory.py).
VisionZip arms are harness runs on the pruned sequence: same rules, at their (shorter) L.
Video-specific cache methods (ported; they act on all layers, layer 0 included):
  InfiniPot-V   exact K,V of the compressed prefill cache + streamed question + answer
                (cache_len + n_stream + n_gen). Storage after prefill; the block-wise prefill
                transiently holds up to one extra block.
  ReKV          stored: the FULL exact cache (needed to answer later questions about the same video; the
                official method offloads it to CPU) + one mean key per frame/temporal group per head per layer.
                resident: only the retrieved blocks + sink/recent + question/answer are on the GPU during
                decoding (reported in the csv, not in the table).

    python scripts/revision/C/memory_c.py   ->  results/revision/C/tables/memory_c.{csv,md}
"""
import csv, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "vision"))
from memory import per_layer, rule_of, N_LAYERS, TOK
from paper_table import MCOLS, MROWS, RUNS, R


def recs(roots, d):
    for r in roots:
        p = os.path.join(RUNS, r, d)
        fs = sorted(f for f in glob.glob(os.path.join(p, "shard*.jsonl")) if "_timing" not in f and "_pilot" not in f)
        if fs:
            out = [json.loads(l) for f in fs for l in open(f) if l.endswith("\n")]
            return p, json.load(open(os.path.join(p, "args.json"))), out
    return None, None, []


def model_bytes(arm, args, r):
    L = r["prefill_len"] + r.get("n_stream", 0) + r["n_gen"]
    if arm.startswith("infinipot"):
        return N_LAYERS * 2 * (r["cache_len"] + r["n_stream"] + r["n_gen"]) * TOK, None
    if arm == "rekv":
        frames = r["n_visual"] // (r.get("tpf") or 210)   # LLaVA-Video: 210 tokens per frame
        stored = N_LAYERS * (2 * L * TOK + frames * TOK)
        resident = N_LAYERS * 2 * (r["cache_len"] + r["n_stream"] + r["n_gen"]) * TOK
        return stored, resident
    rule = rule_of(args)
    return per_layer(rule, L) * (N_LAYERS - 1) + per_layer("full", L), None


def main():
    groups = {}
    for bench, mlab, roots, _ in MCOLS:
        groups[bench, mlab] = roots
    rows = []
    for (bench, mlab), roots in groups.items():
        _, a0, ref = recs(roots, "full_n64")
        refm = sum(model_bytes("full_n64", a0, r)[0] for r in ref) / len(ref)
        for arm, *_ in MROWS:
            p, a, rs = recs(roots, arm)
            if not rs:
                rows.append([bench, mlab, arm, 0, "", "", ""]); continue
            b = [model_bytes(arm, a, r) for r in rs]
            m = sum(x[0] for x in b) / len(b)
            res = sum(x[1] for x in b) / len(b) if b[0][1] is not None else None
            rows.append([bench, mlab, arm, len(rs), f"{m / 2**20:.1f}", f"{100 * m / refm:.1f}",
                         "" if res is None else f"{100 * res / refm:.1f}"])
    os.makedirs(os.path.join(R, "tables"), exist_ok=True)
    with open(os.path.join(R, "tables", "memory_c.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bench", "model", "arm", "n", "kv_MiB_mean", "pct_of_fullN", "rekv_resident_pct"])
        w.writerows(rows)
    keys = list(groups)
    md = ["| arm | " + " | ".join(f"{b} {m}" for b, m in keys) + " |", "|---|" + "---|" * len(keys)]
    for arm, *_ in MROWS:
        cells = []
        for b, m in keys:
            x = next(r for r in rows if r[0] == b and r[1] == m and r[2] == arm)
            cells.append("--" if not x[3] else x[5] + (f" (resident {x[6]})" if x[6] else ""))
        md.append(f"| {arm} | " + " | ".join(cells) + " |")
    open(os.path.join(R, "tables", "memory_c.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
