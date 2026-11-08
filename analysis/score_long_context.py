"""Score the long-context full-pipeline runs (piece 6) and write
results/long_context/table.md, next to the paper's Table 2 columns.

LongBench v2: accuracy over the 503 questions (their `judge` field, as
eval/LongBench2/result.py), Overall and by length Short / Medium / Long.
LongGenBench: CR (completion rate, eval.py's calculate_completion_rate, no
judge needed) and CR x Acc = CR * mean(Acc once, range, periodic) from the
Qwen3-32B judge (eval/LongGenBench/eval.py's CSV). The finished harness
outputs are copied into results/long_context/raw/ (LongGenBench gzipped).
"""
import argparse, csv, glob, gzip, json, os, re, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT

PAPER = {  # results/paper_tables.md, Table 2 (as reported): FreeKV and eta* columns
    ("llama-3.1-chat-8b", "Overall"): (29.22, 29.82), ("llama-3.1-chat-8b", "Short"): (35.00, 35.00),
    ("llama-3.1-chat-8b", "Medium"): (27.44, 28.84), ("llama-3.1-chat-8b", "Long"): (23.15, 25.93),
    ("llama-3.1-chat-8b", "CR"): (78.03, 74.71), ("llama-3.1-chat-8b", "CRxAcc"): (27.62, 28.50),
    ("qwen-2.5-chat-7b", "Overall"): (26.84, 28.03), ("qwen-2.5-chat-7b", "Short"): (34.44, 36.11),
    ("qwen-2.5-chat-7b", "Medium"): (22.33, 22.79), ("qwen-2.5-chat-7b", "Long"): (23.15, 25.93),
    ("qwen-2.5-chat-7b", "CR"): (76.93, 78.54), ("qwen-2.5-chat-7b", "CRxAcc"): (32.81, 32.38),
    ("qwen-2.5-chat-14b", "Overall"): (34.19, 33.00), ("qwen-2.5-chat-14b", "Short"): (41.11, 41.11),
    ("qwen-2.5-chat-14b", "Medium"): (33.49, 30.23), ("qwen-2.5-chat-14b", "Long"): (24.07, 25.93),
    ("qwen-2.5-chat-14b", "CR"): (65.46, 63.92), ("qwen-2.5-chat-14b", "CRxAcc"): (29.39, 28.64),
}
NAMES = {"llama-3.1-chat-8b": "Llama-3.1-8B", "qwen-2.5-chat-7b": "Qwen-2.5-7B", "qwen-2.5-chat-14b": "Qwen-2.5-14B"}


def lb2(fk, model, krank, vrank):
    fs = [f for f in glob.glob(f"{fk}/eval/LongBench2/results/{model}-spec_ret-*oval_r{krank}_etab*_recon1_vr{vrank}_vpca-*.jsonl")
          if "-to" not in os.path.basename(f)]
    if not fs:
        return None
    f = fs[0]
    d = [json.loads(l) for l in open(f) if l.strip()]
    acc = lambda s: 100 * sum(int(p["judge"]) for p in d if s(p)) / max(1, sum(1 for p in d if s(p)))
    return f, dict(n=len(d), Overall=acc(lambda p: True), Short=acc(lambda p: p["length"] == "short"),
                   Medium=acc(lambda p: p["length"] == "medium"), Long=acc(lambda p: p["length"] == "long"),
                   nulls=sum(1 for p in d if p.get("pred") is None))


def parse_blocks(blocks, typ):
    out = {}
    for b in blocks:
        m = re.search(rf"{typ} (\d+)", b)
        if m and int(m.group(1)) not in out:
            out[int(m.group(1))] = b
    return out


def lgb(fk, model, krank, vrank, judge_dir):
    fs = [f for f in glob.glob(f"{fk}/eval/LongGenBench/results_short/{model}-spec_ret-*oval_r{krank}_etab*_recon1_vr{vrank}_vpca*.jsonl")
          if ".part" not in f]
    if not fs:
        return None
    f = fs[0]
    d = json.load(open(f))
    cr = sum(100 * len(set(parse_blocks(x["output_blocks"], x["type"])) & set(range(1, x["number"] + 1))) / x["number"]
             for x in d) / len(d)
    res = dict(n=len(d), CR=cr, capped=sum(1 for x in d if x.get("output_len", 0) >= 16000))
    c = os.path.join(judge_dir, f"{model}.csv")
    if os.path.exists(c):
        row = list(csv.DictReader(open(c)))[-1]
        res["Acc"] = 100 * float(row["Average Accuracy"])
        res["CRxAcc"] = cr * float(row["Average Accuracy"])
        res["judge_CR"] = float(row["Completion Rate"])
    judged = os.path.join(judge_dir, f"{model}.jsonl")
    return (judged if os.path.exists(c) and os.path.exists(judged) else f), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--krank", type=int, default=16)
    ap.add_argument("--vrank", type=int, default=32)
    ap.add_argument("--fk", default=str(ROOT / "third_party/FreeKV/accuracy"))
    ap.add_argument("--judge_dir", default=str(ROOT / "logs/long_context/judge"))
    ap.add_argument("--out", default=str(ROOT / "results/long_context"))
    a = ap.parse_args()
    raw = os.path.join(a.out, "raw"); os.makedirs(raw, exist_ok=True)
    rows, cells = [], {}
    for m in a.models:
        r = lb2(a.fk, m, a.krank, a.vrank)
        if r:
            f, s = r; cells[(m, "lb2")] = s
            shutil.copyfile(f, os.path.join(raw, os.path.basename(f)))
        r = lgb(a.fk, m, a.krank, a.vrank, a.judge_dir)
        if r:
            f, s = r; cells[(m, "lgb")] = s
            with open(f, "rb") as i, gzip.open(os.path.join(raw, os.path.basename(f) + ".gz"), "wb") as o:
                shutil.copyfileobj(i, o)
    fmt = lambda v: "--" if v is None else f"{v:.2f}"
    L = ["# OVAL full pipeline on long context (piece 6)", "",
         f"Keys rank {a.krank} (scoring and reconstruction), values PCA rank {a.vrank}, eta per (layer, KV head) from rule 1 "
         "(one table per model and benchmark, calibrated on held-out data). Generated by "
         "analysis/score_long_context.py. FreeKV and eta* columns: the paper's Table 2 as reported "
         "(eta* = best of five shared etas per cell, retrieval only, exact keys and values).", "",
         "| Model | Benchmark | Metric | FreeKV (paper) | eta* (paper) | **OVAL full pipeline** | n |",
         "|---|---|---|---|---|---|---|"]
    for m in a.models:
        s = cells.get((m, "lb2"))
        for met in ("Overall", "Short", "Medium", "Long"):
            fk_, es = PAPER.get((m, met), (None, None))
            L.append(f"| {NAMES.get(m, m)} | LongBench v2 | {met} | {fmt(fk_)} | {fmt(es)} | "
                     f"**{fmt(s[met]) if s else '--'}** | {s['n'] if s else ''} |")
        s = cells.get((m, "lgb"))
        for met in ("CR", "CRxAcc"):
            fk_, es = PAPER.get((m, met), (None, None))
            v = s.get(met) if s else None
            L.append(f"| {NAMES.get(m, m)} | LongGenBench | {met.replace('x', ' x ')} | {fmt(fk_)} | {fmt(es)} | "
                     f"**{fmt(v)}** | {s['n'] if s else ''} |")
    L += ["", "Notes: LongBench v2 null predictions (no parsable answer) count as wrong, as in their result.py; "
          "LongGenBench CR x Acc = CR x mean(accuracy once, range, periodic) from the Qwen3-32B judge.", ""]
    for (m, b), s in sorted(cells.items()):
        L.append(f"- {m} {b}: " + ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in s.items()))
    open(os.path.join(a.out, f"table_kr{a.krank}_vr{a.vrank}.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
