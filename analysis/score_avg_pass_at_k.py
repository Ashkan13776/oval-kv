"""avg@k and pass@k for the long-reasoning codec grid.

avg@k  = accuracy averaged over the k sampled generations.
pass@k = fraction of problems solved by AT LEAST ONE of the k generations.
Each seed contributes one generation per problem, so k = number of seeds.

A cell's k and n are always printed. A run still in flight has fewer records,
and because problems are intersected across seeds a partial seed shrinks n --
comparing such a cell against a complete one is meaningless, so they are
reported separately rather than mixed in.
"""
import argparse, glob, json, re
import numpy as np
from paths import ROOT

PAT = r"\\boxed{((?:[^{}]|\{[^{}]*\})*)}"

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(ROOT / "results/codec"))
ap.add_argument("--models", nargs="+",
                default=["ds-r1-qwen-7b", "ds-r1-llama-8b", "ds-r1-qwen-14b"])
ap.add_argument("--datasets", nargs="+", default=["MATH50", "AIME24", "GPQA50c"])
ap.add_argument("--etas", nargs="+", default=["0.0", "0.25", "0.5", "0.75", "1.0"])
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--n", type=int, nargs="+", default=None,
                help="expected problems per dataset, to flag partial cells")
A = ap.parse_args()
NEXP = dict(zip(A.datasets, A.n)) if A.n else {"MATH50": 50, "AIME24": 30, "GPQA50c": 50}

def outcomes(path):
    v = {}
    for line in open(path):
        if not line.strip(): continue
        e = json.loads(line)
        m = re.search(PAT, e["pred"])
        v[e["qid"]] = int(bool(m and m.group(1).replace(" ", "")
                               == str(e["answer"]).replace(" ", "")))
    return v

def cells(pattern, nexp):
    full, partial = [], []
    for f in sorted(glob.glob(pattern)):
        v = outcomes(f)
        s = re.search(r"-seed(\d+)", f)
        (full if len(v) == nexp else partial).append((s.group(1) if s else "42", v, len(v)))
    return full, partial

def report(label, pattern, nexp):
    full, part = cells(pattern, nexp)
    if not full:
        return f"{label:<28}{0:>3}{'--':>6}{'--':>9}{'--':>9}" + (
            f"   {len(part)} partial" if part else "   not run")
    q = set(full[0][1])
    for _, v, _ in full[1:]: q &= set(v)
    order = sorted(q)
    M = np.array([[v[x] for x in order] for _, v, _ in full])
    per = " ".join(f"{M[i].mean()*100:.0f}" for i in range(len(full)))
    extra = f"  (+{len(part)} running)" if part else ""
    return (f"{label:<28}{len(full):>3}{len(q):>6}{M.mean()*100:>8.2f}%"
            f"{(M.max(axis=0) > 0).mean()*100:>8.2f}%   {per}{extra}")

for model in A.models:
    header_done = False
    for ds in A.datasets:
        nexp = NEXP.get(ds, 50)
        lines = [report("dense (FullKV)", f"{A.root}/{model}-full/{ds}-*.jsonl", nexp)]
        for e in A.etas:
            lines.append(report(
                f"codec r={A.rank} eta={e}",
                f"{A.root}/{model}-spec_ret/{ds}-*oval_r{A.rank}_eta{e}_q0_recon1*.jsonl", nexp))
            sel = f"{A.root}/{model}-spec_ret/{ds}-*oval_r{A.rank}_eta{e}_q0-pQ2*.jsonl"
            if glob.glob(sel):
                lines.append(report(f"retrieval r={A.rank} eta={e}", sel, nexp))
        if all("not run" in l for l in lines[1:]):
            continue
        if not header_done:
            print(f"\n{'='*86}\n{model}\n{'='*86}")
            header_done = True
        print(f"\n-- {ds}")
        print(f"{'arm':<28}{'k':>3}{'n':>6}{'avg@k':>9}{'pass@k':>9}   per-seed")
        for l in lines: print(l)
