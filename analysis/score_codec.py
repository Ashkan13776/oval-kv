"""Score the codec rank grid, with the harness's own grader logic.

n is 30-50 per cell, so a single rank cell is NOT resolvable: one problem is
2-3 points, and pairwise McNemar over these arms returns p >= 0.5 throughout.
What IS resolvable is the TREND across ranks, so that is what this reports:
Pearson r of accuracy against log2(rank), with a permutation p-value that
shuffles each PROBLEM's outcomes across the rank arms, preserving per-problem
difficulty. Report the trend, not the individual cells.
"""
import argparse, glob, json, random, re
import numpy as np

from paths import ROOT

MAXLEN = 16384
PAT = r"\\boxed{((?:[^{}]|\{[^{}]*\})*)}"
EFF = {1: 1, 4: 4, 8: 8, 16: 16, 32: 31}     # rank clamped to page_size-1

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(ROOT / "results/codec"))
ap.add_argument("--models", nargs="+",
                default=["ds-r1-qwen-7b", "ds-r1-llama-8b", "ds-r1-qwen-14b"])
ap.add_argument("--datasets", nargs="+", default=["MATH50", "AIME24", "GPQA50c"])
ap.add_argument("--ranks", type=int, nargs="+", default=[1, 4, 8, 16, 32])
ap.add_argument("--strict", action="store_true",
                help="require a '**Final Answer**' marker before the boxed answer")
ap.add_argument("--perms", type=int, default=20000)
A = ap.parse_args()

def outcomes(path):
    v = []
    for line in open(path):
        if not line.strip(): continue
        e = json.loads(line)
        p = e["pred"]
        si = p.find("**Final Answer**") if A.strict else 0
        ok = False
        if si != -1:
            m = re.search(PAT, p[si:])
            ok = bool(m and m.group(1).replace(" ", "") == str(e["answer"]).replace(" ", ""))
        v.append(int(ok))
    return np.array(v)

def find(model, ds, tag):
    if tag == "full":
        g = glob.glob(f"{A.root}/{model}-full/{ds}-*.jsonl")
    elif tag.startswith("sel_r"):
        g = glob.glob(f"{A.root}/{model}-spec_ret/{ds}-*oval_r{tag[5:]}_eta0.0_q0-pQ2*.jsonl")
    else:
        g = glob.glob(f"{A.root}/{model}-spec_ret/{ds}-*oval_r{tag[7:]}_eta0.0_q0_recon1*.jsonl")
    return sorted(g)[0] if g else None

def trend(V, ranks, seed=0):
    x = np.log2(np.array([EFF[r] for r in ranks], dtype=float))
    M = np.stack(V); y = M.mean(axis=1) * 100
    if np.std(y) == 0: return 0.0, 1.0
    obs = np.corrcoef(x, y)[0, 1]
    rng = random.Random(seed); hits = 0
    for _ in range(A.perms):
        P = M.copy()
        for j in range(P.shape[1]):
            c = list(P[:, j]); rng.shuffle(c); P[:, j] = c
        yy = P.mean(axis=1) * 100
        rr = 0.0 if np.std(yy) == 0 else np.corrcoef(x, yy)[0, 1]
        if rr >= obs: hits += 1
    return obs, hits / A.perms

found = False
for m in A.models:
    blocks = []
    for ds in A.datasets:
        cells = {}
        for tag in ["full", "sel_r8"] + [f"codec_r{r}" for r in A.ranks]:
            f = find(m, ds, tag)
            if f: cells[tag] = outcomes(f)
        if cells: blocks.append((ds, cells))
    if not blocks: continue
    found = True
    print(f"\n{'='*80}\n{m}   ({'strict' if A.strict else 'loose'} grading)\n{'='*80}")
    print(f"{'dataset':<9}{'n':>4}{'full':>7}{'sel r8':>8}"
          + "".join(f"{('r=%d' % r):>7}" for r in A.ranks) + f"{'trend r':>9}{'p':>7}")
    for ds, cells in blocks:
        n = len(next(iter(cells.values())))
        pc = lambda t: f"{cells[t].mean()*100:.0f}%" if t in cells else "--"
        if all(f"codec_r{r}" in cells for r in A.ranks):
            r_, p_ = trend([cells[f"codec_r{r}"] for r in A.ranks], A.ranks)
            tr = f"{r_:>9.3f}{p_:>7.3f}"
        else:
            tr = f"{'--':>9}{'--':>7}"
        print(f"{ds:<9}{n:>4}{pc('full'):>7}{pc('sel_r8'):>8}"
              + "".join(f"{pc('codec_r%d' % r):>7}" for r in A.ranks) + tr)
if not found:
    print(f"no cells found under {A.root}")
