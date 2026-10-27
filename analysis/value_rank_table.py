"""Value-rank sweep table: avg@k / pass@k per value rank, with paired CIs.

Keys are fixed (rank 8 for scoring and reconstruction, eta per dataset), and
values are plain PCA at each rank (cells tagged _vr<r>_vpca). Two references:

  vs lossless  value rank 32 from the SAME sweep. At page size 32 a centred page
               has rank <= 31, so rank 32 reconstructs values exactly (bf16
               rounding aside): same machine, software stack and keys. This is
               the comparison that isolates value compression.
  vs keys-only the decoupled-rank ablation's recon-rank-8 cells (_cr8): same
               keys, exact values, but run on a different stack (torch
               2.5.1/cu121 vs 2.8/cu128 here), so it also absorbs stack noise.

CIs: paired bootstrap over problems, pairing on (problem, seed), the same
construction as analysis/paired.py.

Only complete seeds are used; k and n are printed per cell.

    python analysis/value_rank_table.py [--model ds-r1-qwen-7b]
"""
import argparse, glob, json, re
import numpy as np
from paths import ROOT

PAT = r"\\boxed{((?:[^{}]|\{[^{}]*\})*)}"
NPROB = {"MATH50": 50, "AIME24": 30, "GPQA50c": 50}

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="ds-r1-qwen-7b")
ap.add_argument("--root", default=str(ROOT / "results/codec"))
ap.add_argument("--etamap", nargs="+", default=["MATH50:0.25", "AIME24:1.0", "GPQA50c:0.25"],
                help="key eta per dataset")
ap.add_argument("--ranks", type=int, nargs="+", default=[32, 16, 8, 4, 1])
ap.add_argument("--boot", type=int, default=10000)
A = ap.parse_args()
KETA = dict(kv.split(":") for kv in A.etamap)


def cell(pattern, nexp):
    """{(seed, qid): correct} over complete seed files matching the glob."""
    M = {}
    for f in sorted(glob.glob(pattern)):
        recs = [json.loads(l) for l in open(f) if l.strip()]
        if len(recs) != nexp:
            continue
        s = re.search(r"-seed(\d+)", f)
        s = s.group(1) if s else "42"
        for e in recs:
            m = re.search(PAT, e["pred"])
            M[(s, e["qid"])] = int(bool(m and m.group(1).replace(" ", "")
                                        == str(e["answer"]).replace(" ", "")))
    return M


def summary(M):
    seeds = sorted({s for s, _ in M}); probs = sorted({q for _, q in M})
    X = np.array([[M[(s, q)] for q in probs] for s in seeds])
    return len(seeds), len(probs), X.mean() * 100, (X.max(0) > 0).mean() * 100


def paired(a, b, rng):
    keys = sorted(set(a) & set(b))
    probs = sorted({q for _, q in keys}); seeds = sorted({s for s, _ in keys})
    D = np.array([[a[(s, q)] - b[(s, q)] for s in seeds] for q in probs]) * 100
    boot = [D[rng.integers(0, len(probs), len(probs))].mean() for _ in range(A.boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return D.mean(), lo, hi


rng = np.random.default_rng(0)
base = f"{A.root}/{A.model}-spec_ret"
print(f"{A.model}: keys rank 8 (scoring + reconstruction), values plain PCA; k = complete seeds")
print("diffs in avg@k points, paired 95% CI; '*' = CI excludes 0\n")
for ds, nexp in NPROB.items():
    ke = KETA[ds]
    ref = cell(f"{base}/{ds}-*oval_r8_eta{ke}_q0_recon1_cr8-pQ2*.jsonl", nexp)
    print(f"-- {ds}  (key eta {ke})")
    print(f"{'values':<22}{'k':>3}{'n':>5}{'avg@k':>9}{'pass@k':>9}   {'vs lossless (r32)':<26}vs keys-only (cr8)")
    if ref:
        k, n, avg, pas = summary(ref)
        print(f"{'exact (keys-only cr8)':<22}{k:>3}{n:>5}{avg:>9.2f}{pas:>9.2f}")
    else:
        print(f"{'exact (keys-only cr8)':<22}   -- not on disk")
    loss = cell(f"{base}/{ds}-*oval_r8_eta{ke}_q0_recon1_vr32_vpca-pQ2*.jsonl", nexp)
    for r in A.ranks:
        M = cell(f"{base}/{ds}-*oval_r8_eta{ke}_q0_recon1_vr{r}_vpca-pQ2*.jsonl", nexp)
        if not M:
            print(f"{'PCA rank ' + str(r):<22}   -- not run")
            continue
        k, n, avg, pas = summary(M)
        def fmt(R):
            if not R:
                return "--"
            m, lo, hi = paired(M, R, rng)
            return f"{m:+6.2f} [{lo:+.2f}, {hi:+.2f}]" + (" *" if lo > 0 or hi < 0 else "")
        d1 = "(reference)" if r == 32 else fmt(loss)
        print(f"{'PCA rank ' + str(r):<22}{k:>3}{n:>5}{avg:>9.2f}{pas:>9.2f}   {d1:<26}{fmt(ref)}")
    print()
