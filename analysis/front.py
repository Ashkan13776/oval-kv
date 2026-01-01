"""The front at the paper config: B=2048, sink 512, recent 512, ds-r1-qwen-7b.

Memory is per KV head per layer for a 3869-token mean sequence (110 paged
pages), fp16. Every arm keeps the full V cache and attends the same number of
tokens; they differ in the per-page DIGEST they must store to choose, and OVAL
additionally replaces the keys themselves with its record.
"""
import json, glob, re, sys
import numpy as np
D = "/home/shahbaa/kv-compression/results/freekv/codec/ds-r1-qwen-7b"
PAT = r"\\boxed{((?:[^{}]|\{[^{}]*\})*)}"
d, m, NP = 128, 32, 110
B_K = d * m * 2            # exact keys per page
B_V = B_K
#            digest bytes/page, keeps exact keys, glob
A = {
 "full":        (0,    True,  f"{D}-full/MATH50-*"),
 "quest":       (512,  True,  f"{D}-quest/MATH50-*b2048-avgSM-skl1*"),
 "freekv":      (512,  True,  f"{D}-spec_ret/MATH50-*b2048-avgSM-pQ2-*"),
 "arkvale":     (512,  True,  f"{D}-spec_ret/MATH50-*b2048-avgSM-arkv-*"),
 "raas":        (0,    False, f"{D}-raas/MATH50-*b2048*"),
 "ours":        (2816, False, f"{D}-spec_ret/MATH50-*b2048-avgSM-oval_r8_eta0.0_q0_recon1-*"),
 "ours_sel":    (2816, True,  f"{D}-spec_ret/MATH50-*b2048-avgSM-oval_r8_eta0.0_q0-pQ2*"),
 "ours_q":      (2816, True,  f"{D}-quest/MATH50-*b2048-avgSM-oval_r8_eta0.0_q0-*"),
 "loki":        (512,  True,  f"{D}-quest/MATH50-*b2048-avgSM-loki*"),
 "sparq":       (0,    True,  f"{D}-quest/MATH50-*b2048-avgSM-sparq*"),
 "infllm":      (1024, True,  f"{D}-quest/MATH50-*b2048-avgSM-infllm*"),
 "centroid":    (256,  True,  f"{D}-quest/MATH50-*b2048-avgSM-centroid*"),
 "shadowkv":    (256,  True,  f"{D}-quest/MATH50-*b2048-avgSM-shadowkv*"),
 "rocketkv":    (256,  True,  f"{D}-quest/MATH50-*b2048-avgSM-rocketkv*"),
 "q4key":       (2560, False, f"{D}-quest/MATH50-*b2048-avgSM-q4key*"),
 "knorm":       (2,    True,  f"{D}-quest/MATH50-*b2048-avgSM-knorm*"),
 "expatt":      (0,    True,  f"{D}-quest/MATH50-*b2048-avgSM-expatt*"),
 "h2o":         (4,    True,  f"{D}-quest/MATH50-*b2048-avgSM-h2o*"),
 "tova":        (4,    True,  f"{D}-quest/MATH50-*b2048-avgSM-tova*"),
 "snapkv":      (0,    True,  f"{D}-quest/MATH50-*b2048-avgSM-snapkv*"),
 "random":      (0,    True,  f"{D}-quest/MATH50-*b2048-avgSM-random*"),
 "pyramid":     (512,  True,  f"{D}-spec_ret/MATH50-*b2048-avgSM-pyr*"),
 "oracle":      (0,    True,  f"{D}-quest/MATH50-*b2048-avgSM-oracle*"),
 "oracle_prev": (0,    True,  f"{D}-spec_ret/MATH50-*b2048-avgSM-oracle*"),
 "stream":      (0,    False, f"{D}-quest/MATH50-s512-r512-*-b0-avgSM-skl1*"),
 "sliding":     (0,    False, f"{D}-quest/MATH50-s512-r2560-*-b0-avgSM-skl1*"),
}
CEIL = {"oracle", "oracle_prev"}
def per_problem(pat):
    byseed = {}
    for f in sorted(glob.glob(pat)):
        s = re.search(r"-seed(\d+)", f); s = int(s.group(1)) if s else 42
        for l in open(f):
            if not l.strip(): continue
            e = json.loads(l); mm = re.search(PAT, e["pred"])
            byseed.setdefault(s, {})[e["qid"]] = int(bool(
                mm and mm.group(1).replace(" ","") == str(e["answer"]).replace(" ","")))
    return byseed
def mem(tag):
    dig, exact, _ = A[tag]
    if tag in ("raas", "stream", "sliding"):
        return (1024 + (0 if tag != "raas" else 2048)) / m * (B_K + B_V)
    if tag == "full":
        return NP * (B_K + B_V)
    return NP * ((B_K if exact else 0) + B_V + dig)
rows = []
for tag, (dig, exact, pat) in A.items():
    bs = per_problem(pat)
    if not bs: continue
    seeds = sorted(bs)
    q = set(bs[seeds[0]])
    for s in seeds: q &= set(bs[s])
    if not q: continue
    q = sorted(q)
    M = np.array([[bs[s][x] for x in q] for s in seeds])
    rows.append((tag, len(seeds), len(q), M.mean()*100, mem(tag)/1024, M))
rows.sort(key=lambda r: -r[3])
print(f"{'arm':<12}{'k':>3}{'n':>5}{'avg@k':>9}{'KiB/hd/lyr':>12}")
print("-"*42)
for t,k,n,a,mm_,_ in rows:
    mark = "  (ceiling)" if t in CEIL else ""
    print(f"{t:<12}{k:>3}{n:>5}{a:>8.2f}%{mm_:>11.0f}{mark}")
# paired vs freekv on the shared problem set
ref = [r for r in rows if r[0]=="freekv"]
if ref:
    print("\npaired vs freekv (bootstrap over problems; CI excluding 0 = real):")
    rf = ref[0]
    rng = np.random.default_rng(0)
    for t,k,n,a,mm_,M in rows:
        if t=="freekv": continue
        q = min(M.shape[1], rf[5].shape[1]); ks = min(M.shape[0], rf[5].shape[0])
        dd = M[:ks,:q].mean(0) - rf[5][:ks,:q].mean(0)
        idx = rng.integers(0,q,(4000,q)); bsv = dd[idx].mean(1)*100
        lo,hi = np.percentile(bsv,[2.5,97.5])
        sig = " *" if lo>0 or hi<0 else ""
        print(f"  {t:<12}{dd.mean()*100:+7.2f}  [{lo:+6.2f},{hi:+6.2f}]{sig}")
