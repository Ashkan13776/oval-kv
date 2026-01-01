"""Pareto front on MATH50: accuracy vs memory, across 18 methods x 4 budgets.

Accuracy is avg@k over seeds with a bootstrap CI, so a point that looks better
is only called better when the CI separates it.

MEMORY is the contentious axis and is NOT uniform across methods, because the
methods are not all doing the same thing:

  RETRIEVAL methods keep the whole KV cache resident and only choose what to
  ATTEND to each step. Their memory is full K + full V + whatever digest they
  store per page. Budget buys them speed, not memory.

  EVICTION methods permanently DISCARD KV. Their memory is sink+recent+budget,
  which is far smaller -- but the discarded tokens are unrecoverable.

Scoring them on one memory axis is the whole point of the front, but it is only
honest if the eviction arms actually evict. Arms that are implemented as
no-evict (they rank pages by an eviction method's criterion but keep the cache)
are marked UPPER_BOUND: their accuracy is what the method would get if eviction
cost nothing, i.e. optimistic. They are plotted with a distinct marker.
"""
import json, glob, re, os, sys
import numpy as np

D = "/home/shahbaa/kv-compression/results/freekv/codec"
MODEL, DS = "ds-r1-qwen-7b", "MATH50"
PAT = r"\\boxed{((?:[^{}]|\{[^{}]*\})*)}"
d_head, m_page = 128, 32
B_K = d_head * m_page * 2        # bytes of keys per page per kv head, fp16
B_V = B_K

# per-page digest bytes, and whether the method still needs the exact keys
#   tag: (digest_bytes_per_page, keeps_exact_keys, evicts, pattern)
SPEC = {
 "full":     (0,    True,  False, f"{D}/{MODEL}-full/{DS}-*.jsonl"),
 "quest":    (512,  True,  False, f"{D}/{MODEL}-quest/{DS}-*.jsonl"),
 "arkvale":  (512,  True,  False, f"{D}/{MODEL}-spec_ret/{DS}-*arkv*.jsonl"),
 "freekv":   (512,  True,  False, f"{D}/{MODEL}-spec_ret/{DS}-*avgSM-pQ2-*.jsonl"),
 "raas":     (0,    False, True,  f"{D}/{MODEL}-raas/{DS}-*.jsonl"),
 "ours":     (2816, False, False, f"{D}/{MODEL}-spec_ret/{DS}-*oval_r8_eta0.0_q0_recon1*.jsonl"),
 "ours_sel": (2816, True,  False, f"{D}/{MODEL}-spec_ret/{DS}-*oval_r8_eta0.0_q0-pQ2*.jsonl"),
 "loki":     (512,  True,  False, f"{D}/{MODEL}-spec_ret/{DS}-*loki*.jsonl"),
 "infllm":   (1024, True,  False, f"{D}/{MODEL}-spec_ret/{DS}-*infllm*.jsonl"),
 "sparq":    (0,    True,  False, f"{D}/{MODEL}-spec_ret/{DS}-*sparq*.jsonl"),
 "centroid": (256,  True,  False, f"{D}/{MODEL}-spec_ret/{DS}-*centroid*.jsonl"),
 "knorm":    (2,    True,  False, f"{D}/{MODEL}-spec_ret/{DS}-*knorm*.jsonl"),
 "pyramid":  (512,  True,  False, f"{D}/{MODEL}-spec_ret/{DS}-*avgSM-pyr*.jsonl"),
 "random":   (0,    True,  False, f"{D}/{MODEL}-spec_ret/{DS}-*random*.jsonl"),
 "oracle":   (0,    True,  False, f"{D}/{MODEL}-spec_ret/{DS}-*oracle*.jsonl"),
 "h2o":      (4,    True,  False, f"{D}/{MODEL}-spec_ret/{DS}-*h2o*.jsonl"),
 "tova":     (4,    True,  False, f"{D}/{MODEL}-spec_ret/{DS}-*tova*.jsonl"),
 "snapkv":   (0,    True,  False, f"{D}/{MODEL}-spec_ret/{DS}-*snapkv*.jsonl"),
 "stream":   (0,    False, True,  f"{D}/{MODEL}-spec_ret/{DS}-*-b0-*.jsonl"),
}
# arms whose paper evicts but whose implementation here does not
UPPER_BOUND = {"h2o", "tova", "snapkv"}

def correct(path):
    out = {}
    for l in open(path):
        if not l.strip(): continue
        e = json.loads(l); mm = re.search(PAT, e["pred"])
        out[e["qid"]] = int(bool(mm and mm.group(1).replace(" ", "")
                                 == str(e["answer"]).replace(" ", "")))
    return out

def mem_bytes(tag, budget, nseq_pages):
    """Bytes per kv head per layer for a sequence of nseq_pages paged pages."""
    dig, exact, evicts, _ = SPEC[tag]
    if evicts:                       # only sink+recent+budget survive
        kept = (1024 + budget) / m_page
        return kept * (B_K + B_V)
    keys = B_K if exact else 0       # ours replaces keys with the record
    return nseq_pages * (keys + B_V + dig)

def pareto(points):
    """points: list of (mem, acc, label). Lower mem better, higher acc better."""
    front = []
    for p in sorted(points):
        if not front or p[1] > front[-1][1]:
            front.append(p)
    return front

rows = []
for tag, (dig, exact, evicts, pat) in SPEC.items():
    byb = {}
    for f in sorted(glob.glob(pat)):
        b = re.search(r"-b(\d+)-", f); s = re.search(r"-seed(\d+)", f)
        if not b: continue
        byb.setdefault(int(b.group(1)), []).append(f)
    for budget, fs in sorted(byb.items()):
        per = [correct(f) for f in fs]
        per = [p for p in per if p]
        if not per: continue
        qids = set(per[0])
        for p in per[1:]: qids &= set(p)
        if not qids: continue
        M = np.array([[p[q] for q in sorted(qids)] for p in per])  # [k, nprob]
        acc = M.mean() * 100
        # bootstrap over problems, the unit that varies
        bs = np.array([M[:, np.random.randint(0, M.shape[1], M.shape[1])].mean()
                       for _ in range(2000)]) * 100
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append(dict(tag=tag, budget=budget, k=M.shape[0], n=M.shape[1],
                         acc=acc, lo=lo, hi=hi,
                         mem=mem_bytes(tag, budget, 110),   # 3520 tok ~ 110 pages
                         ub=tag in UPPER_BOUND))

if not rows:
    print("no results on disk yet"); sys.exit()
rows.sort(key=lambda r: (-r["acc"],))
print(f"{'method':<10}{'B':>6}{'k':>3}{'n':>4}{'avg@k':>9}{'95% CI':>17}{'KiB/head/layer':>16}")
print("-" * 68)
for r in rows:
    flag = " *UB" if r["ub"] else ""
    print(f"{r['tag']:<10}{r['budget']:>6}{r['k']:>3}{r['n']:>4}{r['acc']:>8.2f}%"
          f"{('[%.1f, %.1f]' % (r['lo'], r['hi'])):>17}{r['mem']/1024:>15.1f}{flag}")
pts = [(r["mem"] / 1024, r["acc"], f"{r['tag']}@B{r['budget']}") for r in rows
       if not r["ub"]]
print("\n=== Pareto front (deployable arms only; UB arms excluded) ===")
for m, a, lab in pareto(pts):
    print(f"  {lab:<22}{m:>9.1f} KiB{a:>9.2f}%")
