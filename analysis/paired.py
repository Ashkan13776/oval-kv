"""Paired comparison at matched budget.

The unpaired CIs in pareto.py are dominated by how much problems differ from
each other in difficulty, which is irrelevant: every arm sees the SAME 50
problems under the SAME 8 seeds. Pairing removes that variance, so this is the
test that can actually resolve a 1-point gap.

Reported per problem: mean accuracy over seeds. Then a paired bootstrap over
problems on the difference. A gap is only real if the CI excludes 0.
"""
import json, glob, re, sys
import numpy as np
D = "/home/shahbaa/kv-compression/results/freekv/codec"
M, DS = "ds-r1-qwen-7b", "MATH50"
PAT = r"\\boxed{((?:[^{}]|\{[^{}]*\})*)}"
SPEC = {
 "ours":    f"{D}/{M}-spec_ret/{DS}-*oval_r8_eta0.0_q0_recon1*.jsonl",
 "freekv":  f"{D}/{M}-spec_ret/{DS}-*avgSM-pQ2-*.jsonl",
 "arkvale": f"{D}/{M}-spec_ret/{DS}-*arkv*.jsonl",
 "quest":   f"{D}/{M}-quest/{DS}-*.jsonl",
 "raas":    f"{D}/{M}-raas/{DS}-*.jsonl",
}
def load(pat, budget):
    out = {}
    for f in sorted(glob.glob(pat)):
        b = re.search(r"-b(\d+)-", f)
        if not b or int(b.group(1)) != budget: continue
        s = re.search(r"-seed(\d+)", f); s = int(s.group(1)) if s else 42
        for l in open(f):
            if not l.strip(): continue
            e = json.loads(l); mm = re.search(PAT, e["pred"])
            ok = int(bool(mm and mm.group(1).replace(" ","") == str(e["answer"]).replace(" ","")))
            out.setdefault(s, {})[e["qid"]] = ok
    return out
B = int(sys.argv[1]) if len(sys.argv) > 1 else 128
data = {k: load(v, B) for k, v in SPEC.items()}
data = {k: v for k, v in data.items() if v}
seeds = set.intersection(*[set(v) for v in data.values()])
qids  = set.intersection(*[set(d[s]) for v in data.values() for s, d in [(min(v), v)]])
for k, v in data.items():
    for s in seeds: qids &= set(v[s])
qids = sorted(qids); seeds = sorted(seeds)
print(f"budget {B}: {len(seeds)} seeds x {len(qids)} problems, arms {list(data)}")
# per-problem mean over seeds -> [arm][problem]
per = {k: np.array([np.mean([data[k][s][q] for s in seeds]) for q in qids]) for k in data}
for k in sorted(per, key=lambda k: -per[k].mean()):
    print(f"  {k:<9}{per[k].mean()*100:7.2f}%")
ref = "freekv" if "freekv" in per else sorted(per)[0]
print(f"\npaired vs {ref} (10k bootstrap over problems; CI excluding 0 = real):")
rng = np.random.default_rng(0)
idx = rng.integers(0, len(qids), (10000, len(qids)))
for k in per:
    if k == ref: continue
    d = per[k] - per[ref]
    bs = d[idx].mean(1) * 100
    lo, hi = np.percentile(bs, [2.5, 97.5])
    sig = "  <-- real" if (lo > 0) or (hi < 0) else ""
    print(f"  {k:<9}{d.mean()*100:+7.2f}  [{lo:+6.2f}, {hi:+6.2f}]{sig}")
