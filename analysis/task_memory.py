"""Per-task projected persistent KV storage for the OVAL key codec.

Computed from the ACTUAL sequence lengths of the evaluation runs (input_len +
output_len of every generation on disk), not from a fixed context example.

Procedure, per generation:
  seq        = input_len + output_len            (storage evaluated at end of
                                                  generation = peak persistent KV)
  npages     = max(0, (seq - sink - recent) // page)
  summary    = npages * record_bytes * L * Hkv   (the rank-r records = K-hat)
  retainedK  = (seq - npages*page) * d * B * L * Hkv
               ^ sink + recent + ANY incomplete trailing page; when
                 seq < sink+recent+page this is the whole sequence and the
                 codec never activates (summary = 0).
  denseK     = seq * d * B * L * Hkv ;  denseV = denseK
  V_total    = denseV   (values are never compressed; retained values are
                         already inside denseV -- counted once, not added again)
  compK      = summary + retainedK
  compKV     = compK + V_total

Task figures are the MEAN over every generation of every included eta arm.
Memory is independent of eta at a fixed sequence length and storage layout,
but generated lengths differ by eta, so the average is taken over generations.

record_bytes (bf16) = (d*r + page*r + d) * 2   # basis + coefficients + centroid
"""
import argparse, glob, json, re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/home/shahbaa/oval-kv/results/codec")
ap.add_argument("--model", default="ds-r1-llama-8b")
ap.add_argument("--layers", type=int, default=32)
ap.add_argument("--kv_heads", type=int, default=8)
ap.add_argument("--head_dim", type=int, default=128)
ap.add_argument("--page", type=int, default=32)
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--bytes_per_val", type=int, default=2)
ap.add_argument("--sink", type=int, default=512)
ap.add_argument("--recent", type=int, default=512)
ap.add_argument("--tasks", nargs="+", default=["MATH50", "AIME24", "GPQA50c"])
ap.add_argument("--nexp", type=int, nargs="+", default=[50, 30, 50])
ap.add_argument("--etas", nargs="+", default=["0.0", "0.25", "0.5", "0.75", "1.0"])
ap.add_argument("--require_complete", action="store_true", default=True,
                help="only include eta arms with all 8 seeds at full n")
A = ap.parse_args()

d, B, pg, r = A.head_dim, A.bytes_per_val, A.page, A.rank
scale = A.layers * A.kv_heads
REC = (d * r + pg * r + d) * B
MiB = 1 / 2**20
NEXP = dict(zip(A.tasks, A.nexp))
RES = f"{A.root}/{A.model}-spec_ret"

def gens(task, eta):
    """per-generation seq lengths for one eta arm, or None if arm incomplete"""
    files = sorted(glob.glob(f"{RES}/{task}-*oval_r{r}_eta{eta}_q0_recon1*.jsonl"))
    out, nfull = [], 0
    for f in files:
        rows = [json.loads(l) for l in open(f) if l.strip()]
        if len(rows) != NEXP[task]:
            continue
        nfull += 1
        out += [x["input_len"] + x["output_len"] for x in rows]
    return (out, nfull) if nfull == 8 else (None, nfull)

print(f"{A.model} | rank {r}, page {pg}, bf16 | {A.layers} layers x {A.kv_heads} kv heads, d={d}")
print(f"record = (d*r + page*r + d)*{B} = {REC} B per page per (layer x kv head); "
      f"sink {A.sink} + recent {A.recent} retained exact\n")

hdr = (f"{'Task':<9}{'probs':>6}{'gens':>6}{'meanSeq':>9}{'DenseK':>9}{'DenseV':>9}"
       f"{'DenseKV':>9}{'KeySumm':>9}{'RetainK':>9}{'V_total':>9}{'Summ+V':>9}"
       f"{'CompKV':>9}{'cK/dK':>8}{'cKV/dKV':>9}")
print(hdr); print("-" * len(hdr))

for task in A.tasks:
    arms, lens, incomplete = [], [], []
    for eta in A.etas:
        L, nfull = gens(task, eta)
        if L is None:
            incomplete.append(f"{eta}({nfull}/8)")
        else:
            arms.append(eta); lens += L
    if not arms:
        print(f"{task:<9}{'pending':>6}  -- no complete eta arm "
              f"({', '.join(incomplete) if incomplete else 'not run'})")
        continue

    n = len(lens)
    dK = dV = summ = retK = 0.0
    for seq in lens:
        npages = max(0, (seq - A.sink - A.recent) // pg)
        summ += npages * REC * scale
        retK += (seq - npages * pg) * d * B * scale
        dK   += seq * d * B * scale
    dV = dK
    dK /= n; dV /= n; summ /= n; retK /= n
    Vtot = dV                      # values unchanged; retained V already inside
    compK = summ + retK
    compKV = compK + Vtot
    dKV = dK + dV
    print(f"{task:<9}{NEXP[task]:>6}{n:>6}{sum(lens)/n:>9.1f}"
          f"{dK*MiB:>9.1f}{dV*MiB:>9.1f}{dKV*MiB:>9.1f}{summ*MiB:>9.1f}"
          f"{retK*MiB:>9.1f}{Vtot*MiB:>9.1f}{(summ+Vtot)*MiB:>9.1f}"
          f"{compKV*MiB:>9.1f}{compK/dK:>8.3f}{compKV/dKV:>9.3f}")
    never = sum(1 for s in lens if s < A.sink + A.recent + pg)
    print(f"{'':<9}etas included: {', '.join(arms)}"
          + (f" | pending: {', '.join(incomplete)}" if incomplete else "")
          + f" | {never}/{n} gens never activate compression")
