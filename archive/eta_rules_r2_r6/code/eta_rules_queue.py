"""Queue builder for the eta-rule harness cells.

One harness cell per DISTINCT deployed eta table per (model, dataset), k seeds
each. Tables are identified by the content hash the harness puts in the output
filename (`_etab<hash>`, see oval/page_basis.py::eta_table_hash), so two rules
that select the same table share one cell: the first rule in --rules order
owns it and the others are listed as aliases.

A seed is queued only if its .jsonl holds fewer records than the dataset has
problems (the same per-seed completeness rule as scripts/codec_reasoning_sweep.sh),
so the builder is safe to re-run at any time. Order: datasets in the given
order (MATH50 first: cheapest), then models, then rules, then seeds.

--rules may name recorded variants (r3u, r4w, r6w, r6x): a table not found in
--tables_dir is looked up in --variants_dir (default: the sibling
tables_variants/). List the deployed rules first so a variant whose table equals
a deployed one is an alias of that (already run) cell. Nothing is queued that is
not named in --rules.

  python analysis/eta_rules_queue.py            # dry run: print the plan
  python analysis/eta_rules_queue.py --write logs/eta_rules/jobs.txt
  python analysis/eta_rules_queue.py --count    # number of incomplete cells only
"""
import argparse, glob, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT

NPROB = {"MATH50": 50, "AIME24": 30, "GPQA50c": 50}

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["ds-r1-qwen-7b", "ds-r1-llama-8b"])
ap.add_argument("--datasets", nargs="+", default=["MATH50", "AIME24", "GPQA50c"])
ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 50)))
ap.add_argument("--rules", nargs="+", default=["r1", "r2", "r3", "r4", "r5", "r6"])
ap.add_argument("--tables_dir", default=str(ROOT / "results/eta_rules/tables"))
ap.add_argument("--variants_dir", default=None,
                help="where variant tables live (default: tables_variants/ next to --tables_dir)")
ap.add_argument("--root", default=str(ROOT / "results/codec"))
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--vrank", type=int, default=8)
ap.add_argument("--write", default=None, help="write the job lines here (model|dataset|seed|hash|table)")
ap.add_argument("--count", action="store_true", help="print only the number of incomplete cells")
ap.add_argument("--complete", action="store_true", help="print only the number of complete cells")
ap.add_argument("--require_all", action="store_true", help="exit 2 if any (model, dataset, rule) table is missing")
ap.add_argument("--priority_file", default=str(ROOT / "logs/eta_rules/PRIORITY"),
                help="if it exists: rule names (whitespace-separated) whose seed-cells go to the "
                     "FRONT of the queue, in the order listed; the rest keep their order")
A = ap.parse_args()
if A.variants_dir is None:
    A.variants_dir = os.path.join(os.path.dirname(os.path.abspath(A.tables_dir.rstrip("/"))), "tables_variants")


def table_file(model, ds, rule):
    """Deployed table first, then the recorded variant; None if neither exists."""
    for d in (A.tables_dir, A.variants_dir):
        tf = os.path.join(d, f"{model}_{ds}_{rule}.json")
        if os.path.exists(tf):
            return tf
    return None


def records(path):
    n = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def seed_file(model, ds, h, seed):
    pat = f"{A.root}/{model}-spec_ret/{ds}-*oval_r{A.rank}_etab{h}_q0_recon1_vr{A.vrank}_vpca-pQ2*.jsonl"
    fs = sorted(glob.glob(pat))
    if seed == 42:
        fs = [f for f in fs if "-seed" not in os.path.basename(f)]
    else:
        fs = [f for f in fs if f.endswith(f"-seed{seed}.jsonl")]
    return fs[0] if fs else None


jobs, n_complete, n_partial, missing = [], 0, 0, []
plan = []
for ds in A.datasets:
    for model in A.models:
        owner = {}                                            # hash -> rule
        for rule in A.rules:
            tf = table_file(model, ds, rule)
            if tf is None:
                missing.append((model, ds, rule)); continue
            tab = json.load(open(tf))
            h = tab["hash"]
            if h in owner:
                plan.append(f"   {model} {ds} {rule}: same table as {owner[h]} (hash {h}) -> shares its cell")
                continue
            owner[h] = rule
            q, c, p = [], 0, 0
            for s in A.seeds:
                f = seed_file(model, ds, h, s)
                n = records(f) if f else 0
                if n >= NPROB[ds]:
                    c += 1
                else:
                    if n > 0: p += 1
                    q.append(s)
                    jobs.append((rule, f"{model}|{ds}|{s}|{h}|{tf}"))
            n_complete += c; n_partial += p
            plan.append(f"   {model} {ds} {rule}: hash {h}  complete {c}/{len(A.seeds)}  partial {p}  "
                        f"queued seeds {q if q else '-'}")

prio = open(A.priority_file).read().split() if os.path.exists(A.priority_file) else []
if prio:
    plan.append(f"   priority (from {A.priority_file}): {' '.join(prio)} first")
jobs = [j for _, j in sorted(jobs, key=lambda rj: prio.index(rj[0]) if rj[0] in prio else len(prio))]

if A.count:
    print(len(jobs)); sys.exit(0)
if A.complete:
    print(n_complete); sys.exit(0)
print("eta-rule cells (one per distinct table; a seed is a cell for the scheduler):")
print("\n".join(plan))
for m in missing:
    print(f"   MISSING table: {m[0]} {m[1]} {m[2]}")
print(f"=== {len(jobs)} seed-cells to run, {n_complete} complete, {n_partial} partial, "
      f"{len(missing)} tables missing")
if A.write:
    os.makedirs(os.path.dirname(A.write), exist_ok=True)
    with open(A.write, "w") as f:
        f.write("\n".join(jobs) + ("\n" if jobs else ""))
    print(f"wrote {A.write}")
if A.require_all and missing:
    sys.exit(2)
