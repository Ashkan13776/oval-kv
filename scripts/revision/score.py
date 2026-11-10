"""Score the revision decode cells: avg@8 / pass@8 with 95% bootstrap CIs, and paired CIs
against the rule-1 OVAL cell of the same (model, task) in this repo: the deployed Tab. 1 eta
table (same hash), rank-16 values, budget 2048, results/reasoning/. NOTE: for Qwen-7B these
repo cells score 70.50 / 48.75 / 35.00 avg@8 (MATH500 / AIME24 / GPQA), not Tab. 1's printed
71.75 / 47.92 / 39.50; the authors confirm Tab. 1 is correct, so differences here are against
the repo run, never against the printed number.

Cells come from the queue job files (scripts/revision/jobs_<block>*.txt): one arm = one job id
without its _s<seed> suffix. The output file of each job is located from its arguments
(model, dataset, budget, seed, eta-table hash, page_rep, recon, vrank) and must match exactly
one file. Grading is the repo's (analysis/score_avg_pass_at_k.py): first \\boxed{...}, spaces
stripped, string-equal to the answer.

  avg@k  = mean over problems of the per-problem mean over the k seeds
  pass@k = fraction of problems solved by at least one seed
  CI     = 95% percentile bootstrap over problems (10,000 resamples, rng seed 0)
  paired = same problems and seeds, bootstrap over problems of (arm - reference)

A cell is reported only when all 8 seeds have all problems; otherwise it is listed as
incomplete with its seed count. Nothing is interpolated.

Writes results/revision/<block>/scores/<arm>.csv (problem x seed correctness, 0/1),
results/revision/tables/<block>.csv and prints a markdown table.

  python scripts/revision/score.py A3 [A1 A2 B]
"""
import csv, glob, json, os, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAT = r"\\boxed{((?:[^{}]|\{[^{}]*\})*)}"
NEXP = {"MATH50": 50, "AIME24": 30, "GPQA50c": 50}
SEEDS = list(range(42, 50))
NB, RNG = 10_000, 0
REF_TABLES = [ROOT / "results/eta_rules/vr16/tables", ROOT / "results/revision/A/calib_vr16/tables"]


def outcomes(path):
    v = {}
    for line in open(path):
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:   # trailing record cut by a crash; the harness drops it on resume
            continue
        m = re.search(PAT, e["pred"])
        v[e["qid"]] = int(bool(m and m.group(1).replace(" ", "") == str(e["answer"]).replace(" ", "")))
    return v


def arg(args, name, default=None):
    m = re.search(rf"--{name} (\S+)", args)
    return m.group(1) if m else default


def find(model, task, budget, seed, args, out_root):
    """The one output file this job writes (or None)."""
    d = Path(out_root) / f"{model}-{arg(args, 'method')}"
    cands = glob.glob(str(d / f"{task}-*-b{budget}-*.jsonl"))
    cands = [c for c in cands if (re.search(r"-seed(\d+)\.jsonl$", c) or [None, "42"])[1] == str(seed)]
    rep = arg(args, "page_rep")
    if rep == "oval":
        tab = arg(args, "eta_table")
        key = f"_etab{json.load(open(tab))['hash']}_" if tab else f"_eta{arg(args, 'eta')}_"
        recon = "recon1" if "--oval_recon" in args else "recon0"
        vr = arg(args, "oval_vrank")
        cands = [c for c in cands if key in c and recon in c and
                 ((f"_vr{vr}_" in c) if vr else ("_vr" not in c))]
    else:
        cands = [c for c in cands if "oval" not in c]
    if len(cands) > 1:
        sys.exit(f"ambiguous output for {model} {task} b{budget} s{seed}: {cands}")
    return cands[0] if cands else None


def matrix(files, task):
    """problems x seeds 0/1 matrix if every seed is complete, else (None, n_complete_seeds)."""
    cols, ok = [], 0
    for s in SEEDS:
        f = files.get(s)
        v = outcomes(f) if f else {}
        if len(v) == NEXP[task]:
            ok += 1
            cols.append([v[q] for q in sorted(v)])
    return (np.array(cols).T, ok) if ok == len(SEEDS) else (None, ok)


def boot(x, stat):
    r = np.random.default_rng(RNG)
    idx = r.integers(0, x.shape[0], size=(NB, x.shape[0]))
    b = np.array([stat(x[i]) for i in idx])
    return np.percentile(b, [2.5, 97.5])


avg = lambda m: 100 * m.mean()
pas = lambda m: 100 * m.max(1).mean()


def reference(model, task):
    """Tab. 1 OVAL cell: deployed rule-1 table, rank-16 values, budget 2048."""
    for d in REF_TABLES:
        t = d / f"{model}_{task}_r1.json"
        if t.exists():
            h = json.load(open(t))["hash"]
            files = {}
            for s in SEEDS:
                suf = "" if s == 42 else f"-seed{s}"
                g = [x for d in ("results/reasoning", "results/revision/A3/runs")
                     for x in glob.glob(str(ROOT / f"{d}/{model}-spec_ret/{task}-*-b2048-*_etab{h}_q0_recon1_vr16_vpca-*Mg16384{suf}.jsonl"))]
                if g:
                    files[s] = g[0]
            m, ok = matrix(files, task)
            return m, h, ok
    return None, None, 0


def block(name):
    jobs = defaultdict(dict)
    meta = {}
    for jf in sorted(glob.glob(str(ROOT / f"scripts/revision/jobs_{name}*.txt"))):
        for line in open(jf):
            if not line.strip() or line.startswith("#"):
                continue
            jid, args = line.rstrip("\n").split("|", 1)
            arm = re.sub(r"_s\d+$", "", jid)
            model, task = arg(args, "model"), arg(args, "dataset")
            budget, seed = int(arg(args, "budget")), int(arg(args, "seed"))
            meta[arm] = (model, task, budget)
            f = find(model, task, budget, seed, args, arg(args, "out_root_dir"))
            if f:
                jobs[arm][seed] = f
    outdir = ROOT / f"results/revision/{name}/scores"; outdir.mkdir(parents=True, exist_ok=True)
    rows, refs = [], {}
    for arm in sorted(meta):
        model, task, budget = meta[arm]
        m, ok = matrix(jobs.get(arm, {}), task)
        if m is None:
            rows.append([arm, model, task, budget, f"incomplete ({ok}/8 seeds)"] + [""] * 6)
            continue
        with open(outdir / f"{arm}.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["problem"] + [f"seed{s}" for s in SEEDS])
            for i, r in enumerate(m):
                w.writerow([i] + list(r))
        a, p = avg(m), pas(m)
        alo, ahi = boot(m, avg); plo, phi = boot(m, pas)
        if (model, task) not in refs:
            refs[(model, task)] = reference(model, task)
        rm, rh, rok = refs[(model, task)]
        if rm is not None and budget == 2048:
            d = m - rm
            dlo, dhi = boot(d, avg)
            diff = [f"{avg(d):+.2f}", f"[{dlo:+.2f}, {dhi:+.2f}]", f"rule-1 repo run etab{rh} = {avg(rm):.2f}"]
        else:
            diff = ["", "", "no rule-1 repo run" if rm is None else "budget differs"]
        rows.append([arm, model, task, budget, f"{a:.2f}", f"[{alo:.2f}, {ahi:.2f}]",
                     f"{p:.2f}", f"[{plo:.2f}, {phi:.2f}]"] + diff)
    hdr = ["arm", "model", "task", "budget", "avg@8", "avg@8 CI", "pass@8", "pass@8 CI",
           "avg@8 - rule 1", "paired CI", "reference"]
    tdir = ROOT / "results/revision/tables"; tdir.mkdir(parents=True, exist_ok=True)
    with open(tdir / f"{name}.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([hdr] + rows)
    print(f"\n### {name}\n\n| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")


if __name__ == "__main__":
    for b in sys.argv[1:] or ["A3"]:
        block(b)
