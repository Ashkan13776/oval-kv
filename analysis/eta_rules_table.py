"""Table for the eta-rule study, scored from the files on disk.

Per (model, task): one row per rule (its deployed per-(layer, KV head) table,
identified by the content hash in the harness filename), plus the existing
fixed-eta value-codec cell as a baseline where it exists (qwen: MATH50 eta
0.25, AIME24 1.0, GPQA50c 0.25 at values rank 8).

Columns: eta_hat distribution over layers >= 1 (mean, std, min, median, max,
fraction clipped at 0 / 1), degenerate (layer, head) count, offline regret on
the held-out calibration split (mean over layers >= 1 of L_test(eta_hat) -
min_grid L_test), k, n, avg@k, pass@k, and paired-bootstrap 95% CIs of avg@k
vs the baseline (if any) and vs the best rule: pair on problem x seed,
resample problems, 10k draws.

Partial cells are never mixed in: a seed counts only if its file holds every
problem; a row's k is the number of complete seeds and the pairing uses the
seeds and problems common to the two arms. Rows still running say so.

  python analysis/eta_rules_table.py [--models ...] [--datasets ...] [--out results/eta_rules/table.md]
"""
import argparse, glob, json, os, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT

PAT = r"\\boxed{((?:[^{}]|\{[^{}]*\})*)}"
NPROB = {"MATH50": 50, "AIME24": 30, "GPQA50c": 50}
BASE_ETA = {"ds-r1-qwen-7b": {"MATH50": "0.25", "AIME24": "1.0", "GPQA50c": "0.25"}}
RULE_NAMES = {"r1": "1 endpoint interp.", "r2": "2 quadratic", "r3": "3 eigenspace (bounded)",
              "r4": "4 balanced sacrifice", "r5": "5 sensitivity match", "r6": "6 learned (within-model)",
              "r3u": "3u eigenspace, unbounded [variant]", "r4w": "4w mass-weighted [variant]",
              "r6x": "6x learned, cross-model fit [variant]", "r6w": "6w mass-weighted agg. [variant]"}

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["ds-r1-qwen-7b", "ds-r1-llama-8b"])
ap.add_argument("--datasets", nargs="+", default=["MATH50", "AIME24", "GPQA50c"])
ap.add_argument("--rules", nargs="+", default=["r1", "r2", "r3", "r4", "r5", "r6"])
ap.add_argument("--variants", nargs="*", default=["r3u", "r4w", "r6x", "r6w"],
                help="recorded variants (tables_variants/): shown as rows, never 'best rule'")
ap.add_argument("--root", default=str(ROOT / "results/codec"))
ap.add_argument("--eta_dir", default=str(ROOT / "results/eta_rules"))
ap.add_argument("--boot", type=int, default=10000)
ap.add_argument("--out", default=None)
A = ap.parse_args()
LINES = []
def say(s=""):
    print(s); LINES.append(s)


def outcomes(path):
    v = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        if not line.strip(): continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        m = re.search(PAT, e["pred"])
        v[e["qid"]] = int(bool(m and m.group(1).replace(" ", "") == str(e["answer"]).replace(" ", "")))
    return v


def load_arm(pattern, nexp):
    """-> {seed: {qid: 0/1}} over COMPLETE seeds, number of partial seeds."""
    full, partial = {}, 0
    for f in sorted(glob.glob(pattern)):
        s = re.search(r"-seed(\d+)\.jsonl$", f); s = int(s.group(1)) if s else 42
        v = outcomes(f)
        if len(v) == nexp: full[s] = v
        elif len(v) > 0: partial += 1
    return full, partial


def matrix(arm, seeds, qids):
    return np.array([[arm[s][q] for s in seeds] for q in qids], float)       # [problem, seed]


def paired_ci(arm_a, arm_b, rng):
    """avg@k difference a - b (points) with a 95% CI: pair on (problem, seed),
    resample problems. Returns (diff, lo, hi, n_problems, k_seeds) or None."""
    seeds = sorted(set(arm_a) & set(arm_b))
    if not seeds: return None
    qids = sorted(set.intersection(*[set(arm_a[s]) for s in seeds], *[set(arm_b[s]) for s in seeds]))
    if not qids: return None
    D = (matrix(arm_a, seeds, qids) - matrix(arm_b, seeds, qids)).mean(1) * 100       # per-problem diff
    idx = rng.integers(0, len(qids), (A.boot, len(qids)))
    bs = D[idx].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return D.mean(), lo, hi, len(qids), len(seeds)


def eta_stats(tab):
    e = np.array(tab["eta"])[1:]
    return dict(mean=e.mean(), std=e.std(), min=e.min(), med=np.median(e), max=e.max(),
                at0=(e <= 0).mean(), at1=(e >= 1).mean())


rng = np.random.default_rng(0)
for model in A.models:
    for ds in A.datasets:
        nexp = NPROB[ds]
        rows = []                       # (label, arm dict, partial, meta)
        be = BASE_ETA.get(model, {}).get(ds)
        if be:
            arm, part = load_arm(f"{A.root}/{model}-spec_ret/{ds}-*oval_r8_eta{be}_q0_recon1_vr8_vpca-pQ2*.jsonl", nexp)
            rows.append((f"baseline fixed eta={be}", arm, part, None))
        stats_f = os.path.join(A.eta_dir, f"{model}_{ds}.json")
        st = json.load(open(stats_f)) if os.path.exists(stats_f) else None
        seen = {}
        for rule in A.rules + A.variants:
            variant = rule in A.variants
            tf = os.path.join(A.eta_dir, "tables_variants" if variant else "tables", f"{model}_{ds}_{rule}.json")
            if not os.path.exists(tf):
                if not variant:
                    rows.append((RULE_NAMES[rule], {}, 0, {"missing": True}))
                continue
            tab = json.load(open(tf)); h = tab["hash"]
            arm, part = load_arm(f"{A.root}/{model}-spec_ret/{ds}-*oval_r8_etab{h}_q0_recon1_vr8_vpca-pQ2*.jsonl", nexp)
            meta = {"hash": h, "eta": eta_stats(tab), "deg": tab.get("degenerate_count"),
                    "alias": seen.get(h), "variant": variant}
            if st and "regret" in st and rule in st["regret"]["tables"]:
                meta["regret"] = st["regret"]["tables"][rule]["mean_regret_layers1plus"]
                meta["loss"] = st["regret"]["tables"][rule]["mean_loss_layers1plus"]
            seen.setdefault(h, rule)
            rows.append((RULE_NAMES[rule], arm, part, meta))
        # show a section once its eta tables exist, even before any harness
        # cell finishes -- the per-head eta stats and offline regret are results
        if all(not r[1] for r in rows) and all(not (r[3] or {}).get("hash") for r in rows):
            continue
        say(f"\n### {model} / {ds}   (n_expected = {nexp}; k = complete seeds; CIs: paired bootstrap over problems, 95%)")
        if st and "regret" in st:
            say(f"offline grid optimum on the held-out split: mean loss {np.mean(st['regret']['grid_min_test'][1:]):.4f} "
                f"(relative output error, layers >= 1); calib-grid-opt regret "
                f"{st['regret']['tables']['calib_grid_opt']['mean_regret_layers1plus']:.5f}")
        # best rule by avg@k among complete rows
        def avg(arm):
            if not arm: return None
            seeds = sorted(arm); qids = sorted(set.intersection(*[set(arm[s]) for s in seeds]))
            return matrix(arm, seeds, qids).mean() * 100
        scored = [(avg(r[1]), r[0]) for r in rows if r[1] and r[3] is not None and not r[3].get("variant")]
        best = max(scored)[1] if scored else None
        base = rows[0] if be else None
        say("| arm | eta mean | std | min | med | max | at0 | at1 | degen | regret | k | n | avg@k | pass@k | vs baseline | vs best rule |")
        say("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for label, arm, part, meta in rows:
            if meta and meta.get("missing"):
                say(f"| {label} | table missing |" + " |" * 14); continue
            es = meta["eta"] if meta else None
            etacols = (f" {es['mean']:.3f} | {es['std']:.3f} | {es['min']:.2f} | {es['med']:.3f} | {es['max']:.2f} | "
                       f"{es['at0']:.2f} | {es['at1']:.2f} | {meta.get('deg', '')} | "
                       f"{meta['regret']:.5f} " if es and "regret" in meta else
                       (f" {es['mean']:.3f} | {es['std']:.3f} | {es['min']:.2f} | {es['med']:.3f} | {es['max']:.2f} | "
                        f"{es['at0']:.2f} | {es['at1']:.2f} | {meta.get('deg', '')} | -- " if es else " fixed | | | | | | | | "))
            alias = f" (= {RULE_NAMES[meta['alias']]})" if meta and meta.get("alias") else ""
            if not arm:
                say(f"| {label}{alias} |{etacols}| 0 | -- | -- | -- | | |" + (f"  ({part} running)" if part else "  (not run)"))
                continue
            seeds = sorted(arm); qids = sorted(set.intersection(*[set(arm[s]) for s in seeds]))
            M = matrix(arm, seeds, qids)
            run = f" (+{part} running)" if part else ""
            vb = paired_ci(arm, base[1], rng) if base and base[1] and label != base[0] else None
            vbest = None
            if best and label != best:
                barm = next(r[1] for r in rows if r[0] == best)
                vbest = paired_ci(arm, barm, rng)
            fmt = lambda c: (f"{c[0]:+.2f} [{c[1]:+.2f}, {c[2]:+.2f}]" + (" *" if c[1] > 0 or c[2] < 0 else "")) if c else ""
            say(f"| {label}{alias} |{etacols}| {len(seeds)}{run} | {len(qids)} | {M.mean()*100:.2f} | "
                f"{(M.max(1) > 0).mean()*100:.2f} | {fmt(vb)} | {fmt(vbest) if vbest else ('best' if label == best else '')} |")
        say(f"best rule by avg@k (deployed rules only): {best}.  '*' = CI excludes 0.  "
            f"regret = L_test(eta_hat) - min_grid L_test, mean over layers >= 1 (relative output error).  "
            f"[variant] rows are recorded alternatives (queued on MATH50 only, after the deployed rules).")

if A.out and LINES:
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    with open(A.out, "w") as f:
        f.write("# eta-rule study: harness results\n\nGenerated by analysis/eta_rules_table.py from files on disk.\n")
        f.write("\n".join(LINES) + "\n")
    print(f"\nwrote {A.out}")
