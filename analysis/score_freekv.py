"""Score the FreeKV eta sweep, counting COMPLETE cells only.

Their eval/reasoning/eval.py aggregates whatever .jsonl files exist, including
half-finished ones: it averages a 5-problem seed with a 30-problem seed, and
divides pass@k by n_problem[0] -- which is whichever cell it happened to read
first.  On a live sweep that produces nonsense (pass@k of 320.0).

This wrapper reuses their per-file accuracy logic but drops any seed whose
record count is short of the dataset size, so a partial cell reads as missing
rather than as a wrong number.

  avg@k  mean per-seed accuracy.  Estimates the same quantity at any k, so
         avg@3 is comparable to the paper's avg@8 (just noisier).
  pass@k fraction of problems solved by at least one seed.  NOT comparable
         across different k -- pass@3 is systematically below pass@8.
"""
import json
import sys
from pathlib import Path

from paths import ROOT, FREEKV, use_infinitebench_scorers
FK = FREEKV / "accuracy"
sys.path.insert(0, str(FK))

N_PROBLEMS = {"AIME24": 30, "MATH50": 50, "GPQA50c": 50}
SWEEP = ROOT / "results/sweep"

# Permissive answer extraction, reported ALONGSIDE their strict scorer.
#
# Their get_result requires a \boxed{} that follows a "**Final Answer**" marker.
# Their --loose flag drops the marker requirement but still demands \boxed{} --
# on our GPQA runs that changes nothing (every boxed answer already follows a
# marker), so --loose is NOT what explains the gap to their published column.
#
# What does: 19/50 GPQA generations answer in prose ("**Answer: A**") and score
# zero on format despite being right. Allowing that lifts eta=0 GPQA 26.0 ->
# 38.0, against their published 39.50.
#
# This is OUR extraction, not theirs. It is a diagnostic for how much of a cell
# is format loss; the strict column remains the one comparable to the paper.
ANS_RE = None


def permissive_correct(pred, answer, strict_hit):
    """Superset of their strict scorer: strict_hit OR a prose-stated answer.

    MUST be >= strict by construction. An earlier version took the FIRST
    \\boxed{} in the generation, which on math problems picks up an intermediate
    result and scored BELOW strict -- an incoherent "loose" metric. We take the
    LAST boxed expression (the conclusion) and OR with the strict verdict.
    """
    import re
    if strict_hit:
        return True
    # they compare with spaces stripped from BOTH sides; match that
    ans = str(answer).replace(" ", "")
    boxes = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", pred)
    if boxes and boxes[-1].replace(" ", "") == ans:
        return True
    # prose conclusion, e.g. "**Answer: A**" -- multiple-choice tasks only
    if len(ans) == 1 and ans.isalpha():
        hits = re.findall(r"(?:answer|Answer)\W{0,20}([A-D])\b", pred[-400:])
        return bool(hits) and hits[-1] == ans
    return False
    return False


def cell_key(fname):
    """(dataset, eta, seed) from the output filename."""
    ds = fname.split("-")[0]
    eta = fname.split("eta")[1].split("-")[0] if "eta" in fname else "?"
    seed = fname.split("-seed")[1].split(".")[0] if "-seed" in fname else "42"
    return ds, eta, seed


def main():
    import os
    import types
    os.chdir(FK)                      # their get_result uses relative imports
    from eval.reasoning import eval as fk_eval
    # get_result reads `max_length` and `args` as module globals, which only
    # exist when their eval.py runs as __main__.  Supply them explicitly:
    #   max_length -- only classifies "finished inside the window"; set to our
    #                 max_gen so it means "did not hit the generation cap"
    #   loose=False -- their default, requires the **Final Answer** marker
    fk_eval.max_length = 16384
    fk_eval.args = types.SimpleNamespace(loose=False)
    get_result = fk_eval.get_result

    cells = {}
    for model_dir in sorted(SWEEP.glob("*-spec_ret")):
        model = model_dir.name.replace("-spec_ret", "")
        for f in sorted(model_dir.glob("*.jsonl")):
            ds, eta, seed = cell_key(f.name)
            want = N_PROBLEMS.get(ds)
            n = sum(1 for l in f.open() if l.strip())
            if want is None:
                continue
            # (accuracy, avg_len, avg_len_fin, corrects, n, ...)
            acc, avg_len, _len_fin, corrects, _n, *_ = get_result(str(f))
            rows = [json.loads(l) for l in f.open() if l.strip()]
            # their get_result does `corrects.append(pid+1)` -- the list is
            # ONE-BASED over row position, not a qid. Comparing it against a
            # 0-based enumerate index marks the wrong rows as strict hits and
            # silently inflates the loose column (it read 96% on MATH50 where
            # the true union is 72%).
            strict_ids = {c - 1 for c in corrects}
            loose = 100.0 * sum(
                permissive_correct(r["pred"], r["answer"], i in strict_ids)
                for i, r in enumerate(rows)) / max(len(rows), 1)
            caps = sum(1 for r in rows if r["output_len"] >= 16384)
            cells.setdefault((model, ds, eta), {})[seed] = {
                "n": n, "want": want, "complete": n == want,
                "acc": acc, "avg_len": avg_len, "loose": loose, "caps": caps,
                "corrects": set(corrects),
            }

    for (model, ds, eta), seeds in sorted(cells.items()):
        done = {s: v for s, v in seeds.items() if v["complete"]}
        part = {s: v for s, v in seeds.items() if not v["complete"]}
        line = f"{model}  {ds:<8} eta={eta:<5}"
        if done:
            accs = [v["acc"] for v in done.values()]
            loos = [v["loose"] for v in done.values()]
            caps = sum(v["caps"] for v in done.values())
            tot = sum(v["n"] for v in done.values())
            union = set().union(*[v["corrects"] for v in done.values()])
            k = len(done)
            line += (f" strict@{k}={sum(accs)/k:6.2f}  loose@{k}={sum(loos)/k:6.2f}"
                     f"  pass@{k}={100*len(union)/seeds[list(done)[0]]['want']:6.2f}"
                     f"  cap={caps}/{tot}  seeds={sorted(done)}")
        else:
            line += "  (no complete seed yet)"
        if part:
            line += "  | in-flight: " + ", ".join(
                f"seed{s} {v['n']}/{v['want']}" for s, v in sorted(part.items()))
        print(line)


if __name__ == "__main__":
    main()
