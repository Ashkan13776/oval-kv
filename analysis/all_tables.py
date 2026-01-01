"""Every completed run, scored from disk, as tables.

Four experiments live in this repo, each with its own scorer and its own
budget convention. This prints them all from the raw prediction files so the
numbers cannot drift from a stale notes file.

  1. InfiniteBench   Llama-3.1-8B, FreeKV harness, official IB scorers
  2. LongBench v2    Qwen-2.5-14B, their judge field, eta + rank sweeps
  3. FreeKV Table 3  DeepSeek-R1 x3, reasoning benchmarks, avg@k
  4. Fig. 6 latency  Avg TBT (decode) from the perf engine

BUDGET CONVENTIONS DIFFER and are printed per table: the accuracy harness ADDS
sink+recent to --budget, the perf harness CARVES them OUT.
"""
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from paths import ROOT, FREEKV, use_infinitebench_scorers
use_infinitebench_scorers()
from compute_scores import get_score_one  # noqa: E402

FK = FREEKV / "accuracy"
# Two roots: runs predating the --out_root_dir fix landed in
# results/, later ones in res/. Scan both or half the arms vanish.
from paths import IB_DIRS  # noqa: F811
LB2_DIR = FK / "eval/LongBench2/results"
SWEEP = ROOT / "results/sweep"
FIG6 = ROOT / "results/fig6_latency"

IB_TASKS = [
    ("passkey", "passkey"), ("num-str", "number_string"),
    ("kv-retr", "kv_retrieval"), ("dialog-qa", "longdialogue_qa_eng"),
    ("bk-sum", "longbook_sum_eng"), ("bk-choice", "longbook_choice_eng"),
    ("bk-qa-en", "longbook_qa_eng"), ("bk-qa-zh", "longbook_qa_chn"),
    ("math-find", "math_find"), ("code-run", "code_run"),
    ("code-dbg", "code_debug"),
]
# No task is excluded by hand any more. The old kv-retr exclusion rested on
# FullKV scoring 0.0 there, which was the tuple-KV cache bug, not the task.
# Headroom is now DERIVED: a task can only rank selection quality if the dense
# reference actually beats the sparse arms on it.
HEADROOM_EPS = 0.05


def _pat(frag, task):
    """Glob for one arm's task file. An empty fragment must not produce '**',
    which pathlib rejects as a partial path component."""
    return f"*{frag}*-{task}.jsonl" if frag else f"*-{task}.jsonl"


def rule(title, sub=""):
    print("\n" + "=" * 78)
    print(title)
    if sub:
        print(sub)
    print("=" * 78)


# ---------------------------------------------------------------- 1. IB ----
def table_infinitebench():
    rule("1. InfiniteBench — Llama-3.1-8B-Instruct, FreeKV harness",
         "B=2048 as sink 128 + recent 128 + budget 1792, tau 0.8, page 32, "
         "avgSM,\nspec_ret_steps 2, greedy, 50 seeded-random records/task "
         "(seed 42), official scorers.")
    arms = []
    for label, sub, frag, excl in [
        ("FreeKV (quest)", "spec_ret", "avgSM-pQ2", "oval"),
        ("OVAL r8 eta.25", "spec_ret", "oval_r8_eta0.25", ""),
        ("ArkVale", "arkv", "", ""),
        ("RaaS", "raas", "", ""),
        ("FullKV", "full", "", ""),
    ]:
        dirs = [r / f"llama-3.1-chat-8b-{sub}" for r in IB_DIRS]
        col = {}
        for short, task in IB_TASKS:
            hits = [h for d in dirs if d.is_dir()
                    for h in d.glob(_pat(frag, task))
                    if not (excl and excl in h.name)]
            if len(hits) > 1:
                raise SystemExit(f"ambiguous: {[h.name for h in hits]}")
            if not hits:
                continue
            rows = [json.loads(l) for l in open(hits[0]) if l.strip()]
            if not rows:
                continue
            col[short] = (100.0 * np.mean(
                [get_score_one(r["prediction"], r["ground_truth"], task,
                               "llama-3.1-chat-8b") for r in rows]),
                len(rows))
        if col:
            arms.append((label, col))
    if not arms:
        print("  (no runs)"); return

    hdr = f"{'task':<22}" + "".join(f"{a[0]:>17}" for a in arms)
    print("\n" + hdr); print("-" * len(hdr))
    # headroom = FullKV - best sparse arm. <=0 means every method already ties
    # the dense model, so the task cannot separate them.
    full = dict(arms).get("FullKV")
    sparse = [c for l, c in arms if l != "FullKV"]

    def headroom(short):
        if not full or short not in full:
            return None
        best = max((c[short][0] for c in sparse if short in c), default=None)
        return None if best is None else full[short][0] - best

    for short, _t in IB_TASKS:
        h = headroom(short)
        mark = " =" if h is not None and h <= HEADROOM_EPS else "  "
        line = f"{short + mark:<22}"
        for _l, col in arms:
            if short not in col:
                line += f"{'--':>15}  "
            else:
                v, n = col[short]
                # a cell still being written must not read like a final number
                line += f"{v:>15.1f}{'~' if n < 50 else ' '} "
        print(line)
    print("-" * len(hdr))
    allt = [s for s, _ in IB_TASKS]
    sep = [s for s in allt
           if headroom(s) is not None and headroom(s) > HEADROOM_EPS]
    rows = [(f"mean (all {len(allt)})", allt)]
    if sep:
        rows.append((f"mean ({len(sep)} w/ headroom)", sep))
    for name, keys in rows:
        line = f"{name:<22}"
        for _l, col in arms:
            cells = [col[k] for k in keys if k in col]
            done = len(cells) == len(keys) and all(n == 50 for _v, n in cells)
            line += (f"{np.mean([v for v, _n in cells]):>15.2f}  " if done
                     else f"{'--':>15}  ")
        print(line)
    ns = {n for _l, c in arms for _, n in c.values()}
    if sep:
        print(f"\n  = : FullKV does not beat the best sparse arm, so the task "
              f"cannot rank\n      selection quality. Tasks WITH headroom: "
              f"{', '.join(sep)}.")
    print(f"  ~ : still running, scored from a partial file. "
          f"records/task seen: {sorted(ns)}")


# --------------------------------------------------------------- 2. LB2 ----
def table_longbench2():
    rule("2. LongBench v2 — Qwen-2.5-14B-Instruct",
         "Their own command: --sink 128 --recent 128 --budget 1792 (B=2048), "
         "tau 0.8,\npage 32, avgSM, spec_ret_steps 2, corr 0.8, 64K "
         "truncation, greedy, 503 questions.")
    rows = []
    for f in sorted(LB2_DIR.glob("*.jsonl")):
        data = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        if not data:
            rows.append((f.name, None, len(data))); continue
        n = len(data)
        agg = lambda pred: (100.0 * sum(int(p["judge"]) for p in data if pred(p))
                            / max(1, sum(1 for p in data if pred(p))))
        rows.append((f.name, dict(
            overall=100.0 * sum(int(p["judge"]) for p in data) / n,
            easy=agg(lambda p: p["difficulty"] == "easy"),
            hard=agg(lambda p: p["difficulty"] != "easy"),
            short=agg(lambda p: p["length"] == "short"),
            medium=agg(lambda p: p["length"] == "medium"),
            long=agg(lambda p: p["length"] == "long"),
        ), n))

    def label(name):
        if "-full-" in name:
            return "FullKV"
        m = re.search(r"oval_r(\d+)_eta([\d.]+)", name)
        return f"OVAL r{m.group(1)} eta={m.group(2)}" if m else "FreeKV (quest)"

    hdr = (f"{'arm':<22}{'n':>5}{'overall':>9}{'easy':>8}{'hard':>8}"
           f"{'short':>8}{'medium':>8}{'long':>8}")
    print("\n" + hdr); print("-" * len(hdr))
    for name, s, n in sorted(rows, key=lambda r: label(r[0])):
        if s is None:
            print(f"{label(name):<22}{n:>5}   (empty — run did not complete)")
            continue
        print(f"{label(name):<22}{n:>5}{s['overall']:>9.2f}{s['easy']:>8.2f}"
              f"{s['hard']:>8.2f}{s['short']:>8.2f}{s['medium']:>8.2f}"
              f"{s['long']:>8.2f}")
    print("\n  Paper's published column (their harness): FullKV 33.40, "
          "FreeKV 34.19.\n  Compare WITHIN this table only — our harness "
          "reproduces their own method\n  1.59 points low, so cross-harness "
          "deltas are not interpretable.")


# ------------------------------------------------------------- 3. Table3 ----
def table_freekv_t3():
    rule("3. FreeKV Table 3 — DeepSeek-R1 distills, reasoning benchmarks",
         "sink 512 + recent 512 + --budget 2048, tau 0.9, page 32, avgSM, "
         "spec_ret_steps 2,\nT=0.6 top-p 0.95, max_gen 16384, 1 seed. "
         "strict = their scorer; loose = OUR permissive\nextraction, a "
         "diagnostic for format loss, NOT comparable to the paper.")
    print("\n  !! BUDGET: the accuracy harness ADDS sink+recent to --budget, so "
          "these cells\n     attended 512+512+2048 = 3072 tokens, NOT the "
          "paper's B=2048. Every arm here\n     carries the same offset, so "
          "the eta comparison is internally valid; the\n     LEVELS are not "
          "comparable to the paper's column. Re-running one cell at the\n"
          "     correct b1024 moved it 0.00 points — measured once, not "
          "verified elsewhere.")
    import io, contextlib
    import score_freekv
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        score_freekv.main()
    os.chdir(ROOT)                       # score_freekv chdirs into their tree

    cells = {}
    for line in buf.getvalue().splitlines():
        m = re.match(r"(\S+)\s+(\S+)\s+eta=([\d.]+)\s+strict@\d+=\s*([\d.]+)"
                     r"\s+loose@\d+=\s*([\d.]+).*?cap=(\d+)/(\d+)", line)
        if m:
            model, ds, eta, strict, loose, cap, tot = m.groups()
            cells[(model, ds, float(eta))] = (float(strict), float(loose),
                                              int(cap), int(tot))
    if not cells:
        print("  (no complete cells)"); return

    etas = sorted({e for _m, _d, e in cells})
    for model in ["ds-r1-qwen-7b", "ds-r1-llama-8b", "ds-r1-qwen-14b"]:
        bs = sorted({d for m, d, _e in cells if m == model})
        if not bs:
            continue
        print(f"\n  {model}")
        hdr = f"    {'bench':<16}" + "".join(f"{'eta=' + str(e):>10}" for e in etas)
        print(hdr); print("    " + "-" * (len(hdr) - 4))
        def row(name, get):
            line = f"    {name:<16}"
            for e in etas:
                v = get(e)
                line += f"{v:>10.2f}" if v is not None else f"{'--':>10}"
            print(line)

        for ds in bs:
            row(ds, lambda e, d=ds: (cells[(model, d, e)][0]
                                     if (model, d, e) in cells else None))
            # loose is only worth a row where it actually differs from strict
            if any((model, ds, e) in cells and
                   cells[(model, ds, e)][1] != cells[(model, ds, e)][0]
                   for e in etas):
                row(f"  {ds} loose",
                    lambda e, d=ds: (cells[(model, d, e)][1]
                                     if (model, d, e) in cells else None))
        row("mean (strict)", lambda e: (
            np.mean([cells[(model, d, e)][0] for d in bs
                     if (model, d, e) in cells])
            if all((model, d, e) in cells for d in bs) else None))
        caps = sum(cells[(model, d, e)][2] for d in bs for e in etas
                   if (model, d, e) in cells)
        tots = sum(cells[(model, d, e)][3] for d in bs for e in etas
                   if (model, d, e) in cells)
        print(f"    generations that hit the 16384-token cap: {caps}/{tots}")


# --------------------------------------------------------------- 4. Fig6 ----
def table_latency():
    rule("4. Fig. 6 latency — Avg TBT (decode), ms/token, batch 1",
         "PERF harness: sink/window are CARVED OUT of --budget here (opposite "
         "of the\naccuracy path). 2048/32 = 64 pages; topk 32 + 16 sink + 16 "
         "window = 64 pages = B.\nRTX PRO 6000 Blackwell 97GB; the paper used "
         "A100 40GB, so absolute\noffloading speedups are not expected to "
         "transfer.")
    arms = [("arkvale", "ArkVale"), ("raas", "RaaS"), ("freekv", "FreeKV"),
            ("locks0", "OVAL eta=0"), ("locks50", "OVAL eta=0.5"),
            ("locks100", "OVAL eta=1.0")]
    scen = [("longinput", "Long Input (32K in / 512 out)"),
            ("longgen", "Long Generation (600 in / 16K out)")]
    for model in ["qwen-2.5-chat-7b", "llama-3.1-chat-8b"]:
        print(f"\n  {model}")
        hdr = f"    {'arm':<14}" + "".join(f"{s[1].split(' (')[0]:>22}" for s in scen)
        print(hdr); print("    " + "-" * (len(hdr) - 4))
        base = {}
        for key, lbl in arms:
            line = f"    {lbl:<14}"
            for sc, _t in scen:
                p = FIG6 / f"{model}_{sc}_{key}_bs1.log"
                m = (re.search(r"Avg TBT \(decode\) : ([\d.]+)", p.read_text())
                     if p.exists() else None)
                if not m:
                    line += f"{'--':>22}"; continue
                v = float(m.group(1))
                base.setdefault(sc, v)
                rel = f"({v / base[sc]:.2f}x)" if key != "arkvale" else ""
                line += f"{v:>15.2f} ms {rel:>6}"
            print(line)
        print("    (x) = relative to ArkVale on the same scenario")


if __name__ == "__main__":
    table_infinitebench()
    table_longbench2()
    table_freekv_t3()
    table_latency()
