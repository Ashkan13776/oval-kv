# RESOLVED: kv-retr scored 60.0 vs the paper's 28.0 — our subset was biased

## Symptom

    kv-retr  b=2048   ours 60.0   paper 28.0 ±12.6   delta +32.0   OUTSIDE CI

The tell was not the size of the gap but its direction: we also beat the
paper's **FullKV** reference, which is 28.0 on this task at every budget. A
method attending ~2% of tokens cannot outscore the dense full-cache upper bound
on the same data. So the fault had to be in our protocol, not in LOCKS.

## Root cause

**InfiniteBench's .jsonl files are ordered, not shuffled, and we were taking
the first 50 records.**

`kv_retrieval.jsonl` is sorted by needle position. The first 20 records place
the answer at exactly 0.0%, 0.2%, 0.4%, 0.6% ... of the context — an arithmetic
progression:

| subset | p25 | p50 | p75 | answer in leading 10% |
| --- | --- | --- | --- | --- |
| first 50 (what we ran) | 2.4% | 5.0% | 7.4% | **50 / 50** |
| all 500 | 25.0% | 50.0% | 75.0% | 50 / 500 |

Every record in our subset had its needle inside the leading 10% of context —
precisely where LOCKS' always-kept **sink page** lives. The task was trivial by
construction, and the inflation is largest for exactly the methods that retain
a sink.

## Extent of the contamination

Ordering is not limited to one task. Measured over the shipped data:

| task | ordering | first-50 vs full split |
| --- | --- | --- |
| `passkey` | sorted by needle position | p50 3.5% vs 50.0% |
| `number_string` | sorted by needle position | p50 3.5% vs 50.0% |
| `kv_retrieval` | sorted by needle position | p50 5.0% vs 50.0% |
| `longbook_qa_chn` | sorted by length | median 233K vs 1,139K chars (5x short) |
| `code_debug` | sorted by length | median 667K vs 431K chars (1.55x long) |

At least 5 of 11 tasks were sampled unrepresentatively.

`passkey` and `number_string` still scored exactly 100.0, matching the paper —
but that was luck, not validation. Those two are 100.0 for every method at
every budget in the paper, including Quest. They cannot detect this bias.

## Why my earlier check missed it

I tested whether `kv_retrieval` records were homogeneous and found all 500 have
byte-identical context length (200,011 chars), then concluded a subset could not
matter. Constant *length* is not constant *difficulty*. The needle position is
the difficulty axis on this task and it varies from 0% to 100%.

## Fix

`harness/subset.py` — a seeded random draw
(`random.Random(seed).sample(range(total), 50)`) shared by the runner and the
length-measurement script so they cannot diverge. Default seed 42, recorded in
each run's `run_config.json`.

This also settles the reading of the paper's wording. "Seeded subset of 50
records per task" (tab_infinitebench_main.tex:3) only makes sense as a random
draw — there is nothing to seed about taking the first 50, and on pre-sorted
data first-50 would be indefensible.

## Residual uncertainty

The paper's own seed is unpublished, so our draw is not their draw. With n=50
and per-task CIs above ±9 points on most tasks, per-task cells will move between
seeds. The 11-task Avg (±2.6) remains the primary endpoint.

## Cost

~2.5 h of GPU time discarded (4 tasks of b=2048). All results moved to
`results/_invalid_first50/`.

## Confirmed fixed

Re-ran the same cell with the seeded random subset:

| subset | kv-retr @ b=2048 | vs paper 28.0 ±12.6 |
| --- | --- | --- |
| first 50 (biased) | 60.0 | +32.0, far outside CI |
| **seeded random (seed 42)** | **30.0** | **+2.0, within CI** |

The whole retrieval group now agrees:

    Retr   ours 76.7 ±4.0   paper 76.0 ±4.1   +0.7   within CI

Two things this establishes beyond the fix itself:

1. Our bootstrap is calibrated to theirs — our group CI half-width (4.0) lands
   on their published 4.1, computed independently from our own records.
2. `kv-retr` is the most discriminating retrieval cell in the table (paper has
   Quest 0.0, ShadowKV 14.0, LOCKS 28.0 at this budget), so agreement here is
   evidence the page-local summary genuinely finds carrier pages at ~2%
   density, not merely that easy needle tasks pass.

## Side effect on run cost

The corrected subset is heavier than the biased one: 109.8M prompt tokens vs
92.0M (+19%). `longbook_qa_chn` alone went from 22.9M to 41.5M tokens (median
179K -> 550K), making it 38% of the run, and records over our ~786K ceiling
went from 4 to 17. Revised estimate ~14 h per budget.
