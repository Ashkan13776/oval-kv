# Protocol decisions

Resolving `open-questions.md`. Decided with the user on 2026-08-12. These are
explicit choices standing in for parameters the paper does not publish; each
must be reported alongside any number we produce.

| # | Question | Decision |
| --- | --- | --- |
| Q1 | 50-record subset | ~~First 50 records~~ **REVISED: seeded random draw, seed 42** (`harness/subset.py`). First-50 was wrong — the .jsonl files are pre-sorted, making it a badly biased sample. See [finding-kv-retr.md](finding-kv-retr.md). Not the paper's draw (their seed is unpublished) — see D4. |
| Q2 | Prompt templates | **InfiniteBench official reference implementation**, no GLM chat template on top. |
| Q3 | Generation length | InfiniteBench official **per-task `max_new_tokens`**. |
| Q4 | Sampling | **Greedy**. Inferred, not stated by the paper for this suite. |
| Q5 | Truncation manner | InfiniteBench official **middle-truncation** (keep head+tail, drop middle), applied identically at both budgets. |
| Q5b | Truncation length | **1,048,576 = the model's native window**, NOT the reference impl's 128K. See below. |

## Q5b: why native 1M and not the reference impl's 128K

User directive: "be as exact as possible". The reference `eval_chatglm.py:23`
hardcodes `TRUNCATE_LEN = 128 * 1024` with the comment "Determined by the
model" — but that constant was written for ChatGLM3's 128K window, not
GLM-4-9B-Chat-1M's 1M. Applying it would truncate **215 of 550** records.

At the model's native 1M, **546 of 550 records run untruncated**. The 4 that
don't (all `longbook_qa_chn`) exceed 1M outright, so they were truncated in the
paper too.

Textual support: appC_extres.tex:11 justifies the model choice by "its native
1M context", which reads as a statement that the suite was *not* capped at
128K. The "100K+ context" phrasing is consistent with untruncated data too
(median record is 125-180K).

This remains an inference. If our numbers miss the paper's, re-running at
`TRUNCATE_LEN=131072` is the first alternative to try — it is the single
largest unresolved protocol fork.
| Q6 | `score` metric | InfiniteBench **official per-task scorers**. |

## Scope

Reproducing the **LOCKS row only**, at b=512 and b=2048, accuracy only.
Not reproducing: FullKV, Oracle (exact-LSE), RocketKV, ShadowKV, Quest.
Not reproducing: any latency/efficiency claim (see deviations.md D1).

## Primary endpoint

The 11-task **Avg**: 41.1 ±2.4 at b=512, 43.6 ±2.6 at b=2048.
Per-task cells are secondary — most have CI half-widths above 9 points at
n=50, so they cannot discriminate much on their own.
