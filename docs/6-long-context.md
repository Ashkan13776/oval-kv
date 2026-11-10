# 6. Long context: the full pipeline on LongBench v2 and LongGenBench

Piece 6. Previous: [5. full pipeline](5-full-pipeline.md). Overview: [docs/README.md](README.md).
Agent brief and status: [agents/long-context.md](agents/long-context.md).

**Status (2026-11-10): done.** Pieces 2-4 were built and validated on long
reasoning only. The paper's long-context numbers (README Table 2) are retrieval
only, with exact keys and values. This piece runs all four pieces together on
both Table 2 benchmarks and all three long-context models, so that OVAL's
full-pipeline column sits next to Table 2. Table:
[`results/long_context/table_kr16_vr32.md`](../results/long_context/table_kr16_vr32.md).

## Results

FreeKV and eta* columns: the paper's Table 2 as reported (eta* = best of five
shared etas per cell, retrieval only, exact keys and values). OVAL: keys rank 16
(scoring and reconstruction), values rank 32, per-head eta from rule 1.

| Model | Benchmark | Metric | FreeKV (paper) | eta* (paper) | **OVAL full pipeline** |
|---|---|---|---|---|---|
| Qwen-2.5-7B | LongBench v2 | Overall | 26.84 | 28.03 | **27.83** |
| | | Short / Medium / Long | 34.44 / 22.33 / 23.15 | 36.11 / 22.79 / 25.93 | **36.11 / 22.33 / 25.00** |
| | LongGenBench | CR | 76.93 | 78.54 | **75.29** |
| | | CR x Acc | 32.81 | 32.38 | **32.48** |
| Llama-3.1-8B | LongBench v2 | Overall | 29.22 | 29.82 | **30.02** |
| | | Short / Medium / Long | 35.00 / 27.44 / 23.15 | 35.00 / 28.84 / 25.93 | **36.11 / 27.44 / 25.00** |
| | LongGenBench | CR | 78.03 | 74.71 | **72.48** |
| | | CR x Acc | 27.62 | 28.50 | **27.01** |
| Qwen-2.5-14B | LongBench v2 | Overall | 34.19 | 33.00 | **32.60** |
| | | Short / Medium / Long | 41.11 / 33.49 / 24.07 | 41.11 / 30.23 / 25.93 | **41.11 / 29.77 / 24.07** |
| | LongGenBench | CR | 65.46 | 63.92 | **61.81** |
| | | CR x Acc | 29.39 | 28.64 | **26.66** |

LongBench v2: all 503 questions; unparsable answers count as wrong, as in their
`result.py` (33 / 65 / 34 for Llama-8B / Qwen-7B / Qwen-14B: the model starts
reasoning and hits the answer-length cap before writing a letter, on the same
questions for all three models). LongGenBench: all 400 items of
`Dataset_short.json`, one seed (42).

**Reading the LongGenBench gaps.** The CR gaps come from items where the model
stops after a few entries ("Please wait for my response to continue ...",
"*** continues with the next 90 blocks ***", or a stray `*** finished`, which
the harness treats as the stop string), not from degenerate output: items that
hit the 16K cap are coherent and score ~90. On Llama-8B, a **dense control**
(same harness, same per-item seeds, 3K-token cap, all 400 items) stops early on
87 / 400 items against OVAL's 94 / 400 (by type: Week 4 vs 5, Floor 21 vs 21,
Menu Week 27 vs 30, Block 35 vs 38). The difference is ~1.5 CR points, within
one-seed noise, so dense in this harness sits at ~74, i.e. at the paper's eta*.
On Qwen, 46 of the 100 Qwen-14B Menu Week items stop before prompt + output
leaves the 2,048-token exact window, where OVAL does not act at all.

## The configuration

| component | setting | flag(s) |
|---|---|---|
| page retrieval | OVAL scoring with rank-16 key records, avgSM, FreeKV spec_ret | `--method spec_ret --page_rep oval --oval_rank 16 --GQA_policy avgSM --spec_ret_steps 2 --page_size 32` |
| keys | selected pages rebuilt from the rank-16 key records | `--oval_recon` |
| values | selected pages rebuilt from rank-32 PCA value records (**lossless** at page 32: the rank clamps to 31 = full rank of a centred page) | `--oval_vrank 32 --vbasis pca` |
| eta | one per (layer, KV head), rule 1, one table per (model, benchmark), calibrated at keys rank 16 / values rank 32 on held-out long-context data | `--eta_table results/long_context/eta_rules/kr16_vr32/tables/<model>_<bench>_r1.json` |
| LongBench v2 protocol | the paper's: sink 128, recent 128, budget 1792 (B = 2048), correction 0.8, greedy, inputs middle-truncated to 64K, all 503 questions | `--sink 128 --recent 128 --budget 1792 --spec_ret_corr 0.8`, `KVC_MAX_TOKENS=65536` |
| LongGenBench protocol | FreeKV's command with **top-p 0.95**: sink 512, recent 512, budget 1024, correction 0.9, T 0.95, 16,000 new tokens, `Dataset_short.json` (400 items), one seed (42) | `--spec_ret_corr 0.9 --sink 512 --recent 512 --budget 1024 --max_gen 16000 --temperature 0.95 --top_p 0.95 --seed 42` |

Models: `qwen-2.5-chat-7b`, `llama-3.1-chat-8b`, `qwen-2.5-chat-14b`, all bf16 records (`OVAL_QUANT=0`).

**Why keys rank 16 (not 8).** The calibration's rank check (relative output
error at eta 0.5) showed the rank-8 key error growing with context: on Qwen-7B,
3.1% at 6.5K to 7.2% at 64K, against 2.2% for retrieval alone; rank 16 gives
2.9% at 64K.

**Why top-p 0.95 (not FreeKV's default 1.0).** FreeKV's README command leaves
top-p at 1.0. Sampling the full vocabulary halves the completion rate: on 16
Qwen-14B Week items, same seeds, dense CR was 22.4 at top-p 1.0 and 49.4 at
top-p 0.95 (OVAL at top-p 1.0: 16.7). The brief states the paper used 0.95.

## Where compression acts in long context

The harness prefill is dense. Pages are built at the first decode step, and
selection and reconstruction act only on decode steps. On LongBench v2 that
means the ~10 answer tokens per question, each attending to up to 64K tokens.
On LongGenBench it means the up to 16K generated tokens after a 450-800-token
prompt, close to the reasoning regime rule 1 was built on; until prompt +
output passes sink + recent + budget (2,048 tokens), attention is exact.

## Rule-1 calibration for long context

`analysis/eta_rules_calibrate_lc.py` imports the reasoning calibration's
measurement code (`window_stats`, rule 1, `write_table`) from
`analysis/eta_rules_calibrate.py`, unchanged. Only the data and the capture differ:

| | LongBench v2 | LongGenBench |
|---|---|---|
| held-out data | LongBench-Pro (English), 3 documents in each of the 16k / 32k / 64k / 128k buckets (2 calibrate, 1 test per bucket), framed like LongBench v2's 0-shot prompt, truncated as LongBench v2 is | `Dataset_long.json` from the LongGenBench repo: same four templates, different items, no prompt in common with the evaluated set. 6 prompts, 4 calibrate and 2 test |
| queries | one window ending on a page boundary after 64 greedy answer tokens. Its 96 queries are the decode steps plus the last prompt tokens (the question), at the full context length | the reasoning recipe: 6144 sampled tokens (T 0.95, top-p 1.0, seed 7), 3 windows of 96 generated positions, stride 1024 |
| page layout | sink 128, recent 128, budget 1792 | sink 512, recent 512, budget 1024 |

The capture is **streamed**: a 64K capture of every layer's K/V is 3.8-12.9 GB
per prompt, so each prompt is captured on the GPU, measured, and only its
accumulators are checkpointed (`logs/long_context/calib_ckpt/`, resumable per
prompt). A (layer, window) whose page eigendecomposition fails to converge is
skipped and recorded in the stats file (2 in all, Qwen-7B LongGenBench). The
first four calibration prompts also run the **rank check** above.

## Run

```bash
KV_PY=$(which python) JUDGE_PY=<vllm env python> KRANK=16 VRANK=32 TOPP=0.95 \
  GPUS="0 1" SLOTS=1 STEPS="calib rules" scripts/long_context_pipeline.sh
KV_PY=$(which python) JUDGE_PY=<vllm env python> KRANK=16 VRANK=32 TOPP=0.95 \
  GPUS="0 1" SLOTS=2 STEPS="run judge score" scripts/long_context_pipeline.sh
```

Steps: `calib` (rule-1 stats per model and benchmark) -> `rules` (eta tables) ->
`run` (the largest model's LongGenBench shards, then LongBench v2 for every
model, then the other LongGenBench shards; `LB2_EARLY=0` puts all LongGenBench
first) -> `judge` (Qwen3-32B, vLLM) -> `score`. Every job is resumable, is
gated on the free memory it needs, and leaves
`logs/long_context/done_kr16_vr32_topp0.95/<job>.ok`.

LongGenBench goes through `scripts/long_context_lgb_pred.py`, which wraps
`eval/LongGenBench/pred.py` and adds three things it lacks:
- **real seeds:** pred.py never seeds torch, and its `--seed` only renames the file. The wrapper samples item i under seed `42 * 1000003 + i`.
- **resume:** pred.py appends with no resume, so a restart duplicates records.
- **shards** (4 per model, interleaved, so every shard sees all four types).

Scoring (`analysis/score_long_context.py`):
- LongBench v2: accuracy from the harness `judge` field, Overall and Short / Medium / Long, as their `result.py` does.
- LongGenBench: CR from `calculate_completion_rate` (no judge), and CR x Acc = CR x mean(accuracy once / range / periodic) from the Qwen3-32B judge (`eval/LongGenBench/eval.py`). The judge's own CR matches ours on every model.

The judge needs vLLM, which is not in the harness env. It runs in its own env:
vLLM 0.11.0, torch 2.8.0+cu128, transformers 4.57.1 (0.10.x wheels are built for
CUDA 12.6 and lack sm_120; `pip -U` pulls a transformers that breaks vLLM 0.11).
It takes 3-5 minutes per model on one 97 GB card.

Cost (2x RTX PRO 6000 Blackwell, two jobs per card, measured wall time):
LongGenBench ~18 h of both cards for Qwen-14B and ~14-16 h for each of the 7B /
8B (Qwen-14B outputs average 4K tokens on Week and 10K on Floor items; ~160K
generated tokens per hour for 14B, ~270K for 8B, across both cards);
LongBench v2 2-3.6 h per model on one job slot.

## Setup pitfall: refresh the harness copy of `oval/page_basis.py`

`setup.sh` copies `oval/page_basis.py` into the harness as `kvc/patch/oval.py`.
A copy from before commit 59ebb97 builds value records with batched 128x128
`eigh`. At the first decode step of a 64K prompt, that is ~3.6 s per call and
**~7.6 min per LongBench v2 question** (measured), against ~7 s with the
current 32x32 token-domain form. After pulling, re-run `setup.sh` or
`install -Dm644 oval/page_basis.py third_party/FreeKV/accuracy/kvc/patch/oval.py`.

## Files

- `results/long_context/eta_rules/kr16_vr32/<model>_<bench>.json`: calibration stats (rule-1 sums, loss grid, rank check, prompts used, skipped slices)
- `results/long_context/eta_rules/kr16_vr32/tables/<model>_<bench>_r1.json`: deployable eta tables
- `results/long_context/raw/`: finished harness outputs (LongBench v2 `.jsonl`; LongGenBench `<model>.jsonl.gz` with the judge's verdicts)
- `results/long_context/table_kr16_vr32.md`: the scored table next to the paper's FreeKV and eta* columns
