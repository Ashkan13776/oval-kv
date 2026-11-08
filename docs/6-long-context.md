# 6. Long context: the full pipeline on LongBench v2 and LongGenBench

Piece 6. Previous: [5. full pipeline](5-full-pipeline.md). Overview: [docs/README.md](README.md).
Agent brief and status: [agents/long-context.md](agents/long-context.md).

**Status (2026-11-08): running.** Pieces 2-4 were built and validated on long
reasoning only. The paper's long-context numbers (README Table 2) are retrieval
only, with exact keys and values. This piece runs all four pieces together on
both Table 2 benchmarks and all three long-context models, so that OVAL's
full-pipeline column can sit next to Table 2. Results:
[`results/long_context/table.md`](../results/long_context/table.md).

## The configuration

| component | setting | flag(s) |
|---|---|---|
| page retrieval | OVAL scoring with rank-8 key records, avgSM, FreeKV spec_ret | `--method spec_ret --page_rep oval --oval_rank 8 --GQA_policy avgSM --spec_ret_steps 2` |
| keys | selected pages rebuilt from the rank-8 key records | `--oval_recon` |
| values | selected pages rebuilt from rank-32 PCA value records (**lossless** at page 32: the rank clamps to 31 = full rank of a centred page) | `--oval_vrank 32 --vbasis pca` |
| eta | one per (layer, KV head), rule 1, one table per (model, benchmark), calibrated on held-out long-context data | `--eta_table results/long_context/eta_rules/tables/<model>_<bench>_r1.json` |
| LongBench v2 protocol | the paper's: sink 128, recent 128, budget 1792 (B = 2048), page 32, correction 0.8, greedy, inputs middle-truncated to 64K, all 503 questions | `--sink 128 --recent 128 --budget 1792 --spec_ret_corr 0.8`, `KVC_MAX_TOKENS=65536` |
| LongGenBench protocol | FreeKV's own command: sink 512, recent 512, budget 1024 (default), correction 0.9, T 0.95, top-p 1.0, 16,000 new tokens, `Dataset_short.json` (400 items), one seed (42) | `--spec_ret_corr 0.9 --sink 512 --recent 512 --budget 1024 --max_gen 16000 --temperature 0.95 --seed 42` |

Models: `qwen-2.5-chat-7b`, `llama-3.1-chat-8b`, `qwen-2.5-chat-14b`, all bf16 records (`OVAL_QUANT=0`).

## Where compression acts in long context

The harness prefill is dense. Pages are built at the first decode step, and
selection and reconstruction act only on decode steps. On LongBench v2 that
means the ~10 answer tokens per question, each attending to up to 64K tokens.
On LongGenBench it means the up to 16K generated tokens after a ~600-token
prompt, which is close to the reasoning regime rule 1 was built on.

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
prompt). The first prompts also run a **rank check**: the relative output error
at eta 0.5 for retrieval alone, key ranks 4 / 8 / 16 and value ranks 8 / 16 / 31.
That checks whether the reasoning operating points hold at 64K. In the smoke run
on Qwen-7B, the key rank-8 error grew from 3.1% (6.5K context) to 7.2% (64K),
while retrieval alone grew from 0.7% to 2.2%; rank 16 gave 2.9% at 64K.

## Run

```bash
KV_PY=$(which python) JUDGE_PY=<vllm env python> GPUS="0 1" SLOTS=1 STEPS="calib rules" scripts/long_context_pipeline.sh
KV_PY=$(which python) JUDGE_PY=<vllm env python> GPUS="0 1" SLOTS=2 MINFREE=45000 STEPS="run judge score" scripts/long_context_pipeline.sh
```

Steps: `calib` (rule-1 stats per model and benchmark) -> `rules` (eta tables) ->
`run` (LongBench v2 per model; LongGenBench in 4 interleaved shards per model)
-> `judge` (Qwen3-32B, vLLM) -> `score`. Every job is resumable and leaves
`logs/long_context/done/<job>.ok`.

LongGenBench goes through `scripts/long_context_lgb_pred.py`, which wraps
`eval/LongGenBench/pred.py` and adds three things it lacks:
- **real seeds:** pred.py never seeds torch, and its `--seed` only renames the file. The wrapper samples item i under seed `42 * 1000003 + i`.
- **resume:** pred.py appends with no resume, so a restart duplicates records.
- **shards.**

Scoring (`analysis/score_long_context.py`):
- LongBench v2: accuracy from the harness `judge` field, Overall and Short / Medium / Long, as their `result.py` does.
- LongGenBench: CR from `calculate_completion_rate` (no judge), and CR x Acc = CR x mean(accuracy once / range / periodic) from the Qwen3-32B judge (`eval/LongGenBench/eval.py`).

The judge needs vLLM, which is not in the harness env. It runs in its own env:
vLLM 0.11.0, torch 2.8.0+cu128 (0.10.x wheels are built for CUDA 12.6 and lack sm_120).

## Setup pitfall: refresh the harness copy of `oval/page_basis.py`

`setup.sh` copies `oval/page_basis.py` into the harness as `kvc/patch/oval.py`.
A copy from before commit 59ebb97 builds value records with batched 128x128
`eigh`. At the first decode step of a 64K prompt, that is ~3.6 s per call and
**~7.6 min per LongBench v2 question** (measured), against ~7 s with the
current 32x32 token-domain form. After pulling, re-run `setup.sh` or
`install -Dm644 oval/page_basis.py third_party/FreeKV/accuracy/kvc/patch/oval.py`.

## Files

- `results/long_context/eta_rules/<model>_<bench>.json`: calibration stats (rule-1 sums, loss grid, rank check, prompts used)
- `results/long_context/eta_rules/tables/<model>_<bench>_r1.json`: deployable eta tables
- `results/long_context/raw/`: finished harness outputs (LongBench v2 `.jsonl`; LongGenBench judged `.jsonl.gz`)
- `results/long_context/table.md`: the scored table next to the paper's FreeKV and eta* columns
