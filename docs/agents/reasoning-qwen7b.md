# Brief: full pipeline on long reasoning -- R1-Distill-Qwen-7B

Read [docs/agents/README.md](README.md) first: survey, then confirm the
configuration with the user, then run.

## Status

_Owned by the agent running this brief. Update it at each step: survey
report, confirmed configuration (with date), progress, results, deviations._

- not started

## The task

Run OVAL's full pipeline -- all four pieces together -- on long reasoning for
**`ds-r1-qwen-7b`** (`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`), and report avg@8 / pass@8 per task. Everything you
need to know about the pipeline is in
[docs/5-full-pipeline.md](../5-full-pipeline.md); this brief adds what is
specific to your model. Earlier runs tested each piece on its own; nothing
should be extrapolated from them, which is why this run exists.

## Proposed configuration (confirm with the user before running)

| setting | proposed |
|---|---|
| tasks | MATH50, AIME24, GPQA50c (the harness's reasoning sets) |
| seeds | 42-49 (k = 8), temperature 0.6, top-p 0.95, max_gen 16384 |
| retrieval | `--method spec_ret --page_rep oval --oval_rank 8`, avgSM, spec_ret 2 / 0.9, budget 2048, sink 512, recent 512, page 32, `skip_layer 1`, bf16 records (`OVAL_QUANT=0`) |
| keys | `--oval_recon` (selected pages' keys rebuilt at rank 8) |
| values | `--oval_vrank 16 --vbasis pca` |
| eta | `--eta_table results/eta_rules/vr16/tables/ds-r1-qwen-7b_<task>_r1.json`: rule 1 per (layer, KV head), calibrated with rank-16 values on 6 seed-7 traces per task (4 calibrate), endpoints only (`--ngrid 2`) |
| runner | `MODELS=ds-r1-qwen-7b GPUS="..." SLOTS=2 MINFREE=20000 ./scripts/full_pipeline.sh` |
| cells | 3 tasks x 8 seeds = 24 seed-cells |

## What already exists for this model

| piece(s) | cells in `results/reasoning/ds-r1-qwen-7b-spec_ret/` | summary |
|---|---|---|
| retrieval + keys, fixed eta (exact values) | `_eta{0.25,1.0}_q0_recon1_cr8` (k = 8; MATH50 / GPQA50c at 0.25, AIME24 at 1.0) | `key_rank_decoupled_qwen7b.md` |
| + values PCA rank 16, fixed eta | `_eta{0.25,1.0}_q0_recon1_vr16_vpca` (k = 8) | `value_rank_qwen7b.md` |
| + per-head eta (rule 1), values rank 8 | `_etab<hash>_q0_recon1_vr8_vpca` (k = 8) | `results/eta_rules/table.md` |

The fixed-eta, values-rank-16 cells are the direct baseline: they differ
from the full pipeline only in how eta is chosen. `analysis/eta_rules_table.py
--vrank 16` adds them as the "baseline" row automatically.

## Machine and cost

A machine with 48 GB GPUs: one cell needs ~18-20 GB free, so two cells per
card (`SLOTS=2 MINFREE=20000`). Disk: ~1 GB per calibration trace
(18 traces) plus the model weights.

From the rule-1 cells on the original machine (4 x A6000, 8 concurrent
cells): MATH50 3-5 h per cell, AIME24 and GPQA50c 7-8 h per cell; about
2-3 days for the 24 cells at full occupancy. Calibration: ~25 min of
generation plus a few minutes of measurement per task.

## Open questions for the user (ask them in Step 2)

1. Confirm the configuration above as is.
2. Machine and GPUs to use.

## Notes

- This model has the most complete history; your table is the first
  apples-to-apples test of rule 1 against a hand-picked eta in the full
  configuration.
- Calibration captures from earlier work are not in git; the `capture` step
  regenerates them (seed 7, ~25 min per task). They go to
  `logs/eta_rules/capture/` and are never committed.
- Check the first result filename contains `_etab<hash>_q0_recon1_vr16_vpca`
  before the sweep is far along ([docs/5-full-pipeline.md](../5-full-pipeline.md), step 2).

## Files you own

- `results/eta_rules/vr16/ds-r1-qwen-7b_*.json`, `results/eta_rules/vr16/tables/ds-r1-qwen-7b_*_r1.json`
- `results/reasoning/ds-r1-qwen-7b-spec_ret/*_etab*_q0_recon1_vr16_vpca-*.jsonl` (complete seeds only)
- `results/eta_rules/vr16/table_ds-r1-qwen-7b.md`
- the Status section of this file
