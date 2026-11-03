# Brief: full pipeline on long reasoning -- R1-Distill-Llama-8B

Read [docs/agents/README.md](README.md) first: survey, then confirm the
configuration with the user, then run.

## Status

_Owned by the agent running this brief. Update it at each step: survey
report, confirmed configuration (with date), progress, results, deviations._

- not started

## The task

Run OVAL's full pipeline -- all four pieces together -- on long reasoning for
**`ds-r1-llama-8b`** (`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`), and report avg@8 / pass@8 per task. Everything you
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
| eta | `--eta_table results/eta_rules/vr16/tables/ds-r1-llama-8b_<task>_r1.json`: rule 1 per (layer, KV head), calibrated with rank-16 values on 6 seed-7 traces per task (4 calibrate), endpoints only (`--ngrid 2`) |
| runner | `MODELS=ds-r1-llama-8b GPUS="..." SLOTS=2 MINFREE=20000 ./scripts/full_pipeline.sh` |
| cells | 3 tasks x 8 seeds = 24 seed-cells |

## What already exists for this model

| piece(s) | cells in `results/reasoning/ds-r1-llama-8b-spec_ret/` | summary |
|---|---|---|
| retrieval + keys, eta sweep (exact values) | `_eta{0.0,0.25,0.5,0.75,1.0}_q0_recon1` (k = 8, all 120 cells) | `eta_sweep_llama8b.md` |
| + values PCA rank 8 + per-head eta (rule 1) | `_etab<hash>_q0_recon1_vr8_vpca` (k = 8) | `results/eta_rules/table.md` |

There is **no fixed-eta cell with compressed values** for this model, so the
full pipeline has no baseline that differs from it only in how eta is chosen
(see open question 2).

## Machine and cost

A machine with 48 GB GPUs: one cell needs ~18-20 GB free, so two cells per
card (`SLOTS=2 MINFREE=20000`). Disk: ~1 GB per calibration trace
(18 traces) plus the model weights.

Measured on the original machine (4 x 48 GB, 8 concurrent cells): MATH50
2.9 h, AIME24 5.3 h, GPQA50c 5.4 h per cell; about 2 days for 24 cells.
Calibration: ~25 min of generation plus a few minutes of measurement per
task.

## Open questions for the user (ask them in Step 2)

1. Confirm the configuration above as is.
2. A fixed-eta baseline with values rank 16 (24 more cells, ~2 days)? If
   yes, at which eta per task: the best shared eta of this model's sweep
   (MATH50 0.75, AIME24 0.75, GPQA50c 0.25) or the per-task values used for
   qwen-7b (0.25 / 1.0 / 0.25)?
   Command, after the full pipeline (never interleaved with it):
   `MODEL=ds-r1-llama-8b VRANKS=16 ETAMAP="MATH50:<e> AIME24:<e> GPQA50c:<e>" scripts/value_rank_sweep.sh`;
   then add the model to `BASE_ETA` in `analysis/eta_rules_table.py` so the
   baseline row appears in your table.
3. Machine and GPUs to use.

## Notes

- Rule 1 puts most of this model's heads at eta = 1 (at value rank 8: 65%
  on MATH50); expect a similar table at rank 16 and say if it is not.
- Calibration captures from earlier work are not in git; the `capture` step
  regenerates them (seed 7, ~25 min per task). They go to
  `logs/eta_rules/capture/` and are never committed.
- Check the first result filename contains `_etab<hash>_q0_recon1_vr16_vpca`
  before the sweep is far along ([docs/5-full-pipeline.md](../5-full-pipeline.md), step 2).

## Files you own

- `results/eta_rules/vr16/ds-r1-llama-8b_*.json`, `results/eta_rules/vr16/tables/ds-r1-llama-8b_*_r1.json`
- `results/reasoning/ds-r1-llama-8b-spec_ret/*_etab*_q0_recon1_vr16_vpca-*.jsonl` (complete seeds only)
- `results/eta_rules/vr16/table_ds-r1-llama-8b.md`
- the Status section of this file
