# Brief: full pipeline on long reasoning -- R1-Distill-Qwen-14B

Read [docs/agents/README.md](README.md) first: survey, then confirm the
configuration with the user, then run.

## Status

_Owned by the agent running this brief. Update it at each step: survey
report, confirmed configuration (with date), progress, results, deviations._

- not started

## The task

Run OVAL's full pipeline -- all four pieces together -- on long reasoning for
**`ds-r1-qwen-14b`** (`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`), and report avg@8 / pass@8 per task. Everything you
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
| eta | `--eta_table results/eta_rules/vr16/tables/ds-r1-qwen-14b_<task>_r1.json`: rule 1 per (layer, KV head), calibrated with rank-16 values on 6 seed-7 traces per task (4 calibrate), endpoints only (`--ngrid 2`) |
| runner | `MODELS=ds-r1-qwen-14b GPUS="..." SLOTS=1 MINFREE=40000 ./scripts/full_pipeline.sh` |
| cells | 3 tasks x 8 seeds = 24 seed-cells |

## What already exists for this model

| piece(s) | cells in `results/reasoning/ds-r1-qwen-14b-spec_ret/` | summary |
|---|---|---|
| retrieval + keys, eta sweep (exact values) | `_eta{0.0,0.25,0.5,0.75,1.0}_q0_recon1` (k = 8, all 120 cells) | `eta_sweep_qwen14b.md` |

**Nothing else has been run on this model**: value compression and rule 1
were developed on 7B/8B only, and there are no calibration captures.

## Machine and cost

One cell needs ~40 GB free (`MINFREE=40000`): one cell per 48 GB card
(`SLOTS=1`), two per 80 GB+ card. Disk: ~1.5 GB per calibration trace
(18 traces) plus ~30 GB of weights. On Blackwell set
`TORCH_CUDA_ARCH_LIST=12.0` and read [docs/harness/freekv-port.md](../harness/freekv-port.md).

Measured on machine C (2 x RTX PRO 6000 Blackwell, 97 GB): MATH50 4.2 h,
AIME24 7.6 h, GPQA50c 7.7 h per cell; with two cells at a time about 4 days
for 24 cells. Calibration: generation with the 14B model is slower than
for 7B/8B (estimate ~45 min per task).

## Open questions for the user (ask them in Step 2)

1. Confirm the configuration above as is. In particular: value rank 16
   and rule 1 were validated on 7B/8B only -- go straight to the full
   pipeline, or first run a short check (e.g. one task, k = 2)?
2. A fixed-eta baseline with values rank 16 (24 more cells)? If yes, at
   which eta per task: this model's best shared eta (MATH50 0.25, AIME24
   1.0, GPQA50c 0.0) or the qwen-7b values (0.25 / 1.0 / 0.25)?
   Command, after the full pipeline (never interleaved with it):
   `MODEL=ds-r1-qwen-14b VRANKS=16 ETAMAP="MATH50:<e> AIME24:<e> GPQA50c:<e>" scripts/value_rank_sweep.sh`;
   then add the model to `BASE_ETA` in `analysis/eta_rules_table.py` so the
   baseline row appears in your table.
3. Machine and GPUs to use (`SLOTS`, `MINFREE` depend on card size).

## Notes

- The calibration script knows this model (`MODEL2PATH`); the per-head eta
  table has 48 layers x 8 KV heads.
- Calibration captures from earlier work are not in git; the `capture` step
  regenerates them (seed 7, ~25 min per task). They go to
  `logs/eta_rules/capture/` and are never committed.
- Check the first result filename contains `_etab<hash>_q0_recon1_vr16_vpca`
  before the sweep is far along ([docs/5-full-pipeline.md](../5-full-pipeline.md), step 2).

## Files you own

- `results/eta_rules/vr16/ds-r1-qwen-14b_*.json`, `results/eta_rules/vr16/tables/ds-r1-qwen-14b_*_r1.json`
- `results/reasoning/ds-r1-qwen-14b-spec_ret/*_etab*_q0_recon1_vr16_vpca-*.jsonl` (complete seeds only)
- `results/eta_rules/vr16/table_ds-r1-qwen-14b.md`
- the Status section of this file
