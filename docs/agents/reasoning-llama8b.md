# Brief: full pipeline on long reasoning -- R1-Distill-Llama-8B

Read [docs/agents/README.md](README.md) first: survey, then confirm the
configuration with the user, then run.

## Status

_Owned by the agent running this brief. Update it at each step: survey
report, confirmed configuration (with date), progress, results, deviations._

### 2026-11-07 03:40: vr32 complete; summary of value ranks 8 / 16 / 32

All 24 of 24 vr32 seed-cells are complete with no failures. The scored table is
`results/eta_rules/vr32/table_ds-r1-llama-8b.md`.

avg@8 / pass@8, k = 8. In every row, rule 1 is recalibrated at that value rank.

| task | vr8 | vr16 | vr32 (lossless) | vr32 - vr16 (paired 95% CI) | vr32 - vr8 |
|---|---|---|---|---|---|
| MATH50 | 66.25 / 76.00 | 65.75 / 76.00 | 66.25 / 76.00 | +0.50 [-2.00, +3.50] | 0.00 [-3.25, +3.50] |
| AIME24 | 48.33 / 83.33 | 47.08 / 76.67 | 45.00 / 80.00 | -2.08 [-9.58, +5.00] | -3.33 [-10.00, +3.75] |
| GPQA50c | 37.50 / 68.00 | 36.50 / 72.00 | 35.50 / 72.00 | -1.00 [-5.75, +3.75] | -2.00 [-6.50, +2.75] |

**Conclusion.** For R1-Distill-Llama-8B at budget 2048, value rank has no
measurable effect on accuracy. Lossless values (rank 31) are no better than
PCA rank 8 on any task, and every CI includes 0. Rank-8 values are therefore
enough, and the accuracy gap to dense attention, if there is one, comes from
retrieval or key compression rather than from values.

AIME24 and GPQA50c show a small, non-significant downward drift as the value
rank rises. With n = 30 and 50 problems, this is consistent with noise.

### 2026-11-06 16:30: vr32 MATH50 and AIME24 complete

16 of 24 seed-cells were complete. The other user's job left the GPUs at about
14:50 on Nov 5, and 8 cells ran at once from then on.

### 2026-11-05 12:00: vr32 calibration done, sweep running

Rule 1 has been recalibrated at value rank 32. The harness clamps that to 31,
so values are lossless. Outputs are in `results/eta_rules/vr32/`. All three
tables have 0/248 degenerate entries.

| task | hash | eta mean / median | at0 / at1 | vr16 mean / at1 |
|---|---|---|---|---|
| MATH50 | `67e36f28` | 0.832 / 1.000 | 0.05 / 0.56 | 0.841 / 0.60 |
| AIME24 | `f82bc63c` | 0.694 / 0.815 | 0.10 / 0.35 | 0.710 / 0.36 |
| GPQA50c | `130f512f` | 0.758 / 0.877 | 0.08 / 0.42 | 0.770 / 0.44 |

As the value rank goes up, the tables keep moving slightly toward lower eta.
From rank 16 to rank 32 the shift is smaller than from rank 8 to rank 16.

The stats step took 1.8 / 3.5 / 3.1 h per task, against about 20 min at rank
16. From about 09:30 another user's training job has also been on all four
GPUs, using about 11 GB on each. The queue has 24 seed-cells. The sweep
started at 11:52, and the first file has the correct tags:
`MATH50-...-oval_r8_etab67e36f28_q0_recon1_vr32_vpca-...-Mg16384.jsonl`.
With the other job's memory on each GPU, the `MINFREE=20000` guard allows
only one cell per GPU, so 4 cells run at once instead of 8. The sweep will
take roughly twice as long until that job ends.

### 2026-11-05 03:30: vr16 complete; vr32 calibration started

All 24/24 vr16 seed-cells are complete, with no failures since the 14:29 relaunch.
The scored table is `results/eta_rules/vr16/table_ds-r1-llama-8b.md`. GPQA50c
cells took about 13-16 h with 8 cells running together.

| task | k | n | avg@8 | pass@8 | vs rule 1 at value rank 8 (avg@8, paired 95% CI) |
|---|---|---|---|---|---|
| MATH50 | 8 | 50 | 65.75 | 76.00 | -0.50 [-2.50, +1.50] (66.25 / 76.00) |
| AIME24 | 8 | 30 | 47.08 | 76.67 | -1.25 [-7.92, +5.00] (48.33 / 83.33) |
| GPQA50c | 8 | 50 | 36.50 | 72.00 | -1.00 [-5.75, +3.75] (37.50 / 68.00) |

Values at rank 16 give no measurable gain over rank 8 on any task. Every
difference is within noise. The value error at rank 8 is not what limits
accuracy at this budget.

The vr32 pipeline started automatically at 03:30. It recalibrates rule 1 at
value rank 32 (clamped to 31) from the same captures, so the capture step is
skipped.

### 2026-11-04: follow-up confirmed by the user — value rank 32

After the vr16 run, the full pipeline is re-run with `VRANK=32`. The harness
clamps 32 to 31, so values are lossless. Rule 1 is recalibrated at value rank
32 from the same seed-7 captures. Everything else is unchanged. It starts
automatically when the vr16 pipeline ends cleanly (`logs/chain_vr32.sh`).
Outputs: `results/eta_rules/vr32/`, `*_etab<hash>_q0_recon1_vr32_vpca-*.jsonl`,
`results/eta_rules/vr32/table_ds-r1-llama-8b.md`.

### 2026-11-04 15:00: MATH50 and AIME24 complete, GPQA50c running

16/24 seed-cells complete, no failures or restarts since the 14:29 relaunch.
With 8 cells running together, cells took about 7.5 h (MATH50) and 10.5-14 h
(AIME24), well above the brief's 2.9 / 5.3 h.

| task | k | n | avg@8 | pass@8 | vs rule 1 at value rank 8 (avg@8, paired 95% CI) |
|---|---|---|---|---|---|
| MATH50 | 8 | 50 | 65.75 | 76.00 | -0.50 [-2.50, +1.50] (66.25 / 76.00) |
| AIME24 | 8 | 30 | 47.08 | 76.67 | -1.25 [-7.92, +5.00] (48.33 / 83.33) |
| GPQA50c | running | | | | |

### 2026-11-03 14:30: calibration done, sweep running

**Rule 1 at value rank 16** (`results/eta_rules/vr16/`, endpoints only, 6 seed-7
traces per task, 4 calibrate). Layers >= 1, 248 (layer, KV head) entries, 0
degenerate in every table:

| task | hash | eta mean / median | at0 / at1 | vs vr8 table: corr, mean abs diff | vr8 mean / at1 | test-split error L(0) / L(1), vr16 (vr8) |
|---|---|---|---|---|---|---|
| MATH50 | `6fbcdc9f` | 0.841 / 1.000 | 0.05 / 0.60 | 0.969, 0.031 | 0.869 / 0.65 | 1.90% / 1.42% (2.13 / 1.64) |
| AIME24 | `85d84e24` | 0.710 / 0.839 | 0.09 / 0.36 | 0.963, 0.068 | 0.774 / 0.47 | 1.38% / 1.28% (1.60 / 1.47) |
| GPQA50c | `cda5c9d0` | 0.770 / 0.911 | 0.08 / 0.44 | 0.969, 0.048 | 0.815 / 0.52 | 2.22% / 1.99% (2.60 / 2.32) |

The tables are close to the rank-8 ones, as the brief expected, but shifted
consistently toward lower eta. The share of heads at eta = 1 falls by 5-11
points. The output error is 11-15% lower at both endpoints, since the values
are better. Calibration wall time: about 21 min of capture plus about 20 min
of stats per task. The stats step is not "minutes" as
`docs/5-full-pipeline.md` says.

**Sweep.** Started 14:10. Every cell failed within about 12 s at the first
launch: flashinfer's JIT build of the `norm` op picked up the base conda's
`x86_64-conda-linux-gnu-cc`, inherited through `CC`/`CXX`, which fails on
`::timespec_get`. I stopped the sweep before any result was written, set
`CC=/usr/bin/gcc CXX=/usr/bin/g++` (gcc 11.4), pre-built the `norm` and `rope`
JIT ops (the only flashinfer ops the harness uses), and relaunched with
`STEPS="sweep score"` at 14:29. The first file is
`MATH50-...-oval_r8_etab6fbcdc9f_q0_recon1_vr16_vpca-pQ2-llb0-pQcs0.9-skl1-t0.6-topp0.95-Mg16384.jsonl`,
which has the correct tags. Progress: 0/24 seed-cells complete.

### 2026-11-03: configuration confirmed by the user

- The configuration in "Proposed configuration" below, unchanged.
  Keys rank 8 for scoring and reconstruction. Values PCA rank 16. Rule-1 eta
  per (layer, KV head), recalibrated at value rank 16 from the endpoints only
  (`--ngrid 2`, no regret stage).
- **No fixed-eta baseline**; per the user it exists elsewhere. Open question 2
  is answered "no".
- Machine: 4 x RTX A6000 48 GB, `GPUS="0 1 2 3" SLOTS=2 MINFREE=20000`,
  calibration on GPU 0. Disk freed to 104 GB at the user's request.
- Env `ovalkv`: python 3.11, torch 2.8.0+cu128, transformers 4.45.2,
  flash-attn 2.8.3, flashinfer 0.2.4 (JIT), CUDA 12.8 nvcc in the env,
  `TORCH_CUDA_ARCH_LIST=8.6`. `setup.sh` ran clean. Both unit-test suites pass.
- Command: `MODELS=ds-r1-llama-8b GPUS="0 1 2 3" SLOTS=2 MINFREE=20000 ./scripts/full_pipeline.sh`.

### 2026-11-03: survey (step 1)

**The four pieces.** OVAL stores a rank-r record (`mu`, basis `B`, coefficients
`C`) of every 32-token page per KV head. Its basis comes from
`M_eta = (1-eta) key PCA + eta output term (W_O metric)`.
1. *Retrieval*: score each page by the logsumexp of its rank-8 reconstructed
   logits, combine over GQA with avgSM, and attend the top 2048 tokens (plus
   sink 512 and recent 512, which the harness adds on top). FreeKV spec_ret
   stays unchanged. Flags: `--method spec_ret --page_rep oval --oval_rank 8 --eta E`.
   Tags: `oval_r8_eta<E>`, `q0` (bf16 records, `OVAL_QUANT=0`).
2. *Key compression*: the selected pages' keys are rebuilt from the same
   rank-8 records, so the records are the key store. Flag: `--oval_recon`,
   tag `_recon1`. Evidence (qwen-7b): rank 8 is no different from exact keys.
3. *Value compression*: the selected pages' values are rebuilt from per-page
   plain-PCA records. Flags: `--oval_vrank 16 --vbasis pca`, tag `_vr16_vpca`.
   Evidence (qwen-7b): ranks 16 and 8 are within noise of lossless.
4. *Per-head eta*: rule 1, `eta* = clip((a-c)/(a+b-2c))`, computed from the
   output errors at eta = 0 and 1 per (layer, KV head). The errors are
   measured on seed-7 calibration traces (4 calibrate, 2 test), with the
   method's own sparse attention, including the value reconstruction. Flag:
   `--eta_table <json>`, tag `etab<8-hex content hash>` in place of `eta<E>`.
   The full pipeline recalibrates at value rank 16.

The full pipeline turns on all four:
`..._oval_r8_etab<hash>_q0_recon1_vr16_vpca-pQ2-llb0-pQcs0.9-skl1-t0.6-topp0.95-Mg16384[-seedS].jsonl`.
`scripts/full_pipeline.sh` runs capture, then stats (`--vrank 16 --ngrid 2`),
then rules (writing `results/eta_rules/vr16/tables`), then the sweep
(`supervise_eta_tables.sh` and `eta_table_sweep.sh`, `jobs_vr16.txt`), then
score.

**On disk for ds-r1-llama-8b.** Eta sweep with keys rebuilt and exact values:
`_eta{0.0..1.0}_q0_recon1`, 15 cells x 8 seeds, all complete. Best shared eta
is 0.75 / 0.75 / 0.25. Rule 1 at value rank 8: `etab{ee278def,735fb9c7,0649aee7}_..._vr8_vpca`,
8/8 seeds each. Its stats used `ngrid 21` with a regret block. Nothing exists
at vr16, and `results/eta_rules/vr16/` does not exist. Calibration captures
are not in git.

**CPU checks on this machine.** `test_eta_rules.py` printed "all checks passed".
`test_page_basis.py` passed checks 1-11. A queue dry-run against the rank-8
tables reports 24/24 complete, as it should. Against the empty vr16 tables it
reports 3 tables missing.

**Inconsistent or unclear (not worked around):**
- Per-cell time: this brief and `harness/parallel-runs.md` give 2.9 / 5.3 / 5.4 h.
  `5-full-pipeline.md` gives 3-5 h for MATH50 and 7-8 h for AIME24 and GPQA50c.
- The full pipeline calibrates with `--ngrid 2`. The rank-8 llama/qwen stats
  used `ngrid 21` with the `regret` stage, so the vr16 table will have no
  regret column. Rule 1's table is the same either way, since it uses only
  the endpoints.
- `docs/2-key-compression.md`, `results/reasoning/README.md` (`cr<R>`) and
  `scripts/oval_recon_rank_sweep.sh` use `--oval_recon_rank`, but
  `patches/freekv.patch` defines no such flag. That script would fail at
  argparse. It does not affect this run (reconstruction rank = scoring rank = 8).
- The queue and supervisor treat missing tables as "0 seed-cells to run", so
  the supervisor would log ALL COMPLETE. `full_pipeline.sh` does not pass
  `--require_all`. The step-2 check catches this.
- In `supervise_eta_tables.sh`, the stall restart uses `pkill -f "^$KV_PY -u -m eval.reasoning.pred .*--eta_table"`.
  That pattern matches across the whole machine, so it would also kill another
  agent's eta-table cells running on the same machine with the same interpreter.
  The same applies to the `pgrep` "ours" count. Two checkouts are safe only on
  separate machines or separate interpreters.
- A fixed-eta baseline at vr16 for llama needs a `BASE_ETA` entry in the shared
  `analysis/eta_rules_table.py`. Its cells (`_eta<E>_q0_recon1_vr16_vpca`) are
  not in "Files you own".
- `docs/5-full-pipeline.md` describes the stats stage's cost as "minutes".
  This machine has no measurement yet.

**Machine (this one).** 4 x RTX A6000 48 GB, all idle. 128 CPU, 251 GB RAM.
**Root disk: 16 GB free (100% used). Too little for the run** (see the proposal).
No env with the harness pins (transformers 4.45.2, flash-attn, flashinfer 0.2.4).
The model weights are not cached.

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
