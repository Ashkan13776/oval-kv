# Brief: full pipeline on long reasoning -- R1-Distill-Qwen-7B

Read [docs/agents/README.md](README.md) first: survey, then confirm the
configuration with the user, then run.

## Status

_Owned by the agent running this brief. Update it at each step: survey
report, confirmed configuration (with date), progress, results, deviations._

### 2026-11-03 -- survey done, configuration proposed, NOT yet confirmed, no GPU job started

**The four pieces, and how this run combines them.** OVAL keeps one rank-r
record per KV page per KV head (`mu`, basis `B`, coefficients `C`), built from
the eigenvectors of `M_eta = (1-eta) XtX/tr + eta XtYGYtX/tr`, and uses it four
ways. (1) *Retrieval*: score every page by the logsumexp of its per-token
logits reconstructed in the basis, combine over the GQA group with avgSM, take
the top pages within the 2048-token budget; FreeKV's spec_ret and its cosine
correction are untouched (`--method spec_ret --page_rep oval --oval_rank 8`).
(2) *Key compression*: attend the winning pages with keys rebuilt from those
same records, `mu + C B^T`, so the records become the only copy of the paged
keys (`--oval_recon`, tag `_recon1`). (3) *Value compression*: rebuild the
winning pages' values from a separate plain-PCA record of the centred page
values (`--oval_vrank 16 --vbasis pca`, tag `_vr16_vpca`). Sink and recent
tokens stay exact for both halves. (4) *Per-head eta*: instead of one swept
eta, rule 1 picks eta per (layer, KV head) in closed form from the output
errors at the two endpoints only -- `eta* = clip((a-c)/(a+b-2c), 0, 1)` with
`a=sum|D0|^2, b=sum|D1|^2, c=sum D0.D1` -- fitted offline on seed-7 calibration
traces (`--eta_table <json>`, tag `_etab<hash>`, 8 hex of sha1 over the
4-decimal table). The pieces compose because eta only enters the record build:
rule 1's loss is measured with the same reconstruction the pipeline uses, which
is why the table must be recalibrated at value rank 16 rather than reused from
`results/eta_rules/tables/` (rank 8). Filenames carry each piece
(`oval_r8` / `etab<hash>` / `q0` / `recon1` / `vr16_vpca`), so the name alone
says which were on; `analysis/eta_rules_queue.py` matches that tag exactly, so
eta-table cells, fixed-eta cells and vr8/vr16 cells never count as each other.

**What is already on disk for ds-r1-qwen-7b.** All three reference families are
complete at k = 8: keys-only key-rank sweep `_eta{0.25,1.0}_q0_recon1_cr{1,4,8,16,32}`,
the value-rank sweep `_eta{0.25,1.0}_q0_recon1_vr{1,4,8,16,32}_vpca` (its
`vr16` arm is this run's direct baseline, 24 complete seed-cells, 24 MB), and
rule 1 at value rank 8 `_etab{5dce7a47,71b9668a,0e97922c}_q0_recon1_vr8_vpca`
with its stats and tables in `results/eta_rules/`. `results/eta_rules/vr16/`
does not exist yet; it is mine to create. CPU checks run here: both unit test
suites pass (`test_eta_rules.py` "all checks passed", `test_page_basis.py`
checks 1-11), and a queue dry run reports 24/24 complete at vr8 and 3 missing
tables at vr16 -- i.e. exactly the 24 seed-cells this run must produce.

**Blocker: disk.** `/` is 100% full with **3.0 GB free** (3.5 TB, shared).
The run needs roughly 8.3 GB of calibration captures (~420 MB per trace for
this model: 28 layers x 4 KV heads x ~6.3K tokens of post-RoPE K and V, plus
3 x 96 query windows; 6 traces x 3 tasks, plus ~0.7 GB of o_proj weights),
~0.75 GB for the FreeKV checkout `setup.sh` clones, and ~25 MB of results.
Per-task cleanup of captures after `stats` would cut the peak to ~3.2 GB, still
above what is free. Nothing can start until space is freed; see the proposal.

**Other findings.**
- `setup.sh` has not been run in this clone (`third_party/` absent). The old
  predecessor tree `/home/shahbaa/kv-compression/vendor/FreeKV` is at the same
  pinned commit but carries unrelated local modifications, so a fresh clone is
  the right move, not a reuse.
- The environment on this machine is conda `freekv`: torch 2.5.1+cu121,
  transformers 4.45.2 (FreeKV's pin), flash-attn 2.7.4.post1, flashinfer 0.2.4,
  driver 535 / CUDA 12.2. `docs/3-value-compression.md` and the top-level
  README describe the env as `ovalkv` with torch 2.8.0+cu128 and flash-attn
  2.8.3. No env matching the docs exists here. The model weights
  (`DeepSeek-R1-Distill-Qwen-7B`, 15 GB) are already in the HF cache.
- `results/reasoning/README.md`'s worked filename example is
  `...oval_r8_etab71b9668a_q0_recon1_vr16_vpca-...`, but `71b9668a` is the hash
  of the **rank-8**-calibrated AIME24 table (`results/eta_rules/tables/`). That
  pairing is the one `docs/5-full-pipeline.md` explicitly forbids ("Do not use
  the rank-8 tables with `--oval_vrank 16`"), and it is what step 2's check
  tells an agent to look for. The example should use a placeholder hash.
- Cost arithmetic in this brief does not close: at the quoted per-cell times
  (MATH50 3-5 h, AIME24/GPQA50c 7-8 h), the 24 cells are ~152 cell-hours, which
  at the stated full occupancy (4 GPUs x 2 slots = 8 concurrent) is ~19 h, not
  the "2-3 days" the brief gives. The 2-3 days is consistent with the machine
  being shared with another user's jobs, as `docs/3-value-compression.md`
  records. All four A6000s are idle right now, so I plan on ~20-24 h.

### 2026-11-03 -- configuration confirmed by the user; calibration done, sweep running

Confirmed as proposed: keys rank 8 for scoring and reconstruction, values PCA
rank 16, eta per (layer, KV head) from rule 1 recalibrated at rank-16 values,
everything else as in the earlier long-reasoning runs. Machine: this one,
4 x RTX A6000 48GB, `GPUS="0 1 2 3" SLOTS=2 MINFREE=20000`, calibration on
GPU 0. Env: conda `freekv` (torch 2.5.1+cu121, transformers 4.45.2,
flash-attn 2.7.4.post1, flashinfer 0.2.4) -- NOT the torch 2.8/cu128 env the
docs describe; no such env exists on this machine. The user authorised freeing
disk from caches, which unblocked the run.

**Calibration (complete).** `setup.sh` applied `patches/freekv.patch` cleanly;
both unit suites pass. Captures and stats were run one task at a time
(`STEPS="capture stats" DATASETS=<task>`, a documented `full_pipeline.sh`
mode) and each task's ~2.4 GB of captures deleted once its stats file landed,
to stay inside the free disk; this changes nothing in the results. Capture
~19 min and stats ~9 min per task. All three tasks: **degenerate 0/108**
(27 layers >= 1 x 4 KV heads; layer 0 recorded but dense under `skip_layer 1`).

| task | eta mean / median | at 0 | at 1 | vr16 table hash | (vr8 hash) |
|---|---|---|---|---|---|
| MATH50 | 0.463 / 0.544 | 0.33 | 0.11 | `0c99a26a` | `5dce7a47` |
| AIME24 | 0.389 / 0.327 | 0.41 | 0.11 | `fa1b4e40` | `71b9668a` |
| GPQA50c | 0.392 / 0.367 | 0.39 | 0.10 | `86e2340d` | `0e97922c` |

All three hashes differ from the rank-8 tables, confirming the recalibration
reached the harness and that the forbidden rank-8-table/rank-16-values
combination is not in use. Recalibration moves eta materially (AIME24 median
0.47 -> 0.33), so it was not a formality.

**Sweep (running, started 2026-11-03 13:38).** 24 seed-cells queued, 8
concurrent. First output filename verified to carry every piece:
`MATH50-...-avgSM-oval_r8_etab0c99a26a_q0_recon1_vr16_vpca-pQ2-llb0-pQcs0.9-skl1-t0.6-topp0.95-Mg16384.jsonl`.

**Deviation / fix worth recording.** The first sweep launch died instantly in
every cell with `RuntimeError: Ninja is required to load C++ extensions`:
flashinfer's JIT needs `ninja` and `nvcc` on `PATH`, and I had launched with an
absolute `KV_PY` without the `export PATH=$(dirname $KV_PY):$PATH
CUDA_HOME=$(dirname $(dirname $KV_PY))` line from `docs/5-full-pipeline.md`
step 0. Both binaries are present in the env. Stopped the supervisor before it
spent its three strikes, no partial files were written, pre-warmed the JIT
once, and relaunched with the documented exports. Nothing about the
configuration changed.

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
