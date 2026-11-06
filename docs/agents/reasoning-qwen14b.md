# Brief: full pipeline on long reasoning -- R1-Distill-Qwen-14B

Read [docs/agents/README.md](README.md) first: survey, then confirm the
configuration with the user, then run.

## Status

_Owned by the agent running this brief. Update it at each step: survey
report, confirmed configuration (with date), progress, results, deviations._

**2026-11-06 04:40: sweep running, 8/24 seed-cells complete. MATH50 is finished, avg@8 = 69.25** (seeds 42-49: 68 / 72 / 68 / 66 / 70 / 74 / 68 / 68). That's in line with the eta-sweep avg@8 of 67.0-69.3; the final avg@8 / pass@8 come from the score step. AIME24 seeds 42 and 43 are running, on GPU 1 since 04:18 and on GPU 0 since 04:36. MATH50 cells took 3.0-3.3 h on either GPU. The finish ETA will be revised once the first AIME24 cell's time is known.

### Progress

| step | result |
|---|---|
| capture | 18 seed-7 traces, 6144 tokens each; ~200 s per trace on an unshared GPU |
| stats (`--vrank 16 --ngrid 2`) | ~28 min per task at 14B (~95 s per query window) |
| rules | MATH50 `2851b0ec`, AIME24 `3b5da3dd`, GPQA50c `8db7317a`; **degenerate 0/376 each** |
| sweep | started 15:13; filenames carry `_etab<hash>_q0_recon1_vr16_vpca` (checked) |

Rule 1 eta* (layers 1-47 x 8 KV heads):

| task | mean | median | at 0 | at 1 |
|---|---|---|---|---|
| MATH50 | 0.695 | 0.792 | 8% | 26% |
| AIME24 | 0.589 | 0.617 | 10% | 16% |
| GPQA50c | 0.629 | 0.673 | 9% | 19% |

### Deviations and incidents (none change the confirmed configuration)

- **GPU 1 is shared** with another user's jobs (from 12:53: a `lerobot-train` run, later a policy server and an Isaac Sim eval). AIME24 capture ran ~2x slower and its stats ~10x slower there, so AIME24 stats was killed and re-run on GPU 0. The stats file is written only at the end, so nothing partial was kept. Sweep cells on GPU 1 may run slower while those jobs are active.
- **`CUDA_HOME`:** following docs/5's setup (`CUDA_HOME` = the harness env) broke flashinfer's JIT build on machine C, because the `freekv` env has no `nvcc`. The first two cells died at startup (exit 1, no records). The fix is `CUDA_HOME` = the `pyg310` env (CUDA 12.8 nvcc, as `docs/harness/freekv-port.md` says), with `TORCH_CUDA_ARCH_LIST=12.0`.
- **Supervisor lock:** after the supervisor was killed, an orphaned `sleep` child kept `.supervisor_eta_vr16.lock`. The relaunched supervisor exited 0 ("another supervisor holds the lock"), and `full_pipeline.sh` then went on to score and wrote an empty table, which was deleted. This is a shared-script gotcha: `full_pipeline.sh` treats "lock held" as success.

### Confirmed configuration (2026-11-05)

- ds-r1-qwen-14b; MATH50, AIME24, GPQA50c; seeds 42-49 (24 seed-cells); T 0.6, top-p 0.95, max_gen 16384.
- `--method spec_ret --page_rep oval --oval_rank 8 --oval_recon --oval_vrank 16 --vbasis pca --eta_table results/eta_rules/vr16/tables/ds-r1-qwen-14b_<task>_r1.json`, avgSM, spec_ret 2 / 0.9, budget 2048, sink 512, recent 512, page 32, skl1, `OVAL_QUANT=0`.
- Rule 1 calibrated at value rank 16 (`--vrank 16 --ngrid 2`), 6 seed-7 traces per task, 4 of them calibrate.
- Machine C (`mint00`, 2 x RTX PRO 6000 Blackwell 97 GB), env `freekv` (torch 2.8.0+cu128), `TORCH_CUDA_ARCH_LIST=12.0`. Both GPUs used: calibration split across GPU 0 (MATH50, GPQA50c) and GPU 1 (AIME24), then `full_pipeline.sh` with `GPUS="0 1" SLOTS=1 MINFREE=40000` for rules, sweep and score.
- No separate pre-check: MATH50 seeds 42/43 run first and are checked (tag, accuracy against the eta sweep).
- **No fixed-eta baseline** (the user's decision).
- Harness refreshed before the run: the stale `third_party/FreeKV` tree was stashed and `./setup.sh` re-applied the current patch.

### Survey report

OVAL keeps a rank-r record (mean, basis, coefficients) of every 32-token KV page per KV head and uses it four ways:

| # | piece | what it does | flag(s) | filename tag | evidence alone |
|---|---|---|---|---|---|
| 1 | retrieval | score each page by the logsumexp of its logits rebuilt in a rank-8 basis of `M_eta` (key PCA at eta 0, output-aware at eta 1); FreeKV's avgSM / top-k / spec_ret do the rest | `--method spec_ret --page_rep oval --oval_rank 8 --eta E` | `oval_r8_eta<E>` | eta sweeps on 3 models; no single eta wins |
| 2 | key compression | selected pages' keys rebuilt from the same rank-8 records (`mu + C B^T`); sink/recent exact | `--oval_recon` | `_recon1` (`cr<R>` if recon rank differs) | qwen-7b k=8: rank 8 = exact keys |
| 3 | value compression | selected pages' values rebuilt from per-page plain PCA records | `--oval_vrank 16 --vbasis pca` | `_vr16_vpca` | qwen-7b k=8: rank 16 and 8 within noise of lossless |
| 4 | per-head eta | rule 1: from the output errors at eta 0 and 1 on 4 seed-7 calibration traces, `eta* = clip((a-c)/(a+b-2c))` per (layer, KV head) | `--eta_table <json>` | `_etab<8-hex hash>` replacing `_eta<E>` | qwen-7b/llama-8b at value rank 8: >= fixed eta, n.s. |

Full pipeline = all four in one harness call; rule 1 is recalibrated with rank-16 values because its loss includes the value reconstruction. `scripts/full_pipeline.sh`: capture -> stats (`--vrank 16 --ngrid 2`) -> rules -> supervised sweep (`supervise_eta_tables.sh` / `eta_table_sweep.sh`, queue from `eta_rules_queue.py`) -> score. Expected cell name: `<task>-...-oval_r8_etab<hash>_q0_recon1_vr16_vpca-pQ2-...-seed<S>.jsonl`.

**On disk for ds-r1-qwen-14b:** the 120-cell eta sweep (`_eta{0,0.25,0.5,0.75,1.0}_q0_recon1`, exact values). No value-compression cells, no rule-1 stats/tables, no calibration captures. Weights are cached on machine C (28 GB). CPU checks: `test_eta_rules.py` and `test_page_basis.py` pass; queue dry-run reports the 3 qwen-14b tables missing, as expected.

**Inconsistencies / things to fix before running:**

1. **The installed harness on machine C is stale.** `third_party/FreeKV` carries an older patch: `eval/util.py` has no `--eta_table` / `--oval_vrank`. The current `patches/freekv.patch` applies cleanly to the pinned commit (checked in a scratch worktree). Needs `./setup.sh` re-run on a reset tree; the only local-only edit there (o1 -> reasoning dataset path in `pred.py`) is redundant with the symlink `setup.sh` creates.
2. **Slots at 14B conflict.** `docs/harness/parallel-runs.md` says run one 14B cell at a time on machine C (`GPUS="0"`): 3 cells died with `CUDA error: the launch timed out` while two ran concurrently. The brief says two per 80 GB+ card and estimates its 4 days with two cells at a time. Neither card drives a display, so it is not a display watchdog; the cause is unknown.
3. Capture time for 14B: the brief says ~45 min per task under "Machine and cost" and ~25 min under "Notes".
4. The brief calls the per-cell times "measured on machine C"; `parallel-runs.md` says machine A.
5. `docs/5-full-pipeline.md` still says "nothing run yet", but qwen-7b is complete (commit 1b7ba64). That's a shared doc, so I have not edited it.
6. `full_pipeline.sh` runs all calibration on one GPU (`CALIB_GPU`), so a second GPU sits idle for the whole calibration.
7. The supervisor's lock, queue (`logs/eta_rules/jobs_vr16.txt`), record count and stall `pkill` pattern are per checkout and per value rank, not per model. That's safe only while one model runs from this checkout, which is the case here.
8. Old logs under `logs/` still carry the `codec_` prefix and `results/codec/` paths. They are local only and never committed.

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
