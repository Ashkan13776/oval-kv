# Closed-form rules for the key-basis mixing coefficient eta

Six rules that pick eta without an accuracy sweep, evaluated per (layer, KV
head) on `ds-r1-qwen-7b` and `ds-r1-llama-8b`, MATH50 / AIME24 / GPQA50c,
k = 8 seeds, in the value-codec configuration (keys rank 8 scored and
reconstructed, values plain PCA rank 8, FreeKV spec_ret protocol). No
reference eta grid is run in the harness; the rules are compared with each
other, with the existing fixed-eta qwen cells (MATH50 0.25 / AIME24 1.0 /
GPQA50c 0.25, `..._eta{E}_q0_recon1_vr8_vpca`) as a baseline row, and with an
**offline** fine-grid optimum measured on held-out calibration data.

Code: `oval/eta_rules.py` (the rules, pure math), `analysis/eta_rules_calibrate.py`
(measurement), `analysis/eta_rules_queue.py` + `scripts/eta_table_sweep.sh` +
`scripts/supervise_eta_tables.sh` (harness cells), `analysis/eta_rules_table.py`
(scoring), `analysis/test_eta_rules.py` (unit tests), `logs/run_eta_rules.sh`
(the chained pipeline). Results: `results/eta_rules/`.

## Definitions

Per page (32 tokens) and KV head: `X` centred keys, `Y` centred values,
`G = sum_{i in group} W_O,i^T W_O,i`,

    A = X^T X / tr(X^T X)          C = X^T Y G Y^T X / tr(X^T Y G Y^T X)
    M_eta = (1-eta) A + eta C      P_eta = Proj_r(M_eta),  r = 8

`build_summary` computes the same basis through the page x page Gram route;
the calibration script works in the row space of `X` (its right singular
vectors `V`, 32 coordinates), where `A`, `C`, `T`, `P_eta` and `P_eta'` are
32 x 32 and the d x d forms are recovered as `V (.) V^T`. All etas are clipped
to `[0, 1]` and rounded to 4 decimals when written to a table.

### The retrieval loss L(eta)

For a calibration query `q` at decode position `t`, with the method's sparse
attention at eta (OVAL scoring of every page with the rank-8 key records at
eta, avgSM GQA combine, top-64 pages per KV head, selected keys rebuilt from
the rank-8 key records, selected values rebuilt from the rank-8 value-PCA
records, sink 512 and recent 512 exact) and dense attention over the whole
context:

    Delta_h(q)  = sum_{i in group(h)} W_O,i (o_i(q) - o_hat_i(q))      in R^{d_model}
    L_h(eta)    = sum_q ||Delta_h(q)||^2 / sum_q ||u_h(q)||^2,   u_h = sum_i W_O,i o_i

i.e. the KV head's contribution to the residual stream, relative to its dense
norm. That is the "G-metric" error made exact: `||W_O (o - o_hat)||^2` with
the GQA cross-head terms kept. The normalisation is per (layer, head), so it
affects only how the number reads, not any rule (rules 1 and 2 are
scale-invariant) nor the argmin. Fine grid: 21 points, eta = 0, 0.05, ..., 1.

**Simplifications vs the harness (documented, deliberate).** Pages are scored
with the *current* query, not spec_ret's previous query with the cosine
correction. Queries come in windows of 96 positions ending at a page boundary
`L`; every selectable page lies before `L - 512`, so all queries in a window
see the same page set, and the recent region is `[L-512, t]` (416..512
tokens) rather than exactly 512. Layer 0 is measured but is dense in the
harness (`skip_layer 1`); its table entries are recorded and unused.

### The rules (each gives eta per (layer, KV head))

1. **Endpoint error interpolation.** `a = sum||Delta_0||^2`, `b = sum||Delta_1||^2`,
   `c = sum Delta_0 . Delta_1` over the head's calibration queries;
   `eta = clip((a-c)/(a+b-2c))`. `a+b-2c = sum||Delta_0 - Delta_1||^2`; if it
   is <= 1e-9 max(a,b) the objective is flat and eta = 0.5 (counted).
2. **Quadratic through L(0), L(1/2), L(1)** of the actual loss above. Convex
   -> clipped vertex; otherwise the endpoint with the smaller loss (counted
   as degenerate).
3. **Local update through the eigenspace.** eta0 = 0.5, M0 = (1-eta0)A + eta0 C,
   `D = C - A`, `P0' = sum_{i<=r, j>r} (u_j^T D u_i)/(lambda_i - lambda_j)(u_j u_i^T + u_i u_j^T)`.
   With `J_p(q)` the Jacobian of the attention output w.r.t. that page's
   logits (`dO/dl_t = p_t (v_t - o)`, dense attention, exact keys and values),
   `g_p = J_p X_p (I - P0) q / sqrt(d)`, `h_p = J_p X_p P0' q / sqrt(d)`,
   summed over pages (simultaneous perturbation) and mapped through `W_O` and
   the GQA group so the quadratic lives in the residual stream:
   `eta = eta0 + sum_q g.h / sum_q h.h`. Pages whose eigengap
   `lambda_r - lambda_{r+1} <= 1e-6 lambda_1` are dropped from the sums
   (counted); a denominator <= 1e-12 sum||g||^2 keeps eta0 (counted).
   **Deployed: the bounded version**, clip to `[eta0-0.25, eta0+0.25]` --
   the derivative is a first-order expansion of an invariant subspace and is
   only trusted locally; the unbounded eta is recorded (`tables_variants/*_r3u`).
4. **Balancing endpoint sacrifices.** `a_p = tr(P_A A) - tr(P_C A)`,
   `b_p = tr(P_C C) - tr(P_A C)` (both >= 0), `eta = sum_p b_p / sum_p (a_p+b_p)`
   over the head's pages with **uniform weights** (deployed: the rule needs
   no queries, which is its point); the attention-mass-weighted variant
   (`w_p` = mean dense attention mass of the page) is recorded (`*_r4w`).
   `sum(a+b) = 0` -> 0.5 (counted).
5. **Matching a more faithful sensitivity matrix.**
   `T_p = (1/d) X_p^T E_q[H_p(q)] X_p`, `H_p(q) = sum_{i in group} J_{p,i}(q)^T W_O,i^T W_O,i J_{p,i}(q)`
   (the exact second moment of the group's residual error under independent
   isotropic per-head queries; it drops only the cross-head terms). `T`, `A`,
   `C` normalised to unit trace, shared eta over the head's pages = closed form
   of the summed Frobenius objective: `sum_p <T_p-A_p, C_p-A_p> / sum_p ||C_p-A_p||^2`.
   `C = A` on every page -> 0.5 (counted).
6. **Learned rule.** `eta_p = sigmoid(b0 + b1 log((b+eps)/(a+eps)) + b2 rho + b3 (gamma_C - gamma_A))`
   with (a, b) from rule 4, `rho = tr(AC)/(||A|| ||C||)`, `gamma_M` = top-r
   eigenvalue mass of M. Fitted by weighted soft-label logistic regression
   (Newton, ridge 1e-3 on the slopes) to per-page targets = argmin over the
   fine grid of the page's **first-order** output error
   `e_p(eta) = mean_q ||sum_i W_O,i J_{p,i} X_p (I - P_eta) q_i||^2 / d`
   (the same Jacobian as rules 3 and 5; the full pipeline loss is not
   separable by page). Page weights = `mean_eta e_p(eta)`, normalised to sum
   to 1 within each (layer, head), so pages where eta matters dominate and
   every head counts equally. **Training split (deployed): one fit per
   model on that model's own calibration pages, the three tasks pooled,
   layers >= 1** -- the targets are offline losses on the calibration traces
   (seed 7, traces 1-4), never accuracy, so this is held out from the test
   traces and from every evaluated (problem, seed) exactly as rules 1-5 are
   (see Decisions). The cross-model fit (betas for model M fitted on the
   OTHER model's pages) is recorded as the variant `*_r6x`. Both sets of
   betas are in `results/eta_rules/rule6_fit.json`. Deployed per-(layer,
   head) eta = uniform mean of the per-page predictions over the head's
   calibration pages (query-free at deployment, like rule 4); the
   attention-mass-weighted mean is recorded (`*_r6w`).

### Calibration data (held out)

Per (model, task): 6 problems spread over the dataset (indices
`round(i*n/6)`), each generated by the model itself with the harness prompt
(`problem + " Please reason step by step, ... \boxed{}.\n<think>\n"`, no chat
template, as `eval/util.py::build_chat` does for ds-r1), **sampling seed 7**
(outside the evaluated 42..49), temperature 0.6 / top-p 0.95, exactly 6144
generated tokens (`min_new_tokens`, so every trace is long enough for the
selection to be real: 160+ selectable pages against a 64-page budget). Post-RoPE
K/V from the cache, queries re-rotated at their own positions, three windows
of 96 queries ending at `T`, `T-1024`, `T-2048` (page-aligned).
**Traces 1-4 calibrate (every rule), traces 5-6 are the test split** for the
offline regret. The eval runs every problem of a dataset under seeds 42-49,
so the calibration problems reappear there with different sampled traces;
the rules only ever see the seed-7 traces.
Cost: ~25 min generation + ~15 min measurement per (model, task); the two
models run on separate GPUs in parallel (~3-4 GPU-hours total).

### Offline regret

`R_h = L_test,h(eta_hat_h) - min_grid L_test,h`, evaluated on the test traces
with the harness's own per-head build (`build_summary` with a tensor eta), and
reported as the mean over layers >= 1. `calib_grid_opt` (the per-head grid
optimum of the calibration split, evaluated on the test split) shows how much
of the optimum transfers at all.

## Harness support: `--eta_table`

`--eta_table <json>` (patch to `eval/util.py` and `kvc/patch/llama.py`) loads
`{"eta": [n_layers][n_kv_heads]}`, gives each layer's attention module a
float64 tensor `[n_kv]`, and `build_summary` broadcasts it over pages. Head
order: with sparsity 1 every KV head is dynamic and `reorder_linear_weights`
keeps the model's order, so row h is the model's KV head h, the same index
`publish_metric` uses for `G[h]`; `load_eta_table` refuses tables with the
wrong shape or any full head. The output filename carries `_etab<hash>` in
place of `_eta<E>`, `hash` = 8 hex of sha1 over the 4-decimal-rounded values
(`oval.eta_table_hash`), so two rules selecting the same table share one cell
and any difference >= 1e-4 anywhere is a different cell. Without the flag
every filename and every number is unchanged (checked: the existing
`..._eta0.25_q0_recon1_vr8_vpca-...` name is reproduced byte for byte).

Exactness (`analysis/test_eta_rules.py`, check 7): a constant table equals the
scalar `--eta E` records **bitwise** for any E (the per-head path forms
`tr.reciprocal() * float32(1-eta)` exactly as Python's `float / tensor` does,
and eta = 0 heads take the plain-PCA branch in place). A mixed table equals
building each head separately with its own scalar eta up to fp32 round-off
(~1e-5 on reconstructed keys and scores): MKL's gemm result depends on the
batch it is handed, so bitwise agreement across different batch shapes is not
available on any path. The launcher's harness smoke test repeats the
constant-table-vs-scalar comparison on the GPU and logs whether the generated
text is identical.

## Running / reproducing

```bash
python analysis/test_eta_rules.py                       # CPU unit tests
python analysis/eta_rules_calibrate.py capture --model ds-r1-qwen-7b --task MATH50
python analysis/eta_rules_calibrate.py stats   --model ds-r1-qwen-7b --task MATH50
python analysis/eta_rules_calibrate.py rules            # needs both models' stats
python analysis/eta_rules_calibrate.py regret  --model ds-r1-qwen-7b --task MATH50
python analysis/eta_rules_queue.py                      # dry run of the cells
scripts/supervise_eta_tables.sh                         # runs the cells (own lock/queue/logs)
python analysis/eta_rules_table.py --out results/eta_rules/table.md
```

`logs/run_eta_rules.sh` chains all of this: it waits for the value-codec sweep
(`run_value_codec.sh` gone and `supervisor.log` showing ALL COMPLETE for vr32;
if the sweep vanishes without that line it waits a 2 h grace period that any
restart resets, logging a WARNING every 30 min, then proceeds -- `touch
logs/eta_rules/HOLD` holds it indefinitely, `touch logs/eta_rules/GO` starts
it now) and for the Llama-8B weights, runs a harness smoke test and a
calibration smoke test and aborts on any failure, then calibrates, queues
MATH50 first, runs the six deployed rules to ALL COMPLETE, then a second
supervisor pass for the variants `r3u r4w r6x` on MATH50 only, and writes the
table. Logs in `logs/eta_rules/` (`launcher.log`, `supervisor.log`,
`sweep_run.log`, `calib_<model>.log`, per-cell `cell_*.log`).

### Outputs (`results/eta_rules/`)

| file | |
|---|---|
| `<model>_<task>.json` | config + timestamp, the loss grid per split/layer/head, every rule's eta and intermediates per (layer, head) (a, b, c; L0, Lm, L1; num/den, eigengap counts; sums for rule 4/5; betas), degenerate flags and counts, the regret block |
| `<model>_<task>_pages.npz` | per-page rows: layer, head, the three rule-6 features, target eta, weight, attention mass, a, b |
| `tables/<model>_<task>_<rule>.json` | the deployed table the harness consumes (`eta`, `hash`, rule, timestamp) |
| `tables_variants/*_r3u / _r4w / _r6w / _r6x` | recorded alternatives (`variant_of`, `deploy: false`); all four get the offline regret, `r3u r4w r6x` are also run in the harness on MATH50 after the deployed rules |
| `rule6_fit.json` | per cell: the deployed within-model fit and the cross-model fit (`r6x`), each with betas, training models/tasks and row count |
| `table.md` | the final table |

The calibration script never overwrites a stats file or a table with a
different hash without `--overwrite`; captures are cached in
`logs/eta_rules/capture/` (~0.5-1 GB per trace).

## GPU time

288 seed-cells (6 rules x 2 models x 3 tasks x 8 seeds), fewer if rules
coincide. With 8 concurrent cells: MATH50 ~3-5 h/cell, AIME24 and GPQA50c
~7-8 h/cell, i.e. roughly 900 cell-hours per model, ~9-10 days for both at
full occupancy (11-12 with the other user's load); MATH50 for both models
(96 cells) lands first, after ~2 days. Calibration adds ~3-4 GPU-hours.
The variant pass (`r3u r4w r6x`, MATH50 only, both models) is at most 48
seed-cells, ~1 day, and runs only after everything above is complete; a
variant whose table hashes equal to its deployed rule's costs nothing.

## Decisions

Three open choices were settled on 2026-10-21, before any GPU time was spent.

1. **Variants: `r3u`, `r4w`, `r6x` are run in the harness on MATH50 only,
   after the six deployed rules; `r6w` is recorded (offline regret) but not
   run.** Each of the three answers a distinct design question -- is the
   +-0.25 trust region of rule 3 load-bearing, does query-dependent page
   weighting beat the query-free rule 4, and does the learned rule transfer
   across architectures -- whereas `r6w` only changes how per-page
   predictions are averaged. Running all variants on every task would add
   ~50% (~4-5 days) for ablations whose primary quantity, the retrieval
   loss, is already measured for every variant on the held-out traces at
   zero GPU cost (the regret stage). MATH50 is the cheapest task, and the
   pass is queued after the deployed rules so the main result is never
   delayed; the queue builder dedupes by table hash, so a variant that
   coincides with its rule is an alias of an already complete cell.
2. **Rule 6 is fitted within model (own calibration pages, tasks pooled);
   the cross-model fit is recorded as `r6x`.** With two models of different
   families the cross-model scheme confounds two questions: whether the
   three features predict the per-page optimum at all, and whether that
   mapping transfers from Qwen-7B to Llama-8B. A within-model fit is held
   out from everything the rule is scored on -- the per-page targets are
   offline first-order losses on the seed-7 calibration traces 1-4, the
   regret is measured on traces 5-6, and accuracy on seeds 42-49 -- in
   exactly the sense rules 1-5 are, which estimate `n_layers x n_kv` free
   numbers from the same traces while rule 6 estimates four. Disjoint-layer
   or leave-one-task-out schemes were rejected: they reduce the training
   set and add an artificial transfer axis to the deployed rule. The
   cross-architecture question is kept as the `r6x` variant (offline regret
   on both models, harness on MATH50).
3. **If the value-codec sweep disappears without ALL COMPLETE for vr32, the
   launcher waits a 2 h grace period, then proceeds loudly.** The eta study
   needs only free GPUs, so waiting indefinitely would idle four GPUs for
   as long as nobody notices, while the 30 min of the first version was
   short enough to race a manual restart of the sweep. Any sweep process
   reappearing resets the clock, an ABORT in `logs/supervisor.log` is named
   as such, a WARNING is logged every 30 min, and `logs/eta_rules/HOLD` /
   `GO` give the user explicit control without editing the running script.
   The schedulers that follow all gate on free GPU memory, so even a late
   restart would share the machine rather than crash it.

**2026-10-28: rule 1 prioritised.** At the user's request rule 1's remaining
cells (Llama AIME24, both models' GPQA50c) were moved to the front of the queue.
`analysis/eta_rules_queue.py` reads rule names from `logs/eta_rules/PRIORITY`
(here `r1`) and puts their seed-cells first, so a supervisor restart keeps the
order. Delete the file to return to the default dataset-major order.

**2026-10-29: other rules paused after rule 1.** Once rule 1's cells were
complete the remaining queue (rules 2-6 and the variants) was paused at the
user's request so the GPUs could time decode latency. The emptied queue is
backed up in `logs/eta_rules/jobs_paused_backup.txt` (local); completed cells
of the other rules are in `results/codec/` and `table.md` but are not
discussed here. To resume, relaunch `scripts/supervise_eta_tables.sh` with
the full rule list, then the variant pass.

## Rule 1 results (complete, k = 8)

Keys rank 8 (scoring and reconstruction), values plain PCA rank 8, one eta
per (layer, KV head) from rule 1; avg@8 / pass@8, change in avg@8 vs the
fixed-eta cell (qwen only: MATH50 0.25, AIME24 1.0, GPQA50c 0.25) with a
paired bootstrap 95% CI over problems.

| model | task | eta mean / median | avg@8 | pass@8 | vs fixed eta |
|---|---|---|---|---|---|
| qwen-7b | MATH50 | 0.57 / 0.68 | 70.75 | 78.00 | +0.50 [-1.75, +3.00] |
| qwen-7b | AIME24 | 0.45 / 0.47 | 51.25 | 80.00 | +4.58 [-2.50, +12.50] |
| qwen-7b | GPQA50c | 0.47 / 0.55 | 34.75 | 64.00 | +1.00 [-2.00, +4.00] |
| llama-8b | MATH50 | 0.87 / 1.00 | 66.25 | 76.00 | -- |
| llama-8b | AIME24 | 0.77 / 0.96 | 48.33 | 83.33 | -- |
| llama-8b | GPQA50c | 0.82 / 1.00 | 37.50 | 68.00 | -- |

Rule 1 is at or above the hand-picked fixed eta on every qwen task; no
difference is significant. Llama has no fixed-eta cell to compare against.

**Output error vs eta** (`analysis/plot_eta_rule1.py` ->
`results/eta_rules/figures/eta_rule1_error.png`). Per model: (a) measured
calibration loss L(eta) for example heads against rule 1's quadratic
(1-eta)^2 a + eta^2 b + 2 eta(1-eta) c, (b) held-out excess output error over
the per-head grid oracle for every shared eta vs rule 1's per-head table,
(c) the distribution of eta*, (d) eta* vs the held-out best grid eta. The
quadratic matches the measured L(0.5) to < 1% (median head). Held-out excess
error, best shared eta -> rule 1: qwen MATH 4.7% -> 0.7%, AIME 3.0% -> 0.3%,
GPQA 3.5% -> 1.3%; llama MATH 6.1% -> 4.1%, AIME 2.5% -> 1.5%, GPQA
3.2% -> 1.5%. Correlation of eta* with the held-out best eta: qwen
0.83-0.87, llama 0.45-0.60.

## Decode latency of rule 1

`scripts/latency_rule1.sh` (+ `analysis/plot_latency_rule1.py` ->
`results/eta_rules/figures/latency_rule1.png`, per-step times in
`results/eta_rules/latency/<model>_<arm>.json`). FreeKV's efficient engine
(`third_party/FreeKV/source`), one RTX A6000 per model, batch 1, greedy,
AIME24 problem 0, exactly 16384 decoded tokens (`--ignore_eos`), one warmup
round; metric = mean ms per decode step of the timed round. Budget 2048
tokens INCLUDING 512 sink + 512 recent (the perf engine's convention, unlike
the accuracy harness), pages of 32.

| model | Full KV | ArkVale | RaaS | FreeKV | OVAL fixed eta (1.0) | OVAL rule 1 |
|---|---|---|---|---|---|---|
| qwen-7b | 24.67 | 71.92 | 28.64 | 35.86 | 40.17 | 40.40 |
| llama-8b | 26.83 | 142.56 | 30.36 | 42.03 | 46.94 | 46.65 |

Rule 1 costs nothing at decode: eta enters only the page-record build, once
per page, and the per-step scoring and attention never read it, so the
rule-1 and fixed-eta arms agree within run-to-run noise (+0.6% / -0.6%).
OVAL is ~12% above FreeKV (the rank-8 record build and scoring).

**Per sequence, a sweep costs ~5x.** Setting eta by sweeping means plugging
each of the 5 grid values {0, .25, .5, .75, 1} in, i.e. decoding the sequence
5 times. Rule 1 decodes it once, plus the cost of finding eta*: generating
the 4 calibration traces (6144 tokens each, dense HF decode) and computing
the endpoint errors, a, b, c and the closed form on them
(`analysis/time_rule1_find.py`, AIME24, A6000; it reproduces the deployed
tables to the 4-decimal rounding). Right panel of the figure, one
16384-token sequence:

| model | sweep (5 decodes) | rule 1 decode | find eta* (traces + compute) | sweep / rule 1 |
|---|---|---|---|---|
| qwen-7b | 55 min | 11 min | 18 min (12.3 + 5.7) | 1.9x on the first sequence, -> 5.0x |
| llama-8b | 64 min | 13 min | 27 min (13.8 + 12.8) | 1.6x on the first sequence, -> 5.0x |

eta* is found once per (model, task) and reused by every later sequence, so
the finding cost is charged in full to a single sequence above and the
ratio tends to the 5x decode ratio after a few sequences. A sweep also needs
a way to pick among the 5 outputs (answers or a score); rule 1 needs no
labels.

Caveats:

- The perf engine has no kernel for the key/value codec, so the OVAL arms
  recall EXACT KV for the selected pages; these numbers time selection, not
  reconstruction.
- At 16K context the full cache fits on one A6000 and full attention is the
  fastest arm; the sparse arms pay offload/recall overhead that only wins at
  longer contexts or larger batches.
- The engine intermittently (about 1 run in 3) hits a device-side "index out
  of bounds" assert in the FIRST generation of a process -- plain FreeKV too,
  independent of OVAL; it disappears under `CUDA_LAUNCH_BLOCKING=1` (a race).
  It never hit a timed round, so the script re-runs a crashed process (up to
  6 times). All 12 runs above completed on the first attempt.
- The per-head path in the engine (`freekv/oval_summary.py`, copied from
  `oval/page_summary.py`) moves each layer's eta row to the GPU once and
  avoids host syncs in the decode loop; it builds records bitwise equal to
  the accuracy harness's (checked at eta 0, 0.25, 1.0 and with constant
  tables vs the scalar path).
