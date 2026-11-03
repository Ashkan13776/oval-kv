# Archive: eta rules 2-6

Six closed-form rules for choosing the key-basis mixing coefficient eta per
(layer, KV head) were implemented and calibrated together (October 2026).
**Rule 1 was kept as the method** (`docs/4-per-head-eta.md`); rules 2-6 and
their variants were stopped on 2026-10-29 and archived on 2026-11-03. Nothing
in the current pipeline reads this folder.

## What is here

| path | |
|---|---|
| `code/` | snapshot of the six-rule code at commit `a2dd95f` (`eta_rules.py`, the calibration, queue, table and test scripts). Reference only: it imports from the repo layout of that commit, so to run it, `git checkout a2dd95f` |
| `stats/<model>_<task>.json` | the full calibration stats: loss grid, every rule's eta and intermediates, the offline regret block |
| `tables/<model>_<task>_r<2-6>.json` | rule 2-6 eta tables as the harness consumed them (`--eta_table`) |
| `tables_variants/` | recorded variants r3u, r4w, r6w, r6x (offline regret only, never run in the harness) |
| `rule6_fit.json` | rule 6 logistic fits (within-model and cross-model betas) |
| `cells/<model>-spec_ret/*_etab<hash>_*.jsonl` | the harness cells of rules 2-6 (keys rank 8 scored and rebuilt, values PCA rank 8, k up to 8) |
| `table_all_rules.md` | the full results table of all six rules as it stood when they were stopped |

## Results when stopped

avg@8 (k = 8 unless noted), keys rank 8 + values PCA rank 8, and offline
regret (relative output error above the per-head grid optimum on held-out
traces; lower is better). Rule 1 is shown for reference. "--" = not run.

| model / task | baseline fixed eta | r1 | r2 | r3 | r4 | r5 | r6 |
|---|---|---|---|---|---|---|---|
| qwen-7b MATH50 avg@8 | 70.25 | **70.75** | 70.00 | **70.75** | 70.25 | 70.00 | 70.50 |
| qwen-7b AIME24 avg@8 | 46.67 | **51.25** | 46.25 | 43.33 | 44.17 | 50.42 | 44.17 |
| qwen-7b GPQA50c avg@8 | 33.75 | 34.75 | -- | -- | -- | -- | -- |
| llama-8b MATH50 avg@8 | -- | 66.25 | 66.00 | 66.50 | **66.75** | 65.50 | 65.00 |
| llama-8b AIME24 avg@8 | -- | 48.33 | 42.00 (k=5) | -- | -- | -- | -- |
| llama-8b GPQA50c avg@8 | -- | 37.50 | -- | -- | -- | -- | -- |
| regret, mean of the 6 cells | | 0.00042 | 0.00040 | 0.00169 | 0.00282 | 0.00189 | 0.00220 |

Rule 2 matches rule 1 offline (both read the actual loss, rule 2 at three
points) but not in the harness on qwen AIME24; rules 3-6 have 4-7x rule 1's
regret. No difference between rules on accuracy is significant.

## The six rules (each gives eta per (layer, KV head))

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
   betas are in `rule6_fit.json` (this folder). Deployed per-(layer,
   head) eta = uniform mean of the per-page predictions over the head's
   calibration pages (query-free at deployment, like rule 4); the
   attention-mass-weighted mean is recorded (`*_r6w`).


The loss L(eta), calibration data and regret definition are shared with rule 1
and documented in `docs/4-per-head-eta.md`.

## Decisions taken while the study ran

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
3. **If the value-compression sweep disappears without ALL COMPLETE for vr32, the
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
user's request so the GPUs could time decode latency; on 2026-11-03 they were
dropped and archived here (their completed cells are in `cells/`, their
results in `table_all_rules.md`).

