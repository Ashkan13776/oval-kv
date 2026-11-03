# Scripts and analysis, by piece of OVAL

Every driver reads `KV_ROOT` (repo root) and `KV_PY` (the harness env's
python); see the top-level README for setup. Pieces are described in
[docs/README.md](../docs/README.md).

## 5. Full pipeline (start here)

| script | what |
|---|---|
| `scripts/full_pipeline.sh` | capture, rule-1 calibration at value rank 16, tables, supervised sweep, scoring, for any set of models ([docs/5-full-pipeline.md](../docs/5-full-pipeline.md)) |

## 1. Retrieval

| script | what | scored / plotted by |
|---|---|---|
| `oval_reasoning_sweep.sh` + `supervise_sweep.sh` | long-reasoning eta x seed grid (keys rebuilt, `--oval_recon`) | `analysis/score_avg_pass_at_k.py` |
| `lb2_eta_sweep.sh`, `lb2_rank_sweep.sh`, `lb2_controls.sh` | LongBench v2, Qwen-2.5-14B | `analysis/score_accuracy.py`, `analysis/all_tables.py` |
| `ib_freekv.sh`, `ib_full_refix.sh` | InfiniteBench, FreeKV vs OVAL in one harness | `analysis/all_tables.py` |
| `freekv_sweep.sh` | FreeKV Table 3 protocol | `analysis/score_freekv.py` |
| `fig6_latency_all.sh`, `lb2_latency_mvp.sh`, `lb2_latency_wait.sh` | decode latency | `analysis/plot_fig6_freekv_style.py`, `analysis/plot_acc_vs_latency.py` |
| `baselines_for_tradeoff.sh` | ArkVale, RaaS, FullKV for the trade-off plot | `analysis/plot_acc_vs_latency.py` |
| `pareto_arms.sh`, `pareto_sched.sh` | 26-arm matched-budget comparison (machine C) | `analysis/pareto.py`, `paired.py`, `front.py`, `plot_pareto.py`, `plot_memory.py`, `baseline_memory.py` |

## 2. Key compression

| script | what | scored / plotted by |
|---|---|---|
| `oval_key_rank_grid.sh` | rank grid, scoring rank = reconstruction rank, seed 42 | `analysis/score_key_rank_grid.py` |
| `oval_recon_rank_sweep.sh` | scoring rank 8, reconstruction rank 1-32, k = 8 | `analysis/score_avg_pass_at_k.py` |
| `finish_stuck_cells.sh` | re-runs cells that hit a CUDA launch timeout | |
| | key-store bytes, closed form | `analysis/oval_memory.py`, `analysis/task_memory.py` |
| | low rank vs int4/int8 quantisation on captured keys | `analysis/oval_vs_quant.py`, `analysis/capture_kv.py` |

## 3. Value compression

| script | what | scored / plotted by |
|---|---|---|
| `value_rank_sweep.sh` | value PCA rank 1-32, k = 8 | `analysis/value_rank_table.py` |
| | output error of value records on real decode state | `analysis/value_compression_check.py` |

## 4. Per-head eta (rule 1)

| script | what | scored / plotted by |
|---|---|---|
| | calibration (capture, stats, tables, regret) | `analysis/eta_rules_calibrate.py` |
| `eta_table_sweep.sh` + `supervise_eta_tables.sh` | harness cells with `--eta_table` (any `VRANK`) | `analysis/eta_rules_queue.py`, `analysis/eta_rules_table.py` |
| `latency_rule1.sh`, `latency_oracle_r1.sh` | decode latency of rule 1 vs the arms and vs a 5-eta sweep | `analysis/plot_latency_rule1.py`, `analysis/time_rule1_find.py` |
| | output error vs eta figure | `analysis/plot_eta_rule1.py` |

## Build and tests

| | |
|---|---|
| `build_freekv.sh`, `fk_configure.sh`, `fk_compile.sh` | build FreeKV's CUDA extension (efficient engine, latency runs) |
| `analysis/test_page_basis.py`, `analysis/test_eta_rules.py` | CPU unit tests of the records, scoring, value records and rule 1 |
| `analysis/all_tables.py`, `analysis/subset.py`, `analysis/paths.py` | shared scoring, InfiniteBench subset, path config |
