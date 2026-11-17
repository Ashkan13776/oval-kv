# Experiments

Each folder holds the drivers for one part of the paper. Shell drivers find the
repository root themselves (`KV_ROOT`, overridable) and use `KV_PY` for the
harness python. See the top-level README for setup.

| folder | paper | what |
|---|---|---|
| [`reasoning/`](reasoning) | Tab. 1, Fig. 2a | long reasoning (MATH500, AIME24, GPQA; R1-Distill Llama-8B, Qwen-7B, Qwen-14B; k = 8) and the accuracy-vs-memory study |
| [`long_context/`](long_context) | Tab. 2 | LongBench v2 and LongGenBench (Llama-3.1-8B, Qwen2.5-7B, Qwen2.5-14B) |
| [`video/`](video) | Tab. 3, Fig. 3, Fig. 1b, App. | Video-MME and LVBench with Qwen2.5-VL-7B (`run_qwen25vl.py`) and LLaVA-Video-7B (`llava/`), token-pruning baselines, LIBERO-Recall, paired statistics |
| [`robot/`](robot) | Tab. 3 (W&D) | LIBERO Watch-and-Do: build the task, plan with the video LLM under each KV method, execute with OpenVLA |
| [`latency/`](latency) | Fig. 2b,c | decode latency in FreeKV's efficient engine |
| [`ablations/`](ablations) | Tab. 4, App. | key / value rank, shared vs per-head eta, rule variants, disjoint calibration |

## Paper artifact -> driver -> scorer

| artifact | driver | scored / plotted by |
|---|---|---|
| Tab. 1 long reasoning | `reasoning/full_pipeline.sh` (capture, rule-1 calibration, decode, score) | `analysis/score_avg_pass_at_k.py`, `analysis/eta_rules_table.py` |
| Fig. 2a accuracy vs memory | `reasoning/pareto_arms.sh`, `reasoning/pareto_sched.sh`, `reasoning/baselines_for_tradeoff.sh` | `analysis/pareto.py`, `analysis/plot_pareto.py`, `analysis/task_memory.py`, `analysis/baseline_memory.py` |
| Tab. 2 LongBench v2 / LongGenBench | `long_context/long_context_pipeline.sh` | `analysis/score_long_context.py` |
| Tab. 3 Video-MME / LVBench | `video/gen_jobs_c1.py`, `gen_jobs_c2.py`, `gen_jobs_c3.py` + `video/worker.sh` | `video/summarize_c.py`, `video/paper_table.py`, `video/memory_c.py`, `video/pair_arms.py` |
| Tab. 3 Watch-and-Do | `robot/watchdo_build.py`, `robot/watchdo_jobs.py`, `robot/watchdo_rollout.py` | `robot/watchdo_table.py` |
| Fig. 3 video Pareto | (Tab. 3 runs) | `video/fig_video_summary.py` |
| Fig. 1b, App. filmstrips | (LIBERO-Recall / Video-MME runs) | `video/fig1b_*.py`, `robot/fig1b_wd.py`, `video/supp_filmstrips.py` |
| LIBERO-Recall (App.) | `video/liberorecall_build.py`, `video/gen_jobs_lr.py` | `video/lr_table.py` |
| VisionZip + OVAL, 128 frames (App.) | `video/visionzip.py`, `video/iso128.sh` | `video/iso128_summary.py` |
| Fig. 2b,c decode latency | `latency/fig6_latency_all.sh`, `latency/latency_rule1.sh` | `analysis/plot_fig6_freekv_style.py`, `analysis/plot_latency_rule1.py` |
| Tab. 4 key / value rank | `ablations/oval_recon_rank_sweep.sh`, `ablations/value_rank_sweep.sh` | `analysis/score_avg_pass_at_k.py`, `analysis/value_rank_table.py` |
| Tab. 4 shared eta, per-head rule vs key PCA | `reasoning/oval_reasoning_sweep.sh` + `reasoning/supervise_sweep.sh` | `analysis/score_avg_pass_at_k.py` |
| Rule-of-thumb fit (App.) | `ablations/a1_calib_disjoint.sh`, `ablations/a5_grid_vr16*.sh`, `ablations/a3_tables.py` | `ablations/score.py`, `analysis/fig_eta_revision.py` |

`ablations/make_jobs.py` writes job files and `ablations/queue.sh` runs them on
the free GPUs (`JOBS=<file> GPUS="0 1" ./experiments/ablations/queue.sh`).

## Environments

- Text models: the FreeKV harness env (`experiments/ablations/env.sh`; set `FREEKV_ENV`).
- Video and robot: `experiments/video/env.sh` (sources `video/llava/env.sh`; set `OVAL_ENV`).
- Data roots: `VMME_ROOT` (Video-MME), `LVB_ROOT` (LVBench), `LR_ROOT` (LIBERO-Recall),
  `WD_ROOT` (Watch-and-Do), `OPENVLA_ROOT` (OpenVLA checkout). Defaults are under `$HOME`.

## Build

`scripts/build_freekv.sh` (with `scripts/fk_configure.sh`, `scripts/fk_compile.sh`)
builds FreeKV's CUDA extension, needed only for the latency runs.
