# OVAL: Output-Aware Local Page Bases for KV Cache Compression

**Ashkan Shahbazi, Chayne Thrash, Soheil Kolouri** · Vanderbilt University

[Project page](http://ashkan-shahbazi.com/oval-kv/) · [Paper (arXiv:2610.06686)](https://arxiv.org/abs/2610.06686)

Page-retrieval sparse attention (Quest, ArkVale, FreeKV, ...) reads only the
top-scoring KV pages at each decoding step, but it still stores every page
exactly, plus a digest for scoring, so the cache ends up slightly larger than
the full cache. **OVAL makes the index the storage.** Each page is kept only as
two low-rank bases: a key basis that scores the page and rebuilds its keys when
the page is selected, and a PCA value basis that rebuilds its values. The exact
pages are dropped and nothing else is added. Because rebuilt keys feed the
attention output, the key basis is **output-aware**: it mixes key geometry with
the attention-output error, using a per-head weight set in closed form from a
few unlabeled calibration traces.

At the same attended budget, OVAL is best or tied-best in eight of nine
long-reasoning settings, matches FullKV on MATH500 with 66% of its memory, is on
par with FullKV on LongBench v2, and on Video-MME / LVBench stores 53% of the
full cache instead of 103% while staying within 0.32 points of exact-page
retrieval.

## Method

For a page with centered keys `X` and values `Y` (rows = tokens), and
`G_h = sum_i W_O,i^T W_O,i` over the query heads that share KV head `h`:

```
M_eta = (1 - eta) * XᵀX / tr(XᵀX)  +  eta * XᵀY G_h YᵀX / tr(XᵀY G_h YᵀX)
B     = top-r eigenvectors of M_eta            (key basis, solved at size m x m)
score = mu·q/√d + logsumexp(C (Bᵀq) / √d),     C = X B
K̂     = 1 muᵀ + C Bᵀ,   V̂ = 1 nuᵀ + T Dᵀ        (rebuilt only for the selected pages)
```

`eta = 0` is page-local key PCA; `eta = 1` keeps only the output directions.
The per-head `eta_h` minimizes a quadratic interpolation of the measured
residual-stream error at the two endpoints:

```
eta_h = clip_[0,1]( (a_h - c_h) / (a_h + b_h - 2 c_h) )
```

Paper setting everywhere: 32-token pages, rank-8 key bases, rank-16 value
bases, per-head `eta_h`. The two bases take half the memory of the exact page.

Implementation: [`oval/page_basis.py`](oval/page_basis.py) (bases, scoring,
reconstruction, value basis, eta tables) and [`oval/eta_rules.py`](oval/eta_rules.py)
(the closed-form rule).

## Repository layout

| path | what |
|---|---|
| `oval/` | the method, plus the modules installed into FreeKV |
| `patches/freekv.patch` | our changes to the FreeKV harness (OVAL and the baseline scorers) |
| `setup.sh` | clones FreeKV at a pinned commit and applies the patch |
| `experiments/` | drivers, one folder per part of the paper: `reasoning/`, `long_context/`, `video/`, `robot/`, `latency/`, `ablations/` ([map](experiments/README.md)) |
| `analysis/` | scoring, tables, figures, CPU unit tests |
| `results/` | raw per-question outputs and tables behind the paper ([guide](results/README.md)) |
| `docs/` | notes per method component and harness ([index](docs/README.md)) |
| `scripts/` | build helpers for FreeKV's CUDA extension (latency runs only) |

## Setup

```bash
./setup.sh                        # FreeKV @ pinned commit + patches/freekv.patch
export KV_PY=$(which python)      # torch 2.8 (cu128) for video; the text runs used torch 2.5.1 (cu121)
pip install -r requirements.txt
./scripts/build_freekv.sh         # only for the latency runs (efficient engine)
python analysis/test_page_basis.py && python analysis/test_eta_rules.py
```

Shell drivers find the repository root themselves; override with `KV_ROOT`.

## Reproducing the paper

| paper | driver | scorer |
|---|---|---|
| Tab. 1 long reasoning | `experiments/reasoning/full_pipeline.sh` | `analysis/score_avg_pass_at_k.py` |
| Tab. 2 LongBench v2 / LongGenBench | `experiments/long_context/long_context_pipeline.sh` | `analysis/score_long_context.py` |
| Tab. 3 Video-MME / LVBench | `experiments/video/gen_jobs_c{1,2,3}.py` + `experiments/video/worker.sh` | `experiments/video/summarize_c.py`, `paper_table.py` |
| Tab. 3 Watch-and-Do | `experiments/robot/watchdo_{build,jobs,rollout}.py` | `experiments/robot/watchdo_table.py` |
| Tab. 4 ablations | `experiments/ablations/{oval_recon_rank_sweep,value_rank_sweep}.sh`, `experiments/reasoning/oval_reasoning_sweep.sh` | `analysis/score_avg_pass_at_k.py`, `analysis/value_rank_table.py` |
| Fig. 2a accuracy vs memory | `experiments/reasoning/pareto_arms.sh` | `analysis/pareto.py`, `analysis/plot_pareto.py` |
| Fig. 2b,c decode latency | `experiments/latency/fig6_latency_all.sh` | `analysis/plot_fig6_freekv_style.py` |
| Fig. 3 video Pareto | (Tab. 3 runs) | `experiments/video/fig_video_summary.py` |

The full map, including appendix studies, is in [experiments/README.md](experiments/README.md).

```bash
MODELS="ds-r1-qwen-7b ds-r1-llama-8b ds-r1-qwen-14b" ./experiments/reasoning/full_pipeline.sh
./experiments/long_context/long_context_pipeline.sh
./experiments/latency/fig6_latency_all.sh
```

## Results

All numbers are from the paper. Baselines and FullKV on text tasks are as
reported by FreeKV; OVAL runs in FreeKV's exact setting and harness (page size
32, attended budget 2048 including exact sink and recent tokens).

### Long reasoning (avg@8, %; k = 8, 8 seeds)

| Model | Task | Full | Razor | RaaS | Quest | ArkVale | ShadowKV | InfiniGen | FreeKV | **OVAL** |
|---|---|---|---|---|---|---|---|---|---|---|
| R1-Llama-8B | MATH500 | 67.25 | 60.50 | 62.50 | 62.00 | 62.50 | 60.25 | 66.50 | 66.75 | **67.25** |
| | AIME24 | 47.08 | 30.00 | 36.25 | 44.17 | 46.67 | 36.50 | 45.83 | **47.50** | **47.50** |
| | GPQA | 39.75 | 34.25 | 33.50 | 37.25 | 39.75 | 36.25 | 27.50 | 41.25 | **44.00** |
| | Average | 51.36 | 41.58 | 44.08 | 47.81 | 49.64 | 44.33 | 46.61 | 51.83 | **52.92** |
| R1-Qwen-7B | MATH500 | 71.75 | 66.75 | 67.00 | 68.00 | 68.25 | 64.75 | 70.00 | 70.00 | **71.75** |
| | AIME24 | 56.66 | 35.42 | 42.92 | 47.50 | 47.92 | 43.75 | 43.34 | **52.92** | 47.92 |
| | GPQA | 35.75 | 32.50 | 33.25 | 38.75 | 34.25 | 33.50 | 35.50 | **39.50** | **39.50** |
| | Average | 54.72 | 44.89 | 47.72 | 51.42 | 50.14 | 47.33 | 49.61 | **54.14** | 53.06 |
| R1-Qwen-14B | MATH500 | 70.25 | 59.75 | 64.75 | 67.25 | 66.25 | 65.00 | 67.00 | 67.50 | **69.25** |
| | AIME24 | 66.25 | 32.50 | 48.75 | 58.33 | 61.25 | 57.25 | 60.00 | 64.17 | **65.42** |
| | GPQA | 53.25 | 38.50 | 44.25 | 51.25 | 53.75 | 51.75 | 38.00 | 56.00 | **57.50** |
| | Average | 63.25 | 43.58 | 52.58 | 58.94 | 60.42 | 58.00 | 55.00 | 62.56 | **64.06** |

pass@8 averages (Llama-8B / Qwen-7B / Qwen-14B): FullKV 80.00 / 77.78 / 80.89,
FreeKV 80.22 / 78.44 / 82.44, OVAL 78.00 / 78.00 / 82.44. Per-task pass@8 and
standard errors are in the paper (Tab. 1).

Memory: on AIME24 with Qwen-14B, OVAL stores 2830 KiB per KV head and layer
(56% of FullKV's 5050 KiB) and is 0.83 points below FullKV; the retrieval
baselines store 101-103%. On MATH500 with Qwen-7B, OVAL matches FullKV (71.75)
at 66% of its memory.

### Long context (%)

| Model | Benchmark | Metric | Full | Razor | RaaS | Quest | ArkVale | ShadowKV | InfiniGen | FreeKV | **OVAL** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | LongBench v2 | Overall | 29.22 | 27.44 | 28.23 | 28.43 | 28.63 | 25.45 | 28.56 | 29.22 | **30.02** |
| | | Short | 34.44 | 33.89 | 33.89 | 33.33 | 33.89 | 32.78 | 32.28 | 35.00 | **36.11** |
| | | Medium | 27.91 | 25.12 | 26.51 | **27.44** | 26.98 | 22.79 | 26.05 | **27.44** | **27.44** |
| | | Long | 23.15 | 21.30 | 22.22 | 22.22 | 23.15 | 18.52 | 24.07 | 23.15 | **25.00** |
| | LongGenBench | CR | 80.03 | 35.90 | 76.63 | 78.03 | 39.36 | **79.28** | 76.68 | 78.03 | 77.87 |
| | | CR×Acc | 26.82 | 12.20 | 26.00 | 27.71 | 10.36 | **30.66** | 26.21 | 27.62 | 28.64 |
| Qwen2.5-7B | LongBench v2 | Overall | 27.44 | 25.25 | 26.24 | 27.63 | 26.84 | 25.84 | 26.44 | 26.84 | **29.82** |
| | | Short | 36.11 | 32.78 | 35.56 | **36.67** | 36.11 | 32.22 | 32.22 | 34.44 | **36.67** |
| | | Medium | 23.72 | 21.86 | 21.86 | 22.79 | 22.33 | 20.00 | 23.26 | 22.33 | **27.44** |
| | | Long | 20.37 | 19.44 | 19.44 | 22.22 | 20.37 | **26.85** | 23.15 | 23.15 | 23.15 |
| | LongGenBench | CR | 79.56 | 42.13 | **77.65** | 62.89 | 75.91 | 35.49 | 72.96 | 76.93 | 77.23 |
| | | CR×Acc | 31.09 | 21.48 | 34.40 | 25.96 | 31.79 | 11.43 | 27.67 | 32.81 | **35.00** |
| Qwen2.5-14B | LongBench v2 | Overall | 33.40 | 34.19 | 32.60 | 33.80 | 34.19 | 34.79 | 32.31 | 34.19 | **34.99** |
| | | Short | 41.11 | **43.33** | 40.56 | 40.00 | 41.11 | 40.56 | 40.56 | 41.11 | 41.11 |
| | | Medium | 31.16 | 30.70 | 32.09 | 33.49 | 33.49 | **34.88** | 29.89 | 33.49 | **34.88** |
| | | Long | 25.00 | **25.93** | 20.37 | 24.07 | 24.07 | 25.00 | 23.95 | 24.07 | 25.00 |
| | LongGenBench | CR | 65.84 | 26.48 | 62.29 | 45.49 | 45.31 | 21.25 | 63.02 | **65.46** | 64.06 |
| | | CR×Acc | 29.35 | 13.89 | **29.79** | 19.76 | 19.65 | 8.25 | 26.08 | 29.39 | 28.83 |

LongBench v2: 503 questions, single run; LongGenBench: 16K-token generation.

### Video and robot episodes (%; 64 frames, our runs on the same questions)

Stored: KV cache kept (GPU + host) as a share of FullKV. Read: tokens attended
per decoding step as a share of the visual tokens. Q / L: Qwen2.5-VL-7B / LLaVA-Video-7B.

| Method | Stored (Q/L) | Read (Q/L) | Video-MME Q | Video-MME L | LVBench Q | LVBench L | W&D Q | W&D L |
|---|---|---|---|---|---|---|---|---|
| FullKV (N) | 100 / 100 | 101.0 / 104.8 | 63.70 | 62.89 | 41.25 | 41.64 | 40.83 | 32.08 |
| *≈2K tokens read per step* | | | | | | | | |
| InfiniPot-V | 16.4 / 19.5 | 16.6 / 20.5 | 58.63 | 58.56 | 36.60 | 38.99 | 37.92 | 24.17 |
| ReKV (keeps full cache) | 100.1 / 100.2 | 14.8 / 15.0 | 59.85 | 59.70 | 37.38 | 39.70 | 39.58 | 24.17 |
| SnapKV (upper bound) | 18.5 / 17.6 | 15.6 / 15.2 | 57.59 | 55.67 | 36.09 | 36.22 | 37.92 | 30.00 |
| Quest | 103.0 / 103.0 | 15.6 / 15.2 | 63.04 | 61.81 | 38.99 | 41.32 | 42.08 | 34.58 |
| FreeKV | 103.0 / 103.0 | 15.6 / 15.2 | 62.52 | 61.44 | 38.48 | 42.16 | 42.92 | 32.50 |
| **OVAL** | **53.1 / 53.0** | 15.6 / 15.2 | 61.93 | 61.37 | 39.70 | 40.87 | 42.50 | 30.83 |

Video-MME and LVBench columns are overall accuracy; W&D is OpenVLA task success
on LIBERO Watch-and-Do (240 episodes). At 52% of their storage, OVAL is within
0.32 and 0.18 points of Quest and FreeKV on average, and both paired 95%
intervals contain zero. Per-duration splits, token-pruning baselines
(VisionZip, InfiniPot-V at 50%), FullKV (N/2) and the 128-frame study are in
the paper.

### Composing with quantization (avg@8, %)

KIVI-style quantization (keys per channel, values per token) of the stored
pages (FreeKV) or of OVAL's bases and coefficients. Memory: stored KiB per KV
head and layer; FullKV stores 1873 (MATH500) and 5050 (AIME24).

| | FreeKV FP16 | 4-bit | 2-bit | OVAL FP16 | 4-bit | 2-bit |
|---|---|---|---|---|---|---|
| MATH500, Qwen-7B: memory | 1923 | 988 | 818 | 1237 | 776 | 691 |
| MATH500, Qwen-7B: avg@8 | 70.00 | 69.25 | 69.25 | 71.75 | 71.50 | 70.75 |
| AIME24, Qwen-14B: memory | 5200 | 2087 | 1521 | 2830 | 1361 | 1090 |
| AIME24, Qwen-14B: avg@8 | 64.17 | 63.75 | 63.33 | 65.42 | 65.00 | 64.58 |

### Decode latency

16K-token AIME24 generation in FreeKV's engine, storing only OVAL's bases: 40
and 47 ms per token on Qwen-7B and Llama-8B, 11% and 12% above FreeKV (36 and
42 ms), and 1.8x and 3.0x faster than ArkVale (72 and 143 ms).

## Citation

```bibtex
@article{shahbazi2026oval,
  title   = {{OVAL}: Output-Aware Local Page Bases for {KV} Cache Compression},
  author  = {Shahbazi, Ashkan and Thrash, Chayne and Kolouri, Soheil},
  journal = {arXiv preprint arXiv:2610.06686},
  year    = {2026}
}
```

## Acknowledgments

OVAL is built on the [FreeKV](https://github.com/sjtu-zhao-lab/FreeKV) harness,
which `setup.sh` clones at a pinned commit and patches.
