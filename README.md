# OVAL — Output-aware Value-Aligned page basis for KV-cache page selection

Page-selecting KV-cache methods summarise each page of keys with a cheap
digest and rank pages by a bound on their attention score. FreeKV and Quest use
an elementwise min/max box; ArkVale uses a mean-absolute-deviation band. OVAL
replaces the digest with a **rank-`r` basis of the page's keys**, tilted toward
the directions that matter for the attention *output*:

```
M_eta = (1 - eta) * XᵀX / tr(XᵀX)  +  eta * XᵀYGYᵀX / tr(XᵀYGYᵀX)
```

`X` are the page's centred keys, `Y` its centred values, and `G = WᵀW` the
value-space metric from the attention output projection, summed over each KV
head's GQA group. The two terms are mixed at matched trace, so `eta` is
scale-free. `eta = 0` is plain key PCA; `eta = 1` is output-only.

A page is scored by reconstructing its per-token logits in that basis and
taking their logsumexp — the page's log attention mass under the rank-`r`
truncation. Everything downstream (GQA combine, top-k, speculative retrieval,
correction) is FreeKV's, untouched.

Implementation: [`oval/page_basis.py`](oval/page_basis.py).

## Start here

OVAL keeps a rank-r record of every KV page and uses it in four
complementary ways. Each piece was first tested on its own; the next run
combines them.

| # | piece | what | doc |
|---|---|---|---|
| 1 | retrieval | score pages with rank-8 key records, attend the top pages | [docs/1-retrieval.md](docs/1-retrieval.md) |
| 2 | key compression | rebuild the selected pages' keys from those records (`--oval_recon`) | [docs/2-key-compression.md](docs/2-key-compression.md) |
| 3 | value compression | rebuild the selected pages' values from rank-16 PCA records (`--oval_vrank 16`) | [docs/3-value-compression.md](docs/3-value-compression.md) |
| 4 | per-head eta | choose eta per (layer, KV head) with rule 1, a closed form fitted on calibration traces (`--eta_table`) | [docs/4-per-head-eta.md](docs/4-per-head-eta.md) |
| 5 | **full pipeline** | all four together on long reasoning (MATH50, AIME24, GPQA50c; R1-Qwen-7B, R1-Llama-8B, R1-Qwen-14B; k = 8) -- **next run** | [docs/5-full-pipeline.md](docs/5-full-pipeline.md), `scripts/full_pipeline.sh` |

Pieces 2-4 exist on long reasoning only; on long context (LongBench v2,
LongGenBench) only retrieval has been run, and transferring the rest is
under way. Agents: start at [docs/agents/README.md](docs/agents/README.md).
[docs/README.md](docs/README.md) is the full map. The paper tables under
"Results" below are piece 1 alone (exact keys and values, best of five etas).

## What this repo contains

| path | |
|---|---|
| `oval/` | the method, plus the modules added to FreeKV |
| `patches/freekv.patch` | our diff against upstream FreeKV (OVAL plus 19 baseline scorers) |
| `scripts/`, `analysis/` | experiment drivers and scoring / figures, mapped to the pieces in [scripts/README.md](scripts/README.md) |
| `results/` | measured numbers; [results/README.md](results/README.md) |
| `docs/` | one doc per piece plus harness notes; [docs/README.md](docs/README.md) |
| `archive/` | finished side studies (eta rules 2-6, the LOCKS reproduction notes); [archive/README.md](archive/README.md) |

FreeKV and InfiniteBench are **not** vendored; `setup.sh` clones them at a
pinned commit and applies the patch.

## Setup

```bash
./setup.sh
export KV_ROOT=$PWD
export KV_PY=$(which python)     # torch >= 2.8, CUDA 12.8
./scripts/build_freekv.sh         # only for the latency runs (efficient engine)
python analysis/test_page_basis.py && python analysis/test_eta_rules.py
```

## Reproducing

```bash
MODELS="ds-r1-qwen-7b ds-r1-llama-8b" ./scripts/full_pipeline.sh   # the full pipeline (piece 5)
MODEL=ds-r1-llama-8b ./scripts/supervise_sweep.sh                  # long-reasoning eta sweep (piece 1)
./scripts/oval_recon_rank_sweep.sh                                 # key reconstruction rank (piece 2)
./scripts/value_rank_sweep.sh                                      # value PCA rank (piece 3)
./scripts/ib_freekv.sh                                             # InfiniteBench: FreeKV vs OVAL
./scripts/lb2_eta_sweep.sh                                         # LongBench v2 eta sweep
./scripts/fig6_latency_all.sh                                      # end-to-end decode latency
python analysis/all_tables.py                                      # score whatever is on disk
```

Every scorer reads the raw prediction files, so tables cannot drift from a
stale summary. The full list of drivers is in [scripts/README.md](scripts/README.md).

## Results

All numbers below are as reported in the paper. `eta*` denotes the best
evaluated mixing parameter for each setting, selected over
`eta` in {0, 0.25, 0.5, 0.75, 1}. Baseline columns are the FreeKV paper's
reported numbers. FullKV is excluded from ranking.

Settings follow that paper: page size 32, B = 2048, S = W = 128 / tau = 0.8
(LongBench v2, greedy, inputs truncated to 64K) and S = W = 512 / tau = 0.9
(LongGenBench, temperature 0.95, top-p 0.95, 16K generation).

### Table 1 — Long reasoning (avg@k, k=8)

Average over MATH500, AIME24 and GPQA for each model. Per-benchmark rows and
pass@k are in [`results/paper_tables.md`](results/paper_tables.md).

| Model | Full | Razor | RaaS | Quest | ArkVale | ShadowKV | InfiniGen | FreeKV | **eta\*** |
|---|---|---|---|---|---|---|---|---|---|
| R1-Llama-8B | 51.36 | 41.58 | 44.08 | 47.81 | 49.64 | 44.33 | 46.61 | 51.83 | **55.69** |
| R1-Qwen-7B | 54.72 | 44.89 | 47.72 | 51.42 | 50.14 | 47.33 | 49.61 | 54.14 | **57.83** |
| R1-Qwen-14B | 63.25 | 43.58 | 52.58 | 58.94 | 60.42 | 58.00 | 55.00 | 62.56 | **66.42** |

### Table 2 — LongBench v2 by context length, and LongGenBench

| Model | Benchmark | Metric | Full | Razor | RaaS | Quest | ArkVale | ShadowKV | InfiniGen | FreeKV | **eta\*** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | LongBench v2 | Overall | 29.22 | 27.44 | 28.23 | 28.43 | 28.63 | 25.45 | 28.56 | 29.22 | **29.82** |
| Llama-3.1-8B | LongBench v2 | Short | 34.44 | 33.89 | 33.89 | 33.33 | 33.89 | 32.78 | 32.28 | **35.00** | **35.00** |
| Llama-3.1-8B | LongBench v2 | Medium | 27.91 | 25.12 | 26.51 | 27.44 | 26.98 | 22.79 | 26.05 | 27.44 | **28.84** |
| Llama-3.1-8B | LongBench v2 | Long | 23.15 | 21.30 | 22.02 | 22.22 | 23.15 | 18.52 | 24.07 | 23.15 | **25.93** |
| Llama-3.1-8B | LongGenBench | CR | 80.03 | 35.90 | 76.63 | 78.03 | 39.36 | **79.28** | 76.68 | 78.03 | 74.71 |
| Llama-3.1-8B | LongGenBench | CR×Acc | 26.82 | 12.20 | 26.00 | 27.71 | 10.36 | **30.66** | 26.21 | 27.62 | 28.50 |
| Qwen-2.5-7B | LongBench v2 | Overall | 27.44 | 25.25 | 26.24 | 27.63 | 26.84 | 25.84 | 26.44 | 26.84 | **28.03** |
| Qwen-2.5-7B | LongBench v2 | Short | 36.11 | 32.78 | 35.56 | **36.67** | 36.11 | 32.22 | 32.22 | 34.44 | 36.11 |
| Qwen-2.5-7B | LongBench v2 | Medium | 23.72 | 21.86 | 21.86 | 22.79 | 22.33 | 20.00 | **23.26** | 22.33 | 22.79 |
| Qwen-2.5-7B | LongBench v2 | Long | 20.37 | 19.44 | 19.44 | 22.22 | 20.37 | **26.85** | 23.15 | 23.15 | 25.93 |
| Qwen-2.5-7B | LongGenBench | CR | 79.56 | 42.13 | 77.65 | 62.89 | 75.91 | 35.49 | 72.96 | 76.93 | **78.54** |
| Qwen-2.5-7B | LongGenBench | CR×Acc | 31.09 | 21.48 | **34.40** | 25.96 | 31.79 | 11.43 | 27.67 | 32.81 | 32.38 |
| Qwen-2.5-14B | LongBench v2 | Overall | 33.40 | 34.19 | 32.60 | 33.80 | 34.19 | **34.79** | 32.31 | 34.19 | 33.00 |
| Qwen-2.5-14B | LongBench v2 | Short | 41.11 | **43.33** | 40.56 | 40.00 | 41.11 | 40.56 | 40.56 | 41.11 | 41.11 |
| Qwen-2.5-14B | LongBench v2 | Medium | 31.16 | 30.70 | 32.09 | 33.49 | 33.49 | **34.88** | 29.89 | 33.49 | 30.23 |
| Qwen-2.5-14B | LongBench v2 | Long | 25.00 | **25.93** | 20.37 | 24.07 | 24.07 | 25.00 | 23.95 | 24.07 | **25.93** |
| Qwen-2.5-14B | LongGenBench | CR | 65.84 | 26.48 | 62.29 | 45.49 | 45.31 | 21.25 | 63.02 | **65.46** | 63.92 |
| Qwen-2.5-14B | LongGenBench | CR×Acc | 29.35 | 13.89 | **29.79** | 19.76 | 19.65 | 8.25 | 26.08 | 29.39 | 28.64 |

### On `eta*`

`eta*` is a per-cell maximum over five eta values, and the maximising eta
differs across metrics within a model, so a whole row is not produced by a
single fixed configuration.

### Scope of the tables above

In these tables OVAL replaces only the page *digest* used for selection: the
retrieved pages are attended with their **original** keys and values, so they
measure retrieval quality (piece 1), not cache compression. Pieces 2-4 and
the full pipeline are in "Start here". Using the rank-8 basis to *approximate* attention
instead costs 20.1% relative error in the attention output (it captures 76% of
centred-key energy), while the same basis retains 99.6% of true top-32
attention mass for ranking: a good index, a poor reconstruction.
