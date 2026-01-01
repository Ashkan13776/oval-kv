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

## What this repo contains

| path | |
|---|---|
| `oval/` | the method, plus the modules added to FreeKV |
| `patches/freekv.patch` | our ~500-line diff against upstream FreeKV |
| `scripts/` | one script per experiment |
| `analysis/` | scoring and figure generation |
| `results/`, `figures/` | measured numbers and rendered figures |
| `docs/` | protocol and deviation notes |

FreeKV and InfiniteBench are **not** vendored; `setup.sh` clones them at a
pinned commit and applies the patch.

## Setup

```bash
./setup.sh
export KV_ROOT=$PWD
export KV_PY=$(which python)     # torch >= 2.8, CUDA 12.8
./scripts/build_freekv.sh
```

## Reproducing

```bash
./scripts/ib_freekv.sh              # InfiniteBench: FreeKV vs OVAL
./scripts/baselines_for_tradeoff.sh # + ArkVale, RaaS, FullKV
./scripts/lb2_eta_sweep.sh          # LongBench v2 eta sweep
./scripts/fig6_latency_all.sh       # end-to-end decode latency
python analysis/all_tables.py       # score whatever is on disk
python analysis/plot_acc_vs_latency.py
```

`analysis/all_tables.py` scores from the raw prediction files every time, so
the tables cannot drift from a stale summary.

## Results

Accuracy tables: [`results/accuracy_tables.txt`](results/accuracy_tables.txt);
machine-readable under `results/longbench2_accuracy/` and
`results/longgenbench_accuracy/`. Raw generations (293 MB for LongGenBench) are
not committed -- regenerate with `scripts/`, then `analysis/score_accuracy.py`.

Settings are the FreeKV paper's: page size 32, B = 2048, S = W = 128 / tau = 0.8
(LongBench v2, greedy, inputs truncated to 64K) and S = W = 512 / tau = 0.9
(LongGenBench, temperature 0.95, top-p 0.95, 16K generation). Baseline columns
are that paper's reported numbers, not our reruns.

### LongBench v2 -- Overall (503 questions)

| model | eta=0 | 0.25 | 0.5 | 0.75 | 1.0 | FullKV | FreeKV | best baseline |
|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | 29.42 | 29.42 | 29.82 | 29.82 | 29.82 | 29.22 | 29.22 | 28.63 |
| Qwen-2.5-7B | 27.63 | 28.03 | 27.63 | 28.03 | 28.03 | 27.44 | 26.84 | 27.63 |

The gain concentrates in the **long** bucket: Llama 25.00-25.93 vs 23.15 dense;
Qwen 25.93 at every eta vs 20.37 dense. The **medium** bucket is slightly weaker
than dense on both, so this is a trade toward long contexts, not a uniform win.

### LongGenBench -- CR x Acc (400 prompts, 16K generation)

| model | eta=0 | 0.25 | 0.5 | 0.75 | 1.0 | FullKV | FreeKV |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | 27.24 | 27.26 | 27.23 | 27.28 | 28.50 | 26.82 | 27.62 |
| Qwen-2.5-7B | 32.38 | 31.91 | 31.86 | 31.57 | 32.28 | 31.09 | 32.81 |
| Qwen-2.5-14B | 28.61 | 27.68 | 26.94 | 28.64 | 27.30 | 29.35 | 29.39 |

### On eta

`eta` is inert. Across five benchmark families -- InfiniteBench (66 cells),
reasoning (36 cells), LongBench v2 (10), LongGenBench CR and CR x Acc (15 each)
-- the spread over `eta` in {0, 0.25, 0.5, 0.75, 1} never exceeds ~2.9 points
and shows no monotone trend. Since `eta = 0` is plain key PCA, the rank-`r`
page basis is what carries the result; the output-aware tilt is not supported
by any measurement here. We report it because it is the hypothesis we set out
to test.

### Scope

OVAL replaces the page *digest* used for selection. The retrieved pages are
attended with their **original** keys, so this buys retrieval quality, not
cache compression -- in the accuracy harness the full KV stays resident and the
bases are added on top. Using the rank-8 basis to *approximate* attention
instead costs 20.1% relative error in the attention output (it captures 76% of
centred-key energy), while the same basis retains 99.6% of true top-32
attention mass for ranking: a good index, a poor codec.
