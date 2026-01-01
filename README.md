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
| `docs/` | protocol notes, deviations, and upstream bugs found |

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

See [`results/all_tables.txt`](results/all_tables.txt). Headline, on
Llama-3.1-8B at `B = 2048` over 11 InfiniteBench tasks:

| method | acc | decode latency |
|---|---:|---:|
| FreeKV | 35.53 | 22.98 ms |
| ArkVale | 35.94 | 76.86 ms |
| RaaS | 35.98 | 27.42 ms |
| **OVAL** (r=8, eta=0.25) | **37.42** | 28.61 ms |
| FullKV (dense) | 42.93 | — |

Read [`docs/open-questions.md`](docs/open-questions.md) before quoting these.
In short: seven of the eleven tasks have **no headroom** — FullKV does not beat
the best sparse arm on them, so they cannot rank selection quality — and
essentially all of OVAL's margin comes from `kv_retrieval`, where it scores
14.0 against 0.0 for all three competitors. On LongBench v2 the whole `eta`
range moves four questions out of 503.

## Upstream bugs found

Documented in [`docs/freekv-port.md`](docs/freekv-port.md), with repros:

- **the KV cache is never written in bf16** on FreeKV's `--method full` path,
  so their FullKV baseline silently emits one correct token and then garbage;
- a **race in the speculative-retrieval thread pool** that corrupts output;
- a fixed 64 MiB FlashInfer workspace too small for 16K-token generation;
- an eviction-indexing failure at large KV × batch × length.
