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
