# 1. Retrieval: OVAL page scoring and the eta sweeps

Piece 1 of 5. Next: [2. key compression](2-key-compression.md). Overview: [docs/README.md](README.md).

## What it is

Each page of 32 tokens stores a rank-r record of its keys per KV head: a mean
`mu`, a basis `B` (d x r) and coefficients `C` (32 x r). The basis is the top-r
eigenvectors of

```
M_eta = (1 - eta) * X^T X / tr(X^T X)  +  eta * X^T Y G Y^T X / tr(X^T Y G Y^T X)
```

with `X`, `Y` the page's centred keys and values and `G = W_O^T W_O` summed over
the KV head's GQA group. `eta = 0` is plain key PCA; `eta = 1` keeps the
directions that matter most for the attention output. A page is scored by
reconstructing its per-token logits in the basis and taking their logsumexp,
its log attention mass under the rank-r record. The top pages within the
budget are attended; everything else (GQA combine, top-k, speculative
retrieval and its correction) is FreeKV's.

Code: `oval/page_basis.py` (`build_summary`, `oval_sel`), installed into the
FreeKV harness by `setup.sh`; harness flags `--method spec_ret --page_rep oval
--oval_rank 8 --eta E`.

In this piece alone, the selected pages are attended with their **exact** keys
and values: the records are an index, not yet the cache.

## Experiments

### Long reasoning: eta sweeps at k = 8

MATH50 / AIME24 / GPQA50c, eta in {0, 0.25, 0.5, 0.75, 1}, seeds 42-49. These
cells were run with key compression on as well (`--oval_recon`, piece 2), so
they measure retrieval + rank-8 key reconstruction at each shared eta.

| model | summary | best shared eta per task (MATH50 / AIME24 / GPQA50c) |
|---|---|---|
| R1-Qwen-7B | [`eta_sweep_qwen7b.md`](../results/reasoning/eta_sweep_qwen7b.md) (partial grid, cells not in the repo) | 0.0-0.25 / 0.5 / 0.0 |
| R1-Llama-8B | [`eta_sweep_llama8b.md`](../results/reasoning/eta_sweep_llama8b.md) (120 cells) | 0.75 / 0.75 / 0.25 |
| R1-Qwen-14B | [`eta_sweep_qwen14b.md`](../results/reasoning/eta_sweep_qwen14b.md) (120 cells) | 0.25 / 1.0 / 0.0 |

Finding: **no single eta is best**. The argmax moves across tasks and models,
and within a task the spread over eta is about the size of the seed noise.
The paper's published per-task eta* did not reproduce on R1-Qwen-7B or
R1-Qwen-14B (`eta_sweep_qwen7b.md`, `eta_sweep_qwen14b.md`). This is what piece 4 addresses: choose eta per
(layer, KV head) from the model's own errors instead of sweeping one value.

Run: `MODEL=<model> scripts/oval_reasoning_sweep.sh` under
`scripts/supervise_sweep.sh`; score with `analysis/score_avg_pass_at_k.py`.
Splitting a sweep across machines: [harness/parallel-runs.md](harness/parallel-runs.md).

### LongBench v2, LongGenBench, InfiniteBench

The paper tables in the top-level README (Table 2) are retrieval only (exact
keys and values) on Llama-3.1-8B, Qwen-2.5-7B and Qwen-2.5-14B, against the
FreeKV paper's baselines; `eta*` there is the best of five etas per cell.

| what | script | output |
|---|---|---|
| LongBench v2 eta sweep (Qwen-2.5-14B) | `scripts/lb2_eta_sweep.sh`, `lb2_rank_sweep.sh`, `lb2_controls.sh` | `analysis/score_accuracy.py`, `analysis/all_tables.py`; numbers in `results/paper_tables.md` |
| InfiniteBench, FreeKV vs OVAL in one harness | `scripts/ib_freekv.sh`, `ib_full_refix.sh` | `analysis/all_tables.py` |
| Table 3 protocol in FreeKV's harness | `scripts/freekv_sweep.sh` | |

### Decode latency

| what | script | output |
|---|---|---|
| end-to-end latency, FreeKV paper Fig. 6 setup | `scripts/fig6_latency_all.sh` | `results/fig6_latency/*.log` |
| latency on LongBench v2 prompts | `scripts/lb2_latency_mvp.sh` | `results/lb2_latency/*.log` |
| accuracy vs latency | `analysis/plot_acc_vs_latency.py` | |

The latency logs predate the rename to OVAL; their banners say `locks`
(`results/README.md`).

### Matched-budget comparison: the 26-arm Pareto front

[pareto-baselines.md](pareto-baselines.md): MATH50, R1-Qwen-7B, B = 2048, the
OVAL arms against 19 KV-retrieval scorers and eviction baselines, with one
memory definition for all. Figure: `results/figures/pareto_math50_b2048.pdf`.
