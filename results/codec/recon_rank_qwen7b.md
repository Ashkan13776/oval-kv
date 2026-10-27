# Decoupling the scoring rank from the codec rank -- ds-r1-qwen-7b

The OVAL record does two jobs at once: it **scores** pages (which to attend)
and, under `--oval_recon`, it **reconstructs** the keys of the pages that win.
Until now both used the same rank, so a loss at rank 8 could not be attributed
to either job.

`--oval_recon_rank` separates them. `--oval_rank` keeps only the scoring rank;
records are stored at `max(score, recon)` and each consumer takes its own
top-r truncation. The basis is eigenvalue-ordered, so a truncation IS the
rank-r record, not an approximation of it -- truncating a rank-32 record to
top-8 reproduces a rank-8 build to 0.00e+00 at r=4/8/16 (bf16 rounding only at
r=1). When scoring uses fewer columns than are stored it gets its own
contiguous slab, because the `oval_sel` CUDA kernel cannot take a strided rank
dimension.

## Settings

Scoring rank fixed at **8**; reconstruction rank swept over **1, 4, 8, 16, 32**;
8 seeds (42-49), so k=8. Protocol is the paper's reasoning protocol, identical
in every cell -- only the reconstruction rank varies within a task.

```
--method spec_ret --page_rep oval --oval_rank 8 --oval_recon
--oval_recon_rank {1,4,8,16,32}
--GQA_policy avgSM --spec_ret_steps 2 --spec_ret_corr 0.9
--temperature 0.6 --top_p 0.95 --max_gen 16384
--budget 2048 --sink 512 --recent 512 --page_size 32
OVAL_QUANT=0  KVC_MAX_TOKENS=32768
```

eta is **per task**, at the value requested for this run:

| task | eta |
|---|---|
| MATH50 | 0.25 |
| AIME24 | 1.0 |
| GPQA50c | 0.25 |

Driver: `scripts/codec_recon_rank_sweep.sh` (`SCORE_RANK`, `RECON_RANKS`,
`ETAMAP`). 120 cells, 132h on 4 x A6000.

Two notes on the sweep itself. **rank 32 is clamped to `page_size-1` = 31**,
where reconstruction is exact -- that arm is the exact-keys ceiling, and its
bf16 record (~10.2 KB/page/head) is larger than the raw 8 KB key page it
replaces, so it is a ceiling and not a shippable configuration. One cell
(AIME24 seed 45, cr32) died of a CUDA OOM when the scheduler reclaimed a GPU
35s after the previous cell exited and the card had not released its context;
it was requeued and completed, so every arm below is k=8.

## Results, avg@8 / pass@8

| recon rank | MATH50 (eta=0.25) | AIME24 (eta=1.0) | GPQA50c (eta=0.25) |
|---|---|---|---|
| 1 | 64.25 / 72.00 | 24.17 / 43.33 | 23.75 / 48.00 |
| 4 | 69.25 / 76.00 | 37.92 / 63.33 | 33.00 / 56.00 |
| **8** | **71.50 / 78.00** | **52.50 / 80.00** | **38.25 / 80.00** |
| 16 | 71.75 / 78.00 | 55.42 / 83.33 | 37.50 / 70.00 |
| 32 (exact) | 69.50 / 76.00 | 52.92 / 80.00 | 40.00 / 82.00 |
| ANOVA | F=8.77, p=5.2e-05 | F=53.55, p=1.9e-14 | F=16.34, p=1.2e-07 |

Paired against the rank-8 arm (same seeds), avg@8:

| recon rank | MATH50 | AIME24 | GPQA50c |
|---|---|---|---|
| 1 | -7.25 [-10.46, -4.04] p=0.0011 | -28.33 [-31.98, -24.68] p<0.0001 | -14.50 [-20.27, -8.73] p=0.0006 |
| 4 | -2.25 [-6.29, +1.79] p=0.229 | -14.58 [-21.40, -7.77] p=0.0015 | -5.25 [-11.17, +0.67] p=0.074 |
| 16 | +0.25 [-2.90, +3.40] p=0.857 | +2.92 [-4.14, +9.97] p=0.361 | -0.75 [-7.31, +5.81] p=0.795 |
| 32 | -2.00 [-6.64, +2.64] p=0.342 | +0.42 [-4.62, +5.45] p=0.851 | +1.75 [-3.57, +7.07] p=0.462 |

## What it says

**Rank 8 is the right operating point on all three tasks.** Every arm above it
-- including exact reconstruction -- is statistically indistinguishable from
rank 8 (six comparisons, all p >= 0.34). Exact keys buy nothing over the rank-8
codec, which is the strongest form of the claim that the codec is not what is
costing accuracy.

**Below rank 8 the penalty scales with task difficulty**, and steeply:

| task | rank 1 -> 8 |
|---|---|
| MATH50 | +7.25 |
| GPQA50c | +14.50 |
| AIME24 | +28.33 |

So the record's rank is set by the **reconstruction** job on hard tasks and is
slack on easy ones; selection is satisfied well below 8 on MATH50 and nowhere
near it on AIME24.

## What this experiment does NOT show

**It is diagnostic, not a memory result.** With scoring pinned at 8, storage is
`max(8, r)`, so ranks 1, 4 and 8 occupy *identical* bytes (2816 B/page/kv head
in bf16) and rank 16 costs more. Lowering the reconstruction rank buys no
memory at all here -- only accuracy loss. The memory-saving configurations are
the diagonal (r, r), which `scripts/codec_grid.sh` covers.

**eta differs per task**, so absolute accuracy is not comparable across the
three columns. The within-task rank curves are the result.
