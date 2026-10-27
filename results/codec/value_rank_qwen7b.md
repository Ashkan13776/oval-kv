# Value codec rank sweep -- ds-r1-qwen-7b, k=8

Keys and values both compressed on the selected pages. Keys: rank 8 for
scoring and reconstruction (`--oval_rank 8 --oval_recon`), key eta fixed per
task (MATH50 0.25, AIME24 1.0, GPQA50c 0.25). Values: plain PCA of the centred
page values (`--oval_vrank R --vbasis pca`), R in {32, 16, 8, 4, 1}. Sink and
recent tokens exact. Protocol otherwise identical to every codec cell (B=2048,
sink/recent 512, page 32, avgSM, spec_ret 2 / 0.9, T=0.6, top-p 0.95,
max_gen 16384, bf16 records, seeds 42-49). Method and derivation:
[`docs/value-codec.md`](../../docs/value-codec.md).

Reproduce: `scripts/value_rank_sweep.sh`; table: `python analysis/value_rank_table.py`.

```
ds-r1-qwen-7b: keys rank 8 (scoring + reconstruction), values plain PCA; k = complete seeds
diffs in avg@k points, paired 95% CI; '*' = CI excludes 0

-- MATH50  (key eta 0.25)
values                  k    n    avg@k   pass@k   vs lossless (r32)         vs keys-only (cr8)
exact (keys-only cr8)   8   50    71.50    78.00
PCA rank 32             8   50    70.50    76.00   (reference)                -1.00 [-3.50, +1.00]
PCA rank 16             8   50    71.75    78.00    +1.25 [-1.00, +4.00]      +0.25 [-1.25, +2.00]
PCA rank 8              8   50    70.25    78.00    -0.25 [-3.00, +2.75]      -1.25 [-3.75, +0.50]
PCA rank 4              8   50    69.50    76.00    -1.00 [-3.00, +1.00]      -2.00 [-4.25, +0.00]
PCA rank 1              8   50    60.75    72.00    -9.75 [-16.50, -4.00] *  -10.75 [-18.00, -4.50] *

-- AIME24  (key eta 1.0)
values                  k    n    avg@k   pass@k   vs lossless (r32)         vs keys-only (cr8)
exact (keys-only cr8)   8   30    52.50    80.00
PCA rank 32             8   30    47.92    80.00   (reference)                -4.58 [-10.42, +1.25]
PCA rank 16             8   30    48.75    76.67    +0.83 [-4.58, +6.25]      -3.75 [-11.25, +3.33]
PCA rank 8              8   30    46.67    73.33    -1.25 [-7.08, +4.58]      -5.83 [-14.58, +1.67]
PCA rank 4              8   30    40.00    66.67    -7.92 [-14.17, -2.50] *  -12.50 [-22.08, -3.33] *
PCA rank 1              8   30    21.25    33.33   -26.67 [-36.25, -17.50] * -31.25 [-42.08, -20.83] *

-- GPQA50c  (key eta 0.25)
values                  k    n    avg@k   pass@k   vs lossless (r32)         vs keys-only (cr8)
exact (keys-only cr8)   8   50    38.25    80.00
PCA rank 32             8   50    36.25    72.00   (reference)                -2.00 [-6.50, +2.75]
PCA rank 16             8   50    34.00    62.00    -2.25 [-6.50, +1.75]      -4.25 [-8.50, +0.00]
PCA rank 8              8   50    33.75    66.00    -2.50 [-7.00, +1.75]      -4.50 [-8.50, -0.50] *
PCA rank 4              8   50    33.50    58.00    -2.75 [-7.25, +1.50]      -4.75 [-8.75, -1.00] *
PCA rank 1              8   50    21.75    46.00   -14.50 [-20.50, -8.50] *  -16.50 [-22.75, -10.75] *
```

## Reading it

- **Reference.** Value rank 32 reconstructs exactly at page size 32 (a centred
  page has rank <= 31; 0.3% residual is bf16 rounding of the record), and it
  ran in the same sweep, on the same machine and software stack, with the same
  keys. It is the reference that isolates value compression. The keys-only
  column (`_cr8`, exact values) comes from the decoupled-rank ablation run on
  a different stack (torch 2.5.1/cu121 vs 2.8/cu128); it sits 1-5 points above
  even the lossless rank-32 run, so that column carries stack/seed noise of
  that size and should not be read as a value-codec effect.
- **Rank 16 and rank 8 are free.** Within noise of lossless on all three tasks
  (largest gap -2.50, every CI includes 0).
- **Rank 4 costs on AIME24 only:** -7.92 [-14.17, -2.50]. MATH50 -1.00 and
  GPQA50c -2.75 are not distinguishable from 0. AIME24 has the longest traces,
  so the largest share of attention goes through compressed pages.
- **Rank 1 fails everywhere:** -9.75 / -26.67 / -14.50.
- **The AIME24 gap to dense is the key codec's, not the value codec's.** Dense
  is 55.00 / 83.33 (machine A); lossless values still give 47.92.
- n=50 (30 for AIME24) problems x 8 seeds; the CIs are paired over problems,
  which removes problem difficulty. Several ranks are compared, so treat a
  borderline interval with care; the rank-4 AIME24 and rank-1 effects are far
  from the boundary.

Environment: 4 x RTX A6000 (sm_86), torch 2.8.0+cu128, transformers 4.45.2,
flash-attn 2.8.3, flashinfer 0.2.4 (JIT), `TORCH_CUDA_ARCH_LIST=8.6`.
