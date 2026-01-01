# Long reasoning at k=8 -- ds-r1-qwen-7b

Retrieval+compression, rank 8. Seeds 42-49. Protocol identical in every
cell; only eta varies. `--oval_recon` replaces the paged keys with the
record, so the summary IS the key store rather than an addition to it.

| dataset | arm | k | avg@k | pass@k |
|---|---|---|---|---|
| MATH50 | dense (FullKV) | 8 | 70.75 | 78.00 |
| MATH50 | codec eta=0.0 | 8 | 71.75 | 78.00 |
| MATH50 | codec eta=0.25 | 8 | 71.75 | 78.00 |
| AIME24 | dense (FullKV) | 8 | 55.00 | 83.33 |
| AIME24 | codec eta=0.0 | 8 | 47.08 | 70.00 |
| AIME24 | codec eta=0.5 | 8 | 49.58 | 70.00 |
| GPQA50c | dense (FullKV) | 8 | 34.75 | 64.00 |
| GPQA50c | codec eta=0.0 | 7 | 38.57 | 72.00 |
| GPQA50c | codec eta=0.25 | 8 | 38.25 | 80.00 |

## The published eta* does not reproduce

| dataset | published eta* | paper gap over eta=0 | measured gap here |
|---|---|---|---|
| MATH50 | 0.25 | +3.75 | +0.00 |
| AIME24 | 0.5 | +6.67 | +2.50 |
| GPQA50c | 0.25 | +9.75 | -0.32 |

The full eta grid is therefore being re-measured here rather than
inherited from the paper's ablation table.
