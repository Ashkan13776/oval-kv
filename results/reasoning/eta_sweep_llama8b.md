# Long reasoning at k=8 -- ds-r1-llama-8b, eta sweep

Retrieval + key compression, rank 8 (`--oval_rank 8 --oval_recon`), exact
values. Seeds 42-49, all 120 cells complete (machine B). Protocol identical in
every cell; only eta varies. Scored with
`python analysis/score_avg_pass_at_k.py --models ds-r1-llama-8b`.

| dataset | eta | k | avg@k | pass@k |
|---|---|---|---|---|
| MATH50 | 0.0 | 8 | 66.75 | 74.00 |
| MATH50 | 0.25 | 8 | 65.50 | 72.00 |
| MATH50 | 0.5 | 8 | 64.75 | 76.00 |
| MATH50 | 0.75 | 8 | **67.25** | 76.00 |
| MATH50 | 1.0 | 8 | 65.00 | 78.00 |
| AIME24 | 0.0 | 8 | 40.83 | 73.33 |
| AIME24 | 0.25 | 8 | 38.33 | 66.67 |
| AIME24 | 0.5 | 8 | 45.42 | 73.33 |
| AIME24 | 0.75 | 8 | **45.83** | 73.33 |
| AIME24 | 1.0 | 8 | 43.75 | 73.33 |
| GPQA50c | 0.0 | 8 | 35.50 | 68.00 |
| GPQA50c | 0.25 | 8 | **39.25** | 72.00 |
| GPQA50c | 0.5 | 8 | 37.00 | 70.00 |
| GPQA50c | 0.75 | 8 | 37.50 | 70.00 |
| GPQA50c | 1.0 | 8 | 38.00 | 72.00 |

The best shared eta differs per task (0.75, 0.75, 0.25) and the spread
within a task (2.5-7.5 points) is of the size of the per-seed noise. With
qwen-14b (best 0.25, 1.0, 0.0; `eta_sweep_qwen14b.md`) this is the evidence
that one global eta is not the right object, which motivated choosing eta
per (layer, KV head) (`docs/4-per-head-eta.md`).
