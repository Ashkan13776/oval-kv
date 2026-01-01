# Long reasoning at k=8 -- ds-r1-qwen-14b

Retrieval+compression, rank 8. Seeds 42-49, all 120 cells complete. Protocol
identical in every cell; only eta varies. `--oval_recon` replaces the paged
keys with the record, so the summary IS the key store rather than an addition
to it. No dense control: the hand-off doc takes FullKV from the paper's column.

Run on machine C (2 x RTX PRO 6000 Blackwell 97GB) via
`scripts/supervise_sweep.sh`.

| dataset | arm | k | avg@k | pass@k |
|---|---|---|---|---|
| MATH50 | codec eta=0.0 | 8 | 68.25 | 76.00 |
| MATH50 | codec eta=0.25 | 8 | **69.25** | 76.00 |
| MATH50 | codec eta=0.5 | 8 | 67.50 | 76.00 |
| MATH50 | codec eta=0.75 | 8 | 67.00 | 72.00 |
| MATH50 | codec eta=1.0 | 8 | 67.25 | 74.00 |
| AIME24 | codec eta=0.0 | 8 | 65.00 | 83.33 |
| AIME24 | codec eta=0.25 | 8 | 63.75 | 83.33 |
| AIME24 | codec eta=0.5 | 8 | 58.75 | 80.00 |
| AIME24 | codec eta=0.75 | 8 | 59.58 | 76.67 |
| AIME24 | codec eta=1.0 | 8 | **65.42** | 83.33 |
| GPQA50c | codec eta=0.0 | 8 | **54.00** | 88.00 |
| GPQA50c | codec eta=0.25 | 8 | 53.50 | 78.00 |
| GPQA50c | codec eta=0.5 | 8 | 51.50 | 76.00 |
| GPQA50c | codec eta=0.75 | 8 | 50.75 | 88.00 |
| GPQA50c | codec eta=1.0 | 8 | 52.25 | 76.00 |

pass@k is only comparable at equal k; every row here is k=8.

## The published eta* does not reproduce

Paired over the shared seeds, 95% CI from the per-seed differences:

| dataset | published eta* | paper gap over eta=0 | measured gap | 95% CI | verdict |
|---|---|---|---|---|---|
| MATH500 | 0.5 | +0.25 | -0.75 | [-4.93, +3.43] | cannot reject |
| AIME24 | 0.25 | +7.50 | **-0.88** | [-6.24, +4.49] | **refuted** |
| GPQA | 0.5 | +4.25 | **-2.50** | [-5.56, +0.56] | **refuted** |

MATH500 survives only because its claimed gap (+0.25) is too small to falsify
at this sample size -- the interval spans 8 points.

**eta=0 is the best or statistically tied arm on all three datasets.** The
per-dataset argmax lands on 0.25, 1.0 and 0.0 -- no consistent optimum, which
is what noise looks like rather than a tunable parameter. On AIME24 the two
endpoints tie (eta=0 65.00, eta=1 65.42) while the middle dips to 58.75; if the
output-aware term carried signal, pure key PCA and pure output-only should not
coincide.

This replicates machine A's qwen-7b result on a second model. See
`reasoning_k8_qwen7b.md`.

## Noise floor, for reading the tables above

Seed-to-seed spread within a single arm, which bounds what any eta effect can
claim:

| dataset | n | pts per problem | spread within eta=0 |
|---|---|---|---|
| MATH50 | 50 | 2.00 | 64-74 (10 pts) |
| AIME24 | 30 | 3.33 | 60-73 (13 pts) |
| GPQA50c | 50 | 2.00 | 48-58 (10 pts) |

The eta means span 2.25 / 6.67 / 3.25 points respectively -- at or below the
single-arm seed spread in every case.

Partial rows mislead badly here. AIME24 eta=0.25 read +1.60 over eta=0 at k=5
(first five seeds `+7 +7 0 +4 -10`); the last three came in `-3 -6 -6` and the
final answer is -0.88. Three separate rows showed an early high reading that
decayed as seeds accumulated. Do not read a row below k=8.

## Run notes

**One cell per GPU at this model size.** Three cells died with
`CUDA error: the launch timed out and was terminated`, all while two 37GB cells
were generating 16K-token traces concurrently. Re-run one at a time, each
completed on the first attempt with exit 0. The ~2.5% failure rate was an
artifact of running two-up, not of the inputs -- `GPUS="0"` is the safe setting
even when a second GPU is free.

The sweep does not retry a failed cell: it builds its queue once at startup, so
a crash leaves that cell short and the row at k=7 until a later launch picks it
up. `scripts/finish_stuck_cells.sh` finishes them sequentially.
