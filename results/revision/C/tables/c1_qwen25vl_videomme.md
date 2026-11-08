# Qwen2.5-VL-7B on Video-MME (no subtitles), query-agnostic stream protocol

Scored on the 2700 questions present in every arm below (benchmark total 2700).

| arm | questions run | status |
|---|---|---|
| FullKV-64 | 2700 | complete |
| OVAL | 2700 | complete |
| FreeKV | 2700 | complete |

## Accuracy (%), overall, 95% CI (bootstrap over videos)

| arm | overall (n=2700) |
|---|---|
| FullKV-64 | 63.70 [61.85, 65.56] |
| OVAL | 61.93 [60.00, 63.85] |
| FreeKV | 62.52 [60.63, 64.41] |

Paired difference vs FullKV-64 (pp), 95% CI; * = CI excludes 0

| arm | overall |
|---|---|
| OVAL | -1.78 [-2.85, -0.70]* |
| FreeKV | -1.19 [-2.22, -0.11]* |

## Accuracy (%), duration, 95% CI (bootstrap over videos)

| arm | short (n=900) | medium (n=900) | long (n=900) |
|---|---|---|---|
| FullKV-64 | 75.11 [72.22, 77.89] | 63.78 [60.67, 66.89] | 52.22 [49.00, 55.44] |
| OVAL | 72.56 [69.67, 75.33] | 62.22 [59.00, 65.33] | 51.00 [47.67, 54.33] |
| FreeKV | 73.78 [70.89, 76.44] | 61.22 [57.89, 64.44] | 52.56 [49.33, 55.78] |

Paired difference vs FullKV-64 (pp), 95% CI; * = CI excludes 0

| arm | short | medium | long |
|---|---|---|---|
| OVAL | -2.56 [-4.22, -1.00]* | -1.56 [-3.67, +0.44] | -1.22 [-2.89, +0.44] |
| FreeKV | -1.33 [-3.00, +0.33] | -2.56 [-4.67, -0.44]* | +0.33 [-1.33, +2.00] |

†: SnapKV is reported, never ranked.
