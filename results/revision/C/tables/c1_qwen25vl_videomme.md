# Qwen2.5-VL-7B on Video-MME (no subtitles), query-agnostic stream protocol

Scored on the 2700 questions present in every arm below (benchmark total 2700).

| arm | questions run | status |
|---|---|---|
| FullKV-64 | 2700 | complete |
| OVAL | 2700 | complete |

## Accuracy (%), overall, 95% CI (bootstrap over videos)

| arm | overall (n=2700) |
|---|---|
| FullKV-64 | 63.70 [61.85, 65.56] |
| OVAL | 61.93 [60.00, 63.85] |

Paired difference vs FullKV-64 (pp), 95% CI; * = CI excludes 0

| arm | overall |
|---|---|
| OVAL | -1.78 [-2.85, -0.70]* |

## Accuracy (%), duration, 95% CI (bootstrap over videos)

| arm | short (n=900) | medium (n=900) | long (n=900) |
|---|---|---|---|
| FullKV-64 | 75.11 [72.22, 77.89] | 63.78 [60.67, 66.89] | 52.22 [49.00, 55.44] |
| OVAL | 72.56 [69.67, 75.33] | 62.22 [59.00, 65.33] | 51.00 [47.67, 54.33] |

Paired difference vs FullKV-64 (pp), 95% CI; * = CI excludes 0

| arm | short | medium | long |
|---|---|---|---|
| OVAL | -2.56 [-4.22, -1.00]* | -1.56 [-3.67, +0.44] | -1.22 [-2.89, +0.44] |

†: SnapKV is reported, never ranked.
