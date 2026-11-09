# Qwen2.5-VL-7B on Video-MME (no subtitles), query-agnostic stream protocol

Scored on the 2700 questions present in every arm below (benchmark total 2700).

| arm | questions run | status |
|---|---|---|
| FullKV-64 | 2700 | complete |
| OVAL | 2700 | complete |
| FreeKV | 2700 | complete |
| Quest | 2700 | complete |
| 4-bit keys | 2700 | complete |
| SnapKV† | 2700 | complete |

## Accuracy (%), overall, 95% CI (bootstrap over videos)

| arm | overall (n=2700) |
|---|---|
| FullKV-64 | 63.70 [61.85, 65.56] |
| OVAL | 61.93 [60.00, 63.85] |
| FreeKV | 62.52 [60.63, 64.41] |
| Quest | 63.04 [61.18, 64.93] |
| 4-bit keys | 63.11 [61.26, 64.96] |
| SnapKV† | 57.59 [55.74, 59.48] |

Paired difference vs FullKV-64 (pp), 95% CI; * = CI excludes 0

| arm | overall |
|---|---|
| OVAL | -1.78 [-2.85, -0.70]* |
| FreeKV | -1.19 [-2.22, -0.11]* |
| Quest | -0.67 [-1.63, +0.33] |
| 4-bit keys | -0.59 [-1.33, +0.15] |
| SnapKV† | -6.11 [-7.59, -4.67]* |

## Accuracy (%), duration, 95% CI (bootstrap over videos)

| arm | short (n=900) | medium (n=900) | long (n=900) |
|---|---|---|---|
| FullKV-64 | 75.11 [72.22, 77.89] | 63.78 [60.67, 66.89] | 52.22 [49.00, 55.44] |
| OVAL | 72.56 [69.67, 75.33] | 62.22 [59.00, 65.33] | 51.00 [47.67, 54.33] |
| FreeKV | 73.78 [70.89, 76.44] | 61.22 [57.89, 64.44] | 52.56 [49.33, 55.78] |
| Quest | 74.22 [71.33, 77.00] | 62.22 [59.00, 65.56] | 52.67 [49.44, 55.89] |
| 4-bit keys | 74.78 [72.00, 77.56] | 63.11 [60.00, 66.11] | 51.44 [48.11, 54.67] |
| SnapKV† | 64.33 [61.33, 67.22] | 59.00 [55.67, 62.22] | 49.44 [46.22, 52.78] |

Paired difference vs FullKV-64 (pp), 95% CI; * = CI excludes 0

| arm | short | medium | long |
|---|---|---|---|
| OVAL | -2.56 [-4.22, -1.00]* | -1.56 [-3.67, +0.44] | -1.22 [-2.89, +0.44] |
| FreeKV | -1.33 [-3.00, +0.33] | -2.56 [-4.67, -0.44]* | +0.33 [-1.33, +2.00] |
| Quest | -0.89 [-2.56, +0.67] | -1.56 [-3.44, +0.33] | +0.44 [-1.11, +2.00] |
| 4-bit keys | -0.33 [-1.56, +0.89] | -0.67 [-2.00, +0.67] | -0.78 [-2.11, +0.44] |
| SnapKV† | -10.78 [-13.56, -8.00]* | -4.78 [-7.33, -2.33]* | -2.78 [-5.00, -0.56]* |

†: SnapKV is reported, never ranked.
