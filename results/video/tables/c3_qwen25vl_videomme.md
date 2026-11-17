# Qwen2.5-VL-7B on Video-MME (no subtitles), query-agnostic stream protocol

Scored on the 2700 questions present in every arm below (benchmark total 2700).

| arm | questions run | status |
|---|---|---|
| FullKV-64 | 2700 | complete |
| FullKV-32 | 2700 | complete |
| OVAL | 2700 | complete |
| VisionZip 50% | 2700 | complete |
| VisionZip 25% | 2700 | complete |
| VisionZip 50% + OVAL | 2700 | complete |
| InfiniPot-V (mem-matched) | 2700 | complete |
| InfiniPot-V (2k) | 2700 | complete |
| ReKV | 2700 | complete |

## Accuracy (%), overall, 95% CI (bootstrap over videos)

| arm | overall (n=2700) |
|---|---|
| FullKV-64 | 63.70 [61.85, 65.56] |
| FullKV-32 | 61.67 [59.78, 63.59] |
| OVAL | 61.93 [60.00, 63.85] |
| VisionZip 50% | 62.93 [61.04, 64.81] |
| VisionZip 25% | 61.78 [59.89, 63.70] |
| VisionZip 50% + OVAL | 61.70 [59.78, 63.63] |
| InfiniPot-V (mem-matched) | 62.78 [60.89, 64.70] |
| InfiniPot-V (2k) | 58.63 [56.70, 60.52] |
| ReKV | 59.85 [57.96, 61.74] |

Paired difference vs FullKV-64 (pp), 95% CI; * = CI excludes 0

| arm | overall |
|---|---|
| FullKV-32 | -2.04 [-3.33, -0.78]* |
| OVAL | -1.78 [-2.85, -0.70]* |
| VisionZip 50% | -0.78 [-1.63, +0.04] |
| VisionZip 25% | -1.93 [-3.07, -0.81]* |
| VisionZip 50% + OVAL | -2.00 [-3.07, -0.93]* |
| InfiniPot-V (mem-matched) | -0.93 [-1.81, -0.04]* |
| InfiniPot-V (2k) | -5.07 [-6.44, -3.70]* |
| ReKV | -3.85 [-5.26, -2.48]* |

Paired difference vs FullKV-32 (pp), 95% CI; * = CI excludes 0

| arm | overall |
|---|---|
| FullKV-64 | +2.04 [+0.78, +3.33]* |
| OVAL | +0.26 [-1.07, +1.59] |
| VisionZip 50% | +1.26 [+0.00, +2.52] |
| VisionZip 25% | +0.11 [-1.22, +1.44] |
| VisionZip 50% + OVAL | +0.04 [-1.22, +1.30] |
| InfiniPot-V (mem-matched) | +1.11 [-0.15, +2.37] |
| InfiniPot-V (2k) | -3.04 [-4.48, -1.63]* |
| ReKV | -1.81 [-3.33, -0.33]* |

## Accuracy (%), duration, 95% CI (bootstrap over videos)

| arm | short (n=900) | medium (n=900) | long (n=900) |
|---|---|---|---|
| FullKV-64 | 75.11 [72.22, 77.89] | 63.78 [60.67, 66.89] | 52.22 [49.00, 55.44] |
| FullKV-32 | 72.67 [69.67, 75.56] | 61.11 [58.00, 64.11] | 51.22 [47.89, 54.56] |
| OVAL | 72.56 [69.67, 75.33] | 62.22 [59.00, 65.33] | 51.00 [47.67, 54.33] |
| VisionZip 50% | 74.56 [71.67, 77.33] | 62.33 [59.22, 65.33] | 51.89 [48.67, 55.22] |
| VisionZip 25% | 72.44 [69.56, 75.33] | 60.89 [57.44, 64.22] | 52.00 [48.78, 55.22] |
| VisionZip 50% + OVAL | 73.56 [70.67, 76.33] | 61.33 [58.11, 64.44] | 50.22 [47.00, 53.56] |
| InfiniPot-V (mem-matched) | 73.22 [70.33, 76.11] | 63.44 [60.33, 66.56] | 51.67 [48.22, 55.00] |
| InfiniPot-V (2k) | 68.00 [65.00, 71.00] | 58.11 [54.67, 61.56] | 49.78 [46.56, 53.00] |
| ReKV | 69.67 [66.78, 72.56] | 57.89 [54.67, 61.00] | 52.00 [48.67, 55.33] |

Paired difference vs FullKV-64 (pp), 95% CI; * = CI excludes 0

| arm | short | medium | long |
|---|---|---|---|
| FullKV-32 | -2.44 [-4.56, -0.33]* | -2.67 [-5.22, -0.22]* | -1.00 [-3.00, +1.00] |
| OVAL | -2.56 [-4.22, -1.00]* | -1.56 [-3.67, +0.44] | -1.22 [-2.89, +0.44] |
| VisionZip 50% | -0.56 [-1.89, +0.78] | -1.44 [-3.22, +0.33] | -0.33 [-1.44, +0.78] |
| VisionZip 25% | -2.67 [-4.44, -0.89]* | -2.89 [-5.22, -0.67]* | -0.22 [-1.89, +1.44] |
| VisionZip 50% + OVAL | -1.56 [-3.22, +0.11] | -2.44 [-4.56, -0.33]* | -2.00 [-3.67, -0.33]* |
| InfiniPot-V (mem-matched) | -1.89 [-3.33, -0.44]* | -0.33 [-2.00, +1.33] | -0.56 [-2.00, +0.89] |
| InfiniPot-V (2k) | -7.11 [-9.44, -4.89]* | -5.67 [-8.33, -3.00]* | -2.44 [-4.56, -0.33]* |
| ReKV | -5.44 [-7.67, -3.11]* | -5.89 [-8.44, -3.33]* | -0.22 [-2.44, +2.11] |

Paired difference vs FullKV-32 (pp), 95% CI; * = CI excludes 0

| arm | short | medium | long |
|---|---|---|---|
| FullKV-64 | +2.44 [+0.33, +4.56]* | +2.67 [+0.22, +5.22]* | +1.00 [-1.00, +3.00] |
| OVAL | -0.11 [-2.00, +1.78] | +1.11 [-1.44, +3.78] | -0.22 [-2.44, +2.11] |
| VisionZip 50% | +1.89 [-0.11, +3.89] | +1.22 [-1.00, +3.44] | +0.67 [-1.44, +2.89] |
| VisionZip 25% | -0.22 [-2.33, +1.89] | -0.22 [-2.78, +2.33] | +0.78 [-1.44, +3.00] |
| VisionZip 50% + OVAL | +0.89 [-1.00, +2.78] | +0.22 [-2.11, +2.56] | -1.00 [-3.22, +1.22] |
| InfiniPot-V (mem-matched) | +0.56 [-1.56, +2.56] | +2.33 [+0.11, +4.67]* | +0.44 [-1.78, +2.67] |
| InfiniPot-V (2k) | -4.67 [-7.11, -2.33]* | -3.00 [-5.67, -0.33]* | -1.44 [-3.78, +0.89] |
| ReKV | -3.00 [-5.56, -0.44]* | -3.22 [-5.89, -0.44]* | +0.78 [-1.78, +3.33] |

†: SnapKV is reported, never ranked.
VisionZip, InfiniPot-V and ReKV are our ports (see REPORT_C.md, C3).
