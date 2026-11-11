# C5: evidence position on LVBench

Bucket = midpoint of `time_reference` / (60 x `duration_minutes`), thirds. 95% CIs: bootstrap over questions (10,000 draws). Paired: same questions.

## LLaVA-Video-7B

Runs: FullKV-64 1549/1549, FreeKV 1549/1549, OVAL 1549/1549. Scored on 1547 common questions. Excluded (span annotated as None): 2 questions (3925, 5044). Span anomalies kept: {'end before start': 13, 'end after duration': 2}.

| arm | first (n=708) | middle (n=520) | last (n=319) | all |
|---|---|---|---|---|
| FullKV-64 | 44.49 [40.82, 48.16] | 40.77 [36.54, 45.00] | 36.99 [31.97, 42.32] | 41.69 [39.24, 44.15] |
| FreeKV | 45.90 [42.23, 49.58] | 39.42 [35.19, 43.65] | 38.56 [33.23, 43.89] | 42.21 [39.75, 44.60] |
| OVAL | 43.93 [40.25, 47.60] | 38.65 [34.42, 42.88] | 37.93 [32.60, 43.26] | 40.92 [38.46, 43.31] |

Paired difference vs FullKV-64 (pp); * = CI excludes 0

| arm | first | middle | last | all |
|---|---|---|---|---|
| FreeKV | +1.41 [-0.85, +3.67] | -1.35 [-3.85, +1.15] | +1.57 [-1.88, +5.02] | +0.52 [-1.03, +2.07] |
| OVAL | -0.56 [-2.82, +1.69] | -2.12 [-4.42, +0.19] | +0.94 [-2.82, +4.70] | -0.78 [-2.26, +0.78] |

Sampled frames inside the span: 0 frames for 838 of 1547 questions (54.17%); median 0, max 64. Distribution: 0: 838, 1: 333, 2: 83, 3: 45, 4: 27, 5: 16, 6: 14, 7: 10 ...

Accuracy on the 709 questions whose span contains >= 1 sampled frame (vs the 838 with none):

| arm | span has a frame | span has no frame |
|---|---|---|
| FullKV-64 | 43.02 [39.35, 46.69] | 40.57 [37.23, 43.91] |
| FreeKV | 43.44 [39.92, 47.11] | 41.17 [37.83, 44.51] |
| OVAL | 42.60 [38.93, 46.26] | 39.50 [36.16, 42.84] |

## Qwen2.5-VL-7B

Runs: FullKV-64 1549/1549, FreeKV 1549/1549, OVAL 1549/1549. Scored on 1547 common questions. Excluded (span annotated as None): 2 questions (3925, 5044). Span anomalies kept: {'end before start': 13, 'end after duration': 2}.

| arm | first (n=708) | middle (n=520) | last (n=319) | all |
|---|---|---|---|---|
| FullKV-64 | 42.94 [39.41, 46.61] | 41.92 [37.69, 46.15] | 36.68 [31.66, 42.01] | 41.31 [38.85, 43.76] |
| FreeKV | 42.09 [38.42, 45.76] | 37.50 [33.46, 41.54] | 32.29 [27.27, 37.62] | 38.53 [36.13, 40.92] |
| OVAL | 42.51 [38.84, 46.19] | 39.62 [35.58, 43.85] | 33.86 [28.84, 39.18] | 39.75 [37.36, 42.15] |

Paired difference vs FullKV-64 (pp); * = CI excludes 0

| arm | first | middle | last | all |
|---|---|---|---|---|
| FreeKV | -0.85 [-3.25, +1.69] | -4.42 [-7.31, -1.54]* | -4.39 [-8.15, -0.94]* | -2.78 [-4.46, -1.10]* |
| OVAL | -0.42 [-2.68, +1.84] | -2.31 [-4.81, +0.19] | -2.82 [-6.27, +0.31] | -1.55 [-3.04, -0.06]* |

Sampled frames inside the span: 0 frames for 838 of 1547 questions (54.17%); median 0, max 64. Distribution: 0: 838, 1: 333, 2: 83, 3: 45, 4: 27, 5: 16, 6: 14, 7: 10 ...

Accuracy on the 709 questions whose span contains >= 1 sampled frame (vs the 838 with none):

| arm | span has a frame | span has no frame |
|---|---|---|
| FullKV-64 | 45.13 [41.47, 48.80] | 38.07 [34.73, 41.41] |
| FreeKV | 42.31 [38.65, 45.98] | 35.32 [32.10, 38.54] |
| OVAL | 41.89 [38.22, 45.56] | 37.95 [34.73, 41.29] |

