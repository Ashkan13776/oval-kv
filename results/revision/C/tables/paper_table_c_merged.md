| Method | Budget | videomme Qwen2.5-VL short | videomme Qwen2.5-VL medium | videomme Qwen2.5-VL long | videomme Qwen2.5-VL overall | videomme LLaVA-Video short | videomme LLaVA-Video medium | videomme LLaVA-Video long | videomme LLaVA-Video overall | lvbench Qwen2.5-VL overall | lvbench LLaVA-Video overall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FullKV ($N$ frames) | 100\% | 75.11 | 63.78 | 52.22 | 63.70 | 75.56 | 61.67 | 51.44 | 62.89 | 41.25 | 41.64 |
| FullKV ($0.69N$ frames) | 69\% | -- | -- | -- | -- | 74.33 | 60.67 | 51.11 | 62.04 | -- | -- |
| FullKV ($N/2$ frames) | 50\% | 72.67* | 61.11* | 51.22 | 61.67* | 74.00 | 57.56* | 51.00 | 60.85* | 38.41* | 39.38* |
| VisionZip$^\ddagger$ | 50\% | 74.56 | 62.33 | 51.89 | 62.93 | 76.00 | 62.33 | 52.00 | 63.44 | 39.90* | 41.51 |
| InfiniPot-V$^\ddagger$ | 50\% | 73.22* | 63.44 | 51.67 | 62.78* | 74.22 | 61.44 | 50.56 | 62.07 | 40.41 | 41.64 |
| VisionZip$^\ddagger$ | 25\% | 72.44* | 60.89* | 52.00 | 61.78* | 72.89* | 60.56 | 49.89 | 61.11* | 38.48* | 40.35 |
| Sliding window | $\approx$15\% | -- | -- | -- | -- | 59.11* | 53.00* | 45.78* | 52.63* | -- | -- |
| InfiniPot-V$^\ddagger$ | $\approx$15\% | 68.00* | 58.11* | 49.78* | 58.63* | 70.00* | 57.89* | 47.78* | 58.56* | 36.60* | 38.99* |
| ReKV$^\ddagger$ | $\approx$15\% | 69.67* | 57.89* | \topthree{52.00} | 59.85* | 70.89* | 57.78* | \topone{50.44} | 59.70* | 37.38* | 39.70 |
| SnapKV$^\dagger$ | $\approx$15\% | 64.33* | 59.00* | 49.44* | 57.59* | 64.33* | 56.00* | 46.67* | 55.67* | 36.09* | 36.22* |
| Quest | $\approx$15\% | \topone{74.22} | \topone{62.22} | \topone{52.67} | \topone{63.04} | \topone{75.00} | \topone{60.56} | \toptwo{49.89*} | \topone{61.81*} | \toptwo{38.99*} | \toptwo{41.32} |
| FreeKV | $\approx$15\% | \toptwo{73.78} | \topthree{61.22*} | \toptwo{52.56} | \toptwo{62.52*} | \toptwo{74.44} | \topone{60.56} | \topthree{49.33*} | \toptwo{61.44*} | \topthree{38.48*} | \topone{42.16} |
| \method{} | $\approx$15\% | \topthree{72.56*} | \topone{62.22} | 51.00 | \topthree{61.93*} | \toptwo{74.44} | \topone{60.56} | 49.11* | \topthree{61.37*} | \topone{39.70*} | \topthree{40.87} |
