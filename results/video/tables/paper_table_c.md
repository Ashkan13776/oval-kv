| Method | Budget | Qwen2.5-VL, Video-MME | LLaVA-Video, Video-MME | Qwen2.5-VL, LVBench | LLaVA-Video, LVBench |
|---|---|---|---|---|---|
| FullKV ($N$ frames) | 100\% | 63.70 | 62.89 | 41.25 | 41.64 |
| FullKV ($N/2$ frames) | 50\% | 61.67* | 60.85* | 38.41* | 39.38* |
| VisionZip$^\ddagger$ | 50\% | \topthree{62.93} | \topone{63.44} | \topthree{39.90*} | 41.51 |
| InfiniPot-V$^\ddagger$ | 50\% | 62.78* | \topthree{62.07} | \toptwo{40.41} | \topthree{41.64} |
| VisionZip$^\ddagger$ | 25\% | 61.78* | 61.11* | 38.48* | 40.35 |
| InfiniPot-V$^\ddagger$ | $\approx$15\% | 58.63* | 58.56* | 36.60* | 38.99* |
| ReKV$^\ddagger$ | $\approx$15\% | 59.85* | 59.70* | 37.38* | 39.70 |
| SnapKV$^\dagger$ | $\approx$15\% | 57.59* | 55.67* | 36.09* | 36.22* |
| Quest | $\approx$15\% | \toptwo{63.04} | 61.81* | 38.99* | 41.32 |
| FreeKV | $\approx$15\% | 62.52* | 61.44* | 38.48* | \toptwo{42.16} |
| 4-bit keys | $\approx$15\% | \topone{63.11} | \toptwo{62.81} | \topone{40.99} | \topone{42.35} |
| \method{} | $\approx$15\% | 61.93* | 61.37* | 39.70* | 40.87 |
