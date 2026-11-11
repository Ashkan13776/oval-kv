# C5: per-frame selection (OVAL, Video-MME Long, real runs)

## LLaVA-Video-7B: 900 of 900 Long questions

- mean selected fraction, all frames: 12.31%
- first 4 + last 4 frames: 27.14%; frames 4-59: 10.19%; ratio 2.66 [2.62, 2.71] (bootstrap over videos)
- frame 0: 25.47%, frame 63: 44.73%; max at frame 63 (44.73%), min at frame 31 (5.16%)

## Qwen2.5-VL-7B: 900 of 900 Long questions

- mean selected fraction, all frames: 14.10%
- first 4 + last 4 frames: 24.79%; frames 4-59: 12.58%; ratio 1.97 [1.92, 2.02] (bootstrap over videos)
- frame 0: 36.54%, frame 63: 20.64%; max at frame 0 (36.54%), min at frame 40 (10.12%)

