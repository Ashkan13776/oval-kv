# Block C stack (one row per run)

All Block C runs on one stack. Machine mint06, 4x NVIDIA RTX A6000 (49,140 MiB), driver 580.178.04,
CUDA 12.8 (torch wheel), cuDNN 9.10.2 (91002), Python 3.11.18, system gcc 11.4 for flashinfer JIT.
Harness env `ovalkv`: torch 2.8.0+cu128, torchvision 0.23.0+cu128, transformers 4.45.2, flash-attn 2.8.3,
flashinfer 0.2.4 (JIT), decord 0.6.0, timm 0.9.16, LLaVA-NeXT `bce12e4` (`--no-deps`), lmms-eval `d45b4ad`
(prompt and scorer only), FreeKV `2c8a7d2` + `patches/freekv.patch`, `oval/page_basis.py` of `vision` installed as
`kvc/patch/oval.py`. This is the stack of the Tab. 5 runs (STATUS.md on `vision`: torch 2.8.0+cu128,
transformers 4.45.2, flash-attn 2.8.3, flashinfer 0.2.4).
Qwen2.5-VL vision side (`~/qwenvl`, a venv over `ovalkv` with `--system-site-packages`): transformers 4.51.3,
tokenizers 0.21.1, qwen-vl-utils 0.0.11; same torch.

| block | model | arm | run dir | GPU | torch | transformers | flash-attn | flashinfer | repo commit | date |
|---|---|---|---|---|---|---|---|---|---|---|
| C (repro) | LLaVA-Video-7B-Qwen2 | FullKV-64 | runs/repro_row1_full_n64 | RTX A6000 | 2.8.0+cu128 | 4.45.2 | 2.8.3 | 0.2.4 | bb7ce66 + revision-c scripts | 2026-11-08 |
| C (repro) | LLaVA-Video-7B-Qwen2 | OVAL | runs/repro_row3_ovalfull_n64 | RTX A6000 | 2.8.0+cu128 | 4.45.2 | 2.8.3 | 0.2.4 | bb7ce66 + revision-c scripts | 2026-11-08 |
