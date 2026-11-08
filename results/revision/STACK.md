# Revision runs: software stack

| date | block | model | arm | GPU | driver | torch | transformers | repo commit |
|---|---|---|---|---|---|---|---|---|
| 2026-11-07 | A3 | ds-r1-qwen-7b | shared/layer/argmin01 tables, OVAL Tab. 1 switches | RTX A6000 (GPUs 0,2,3) | scripts/revision/queue.sh | 2.5.1+cu121 | 4.45.2 | aac57ab |
| 2026-11-07 | A (calib) | ds-r1-llama-8b | r_v=16 calibration | RTX A6000 (GPU 1) | scripts/revision/a_calib_llama_vr16.sh | 2.5.1+cu121 | 4.45.2 | aac57ab |
