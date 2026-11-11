#!/usr/bin/env bash
# eta_h calibration for Qwen2.5-VL-7B, same config as the LLaVA-Video table (results/vision/eta/llava-video-7b_VCAL.json):
# 6 clips (p0-p3 calibrate, p4-p5 test), 3 windows x 96 prefill queries, sink/recent 128, budget 1792,
# r = 8, r_v = 16, 21-point eta grid. Output: results/revision/C/eta/{qwen2.5-vl-7b_VCAL.json, tables/}.
set -eu
cd "$(dirname "$0")/../../.."; source scripts/revision/C/env.sh
M=qwen2.5-vl-7b; O=results/revision/C/eta
C="--model $M --task VCAL --dataset_file $VMME_ROOT/calib_qwen/calib.jsonl --out $O --capture_dir logs/eta_rules/capture \
   --nprob 6 --ncalib 4 --nwin 3 --win_stride 1024 --nq 96 --sink 128 --recent 128 --budget 1792 --page 32 --rank 8 --vrank 16 --ngrid 21"
python analysis/eta_rules_calibrate.py stats $C
python analysis/eta_rules_calibrate.py rules --models $M --tasks VCAL --out $O
python analysis/eta_rules_calibrate.py regret $C
