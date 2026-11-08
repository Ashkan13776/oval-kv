#!/usr/bin/env bash
# Block A: rank-16 calibration for R1-Distill Llama-8B (the repo has Llama tables at r_v=8 only).
# Same protocol and steps as scripts/full_pipeline.sh (capture -> stats --vrank 16 --ngrid 2 -> rules).
set -u
source "$HOME/oval-kv/scripts/revision/env.sh"
cd "$KV_ROOT"
GPU="${GPU:-1}"
OUT=results/revision/A/calib_vr16; CAP=logs/revision/capture; LOG=logs/revision/calib_llama_vr16.log
for T in MATH50 AIME24 GPQA50c; do
  echo "[$(date -Iseconds)] capture $T" >> $LOG
  CUDA_VISIBLE_DEVICES=$GPU $KV_PY analysis/eta_rules_calibrate.py capture --model ds-r1-llama-8b --task $T --capture_dir $CAP >> $LOG 2>&1 || { echo "capture failed $T" >> $LOG; exit 1; }
  echo "[$(date -Iseconds)] stats $T" >> $LOG
  CUDA_VISIBLE_DEVICES=$GPU $KV_PY analysis/eta_rules_calibrate.py stats --model ds-r1-llama-8b --task $T --vrank 16 --ngrid 2 --out $OUT --capture_dir $CAP >> $LOG 2>&1 || { echo "stats failed $T" >> $LOG; exit 1; }
done
$KV_PY analysis/eta_rules_calibrate.py rules --models ds-r1-llama-8b --tasks MATH50 AIME24 GPQA50c --out $OUT >> $LOG 2>&1
echo "[$(date -Iseconds)] DONE" >> $LOG
