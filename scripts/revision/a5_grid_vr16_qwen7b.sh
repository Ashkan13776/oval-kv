#!/usr/bin/env bash
# A5 r_v=16 grid for Qwen-7B on GPU 1, rerun after the 10:14 OOM (decodes shared the GPU).
# logs/revision/reserve/gpu1 keeps the queue off GPU 1; wait until it has drained, run the
# tasks whose stats are missing, then rules and regret for all three, then release GPU 1.
set -u
source "$HOME/oval-kv/scripts/revision/env.sh"; cd "$KV_ROOT"
GPU=1; M=ds-r1-qwen-7b; OUT=results/revision/A/grid_vr16; CAP=logs/revision/capture; LOG=logs/revision/grid_vr16.log
touch logs/revision/reserve/gpu$GPU
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)" -gt 40000 ]; do sleep 120; done
echo "[$(date -Iseconds)] qwen-7b grid restart on GPU $GPU (reserved)" >> $LOG
for T in MATH50 AIME24 GPQA50c; do
  [ -f $OUT/${M}_$T.json ] && continue
  echo "[$(date -Iseconds)] capture $M $T" >> $LOG
  CUDA_VISIBLE_DEVICES=$GPU $KV_PY analysis/eta_rules_calibrate.py capture --model $M --task $T --capture_dir $CAP >> $LOG 2>&1 || { echo "FAILED capture $M $T" >> $LOG; rm -f logs/revision/reserve/gpu$GPU; exit 1; }
  echo "[$(date -Iseconds)] stats ngrid 21 $M $T" >> $LOG
  CUDA_VISIBLE_DEVICES=$GPU $KV_PY analysis/eta_rules_calibrate.py stats --model $M --task $T --vrank 16 --ngrid 21 --out $OUT --capture_dir $CAP >> $LOG 2>&1 || { echo "FAILED stats $M $T" >> $LOG; rm -f logs/revision/reserve/gpu$GPU; exit 1; }
done
$KV_PY analysis/eta_rules_calibrate.py rules --models $M --tasks MATH50 AIME24 GPQA50c --out $OUT >> $LOG 2>&1
for T in MATH50 AIME24 GPQA50c; do
  echo "[$(date -Iseconds)] regret $M $T" >> $LOG
  CUDA_VISIBLE_DEVICES=$GPU $KV_PY analysis/eta_rules_calibrate.py regret --model $M --task $T --vrank 16 --out $OUT --capture_dir $CAP >> $LOG 2>&1 || echo "regret failed $M $T" >> $LOG
done
rm -f logs/revision/reserve/gpu$GPU
echo "[$(date -Iseconds)] DONE" >> $LOG
