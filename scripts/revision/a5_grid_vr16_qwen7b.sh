#!/usr/bin/env bash
# A5 in the OVAL setting: held-out output error on the full 21-point eta grid at
# rank-8 keys and rank-16 values (the Tab. 1 configuration), for F2 and F3.
# Same captures/protocol as Sec. S4.4 (seed 7). Runs on GPU 1 after the A calibrations.
# Writes results/revision/A/grid_vr16/<model>_<task>.json (+ tables/, used only to check
# that the rule reproduces the deployed Tab. 1 table hashes).
set -u
source "$HOME/oval-kv/scripts/revision/env.sh"; cd "$KV_ROOT"
GPU="${GPU:-1}"; OUT=results/revision/A/grid_vr16; CAP=logs/revision/capture; LOG=logs/revision/grid_vr16.log

for M in ds-r1-qwen-7b; do
  for T in MATH50 AIME24 GPQA50c; do
    echo "[$(date -Iseconds)] capture $M $T" >> $LOG
    CUDA_VISIBLE_DEVICES=$GPU $KV_PY analysis/eta_rules_calibrate.py capture --model $M --task $T --capture_dir $CAP >> $LOG 2>&1 || { echo "FAILED capture $M $T" >> $LOG; exit 1; }
    echo "[$(date -Iseconds)] stats ngrid 21 $M $T" >> $LOG
    CUDA_VISIBLE_DEVICES=$GPU $KV_PY analysis/eta_rules_calibrate.py stats --model $M --task $T --vrank 16 --ngrid 21 --out $OUT --capture_dir $CAP >> $LOG 2>&1 || { echo "FAILED stats $M $T" >> $LOG; exit 1; }
  done
  $KV_PY analysis/eta_rules_calibrate.py rules --models $M --tasks MATH50 AIME24 GPQA50c --out $OUT >> $LOG 2>&1
  for T in MATH50 AIME24 GPQA50c; do
    echo "[$(date -Iseconds)] regret $M $T" >> $LOG
    CUDA_VISIBLE_DEVICES=$GPU $KV_PY analysis/eta_rules_calibrate.py regret --model $M --task $T --vrank 16 --out $OUT --capture_dir $CAP >> $LOG 2>&1 || echo "regret failed $M $T" >> $LOG
  done
done
echo "[$(date -Iseconds)] DONE" >> $LOG
