#!/usr/bin/env bash
# Block A1: calibrate eta_h at r_v=16 on problems OUTSIDE the evaluation files
# (results/revision/A/calib_data/*, see MANIFEST.json). Protocol of Sec. S4.4 unchanged:
# seed 7, T=0.6, top-p 0.95, 6144 tokens, 3 windows of 96 queries, traces 1-4 fit, 5-6 held out.
# GPQA is pending (GPQA Diamond is gated). Starts after the Llama in-set calibration finishes.
set -u
source "$HOME/oval-kv/scripts/revision/env.sh"; cd "$KV_ROOT"
GPU="${GPU:-1}"; OUT=results/revision/A/calib_vr16; CAP=logs/revision/capture_disjoint; LOG=logs/revision/calib_disjoint.log
mkdir -p $CAP
until grep -q DONE logs/revision/calib_llama_vr16.log 2>/dev/null; do sleep 120; done
declare -A FILE=( [MATH50]=results/revision/A/calib_data/math_disjoint6.jsonl [AIME24]=results/revision/A/calib_data/aime2023_6.jsonl )
for M in ds-r1-qwen-7b ds-r1-llama-8b; do
  for T in MATH50 AIME24; do
    for STAGE in capture stats; do
      X=""; [ $STAGE = stats ] && X="--vrank 16 --ngrid 2 --out $OUT"
      echo "[$(date -Iseconds)] $STAGE $M $T" >> $LOG
      CUDA_VISIBLE_DEVICES=$GPU $KV_PY analysis/eta_rules_calibrate.py $STAGE --model $M --task $T \
        --dataset_file ${FILE[$T]} --capture_dir $CAP --tag _disjoint $X >> $LOG 2>&1 || { echo "FAILED $STAGE $M $T" >> $LOG; exit 1; }
    done
  done
  $KV_PY analysis/eta_rules_calibrate.py rules --models $M --tasks MATH50 AIME24 --out $OUT --tag _disjoint >> $LOG 2>&1
done
echo "[$(date -Iseconds)] DONE" >> $LOG
