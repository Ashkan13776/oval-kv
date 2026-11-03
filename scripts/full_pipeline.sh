#!/usr/bin/env bash
# The complete OVAL configuration on long reasoning, end to end, for one or
# more models. Read docs/5-full-pipeline.md first.
#
#   retrieval  OVAL page scoring with the rank-8 key records (avgSM, spec_ret)
#   keys       selected pages rebuilt from the rank-8 key records  (--oval_recon)
#   values     selected pages rebuilt from rank-VRANK PCA records  (--oval_vrank)
#   eta        one per (layer, KV head), rule 1, calibrated AT value rank VRANK
#
# Steps (each one skips work whose output already exists, so re-running after
# a crash resumes):
#   1. capture   seed-7 calibration traces per (model, task)      [GPU, CALIB_GPU]
#   2. stats     rule 1's endpoint errors at eta = 0 and 1         [GPU, CALIB_GPU]
#   3. rules     per-(layer, KV head) eta tables -> results/eta_rules/vr<VRANK>/tables
#   4. sweep     supervised harness cells, k seeds x tasks x models [all GPUS]
#   5. score     results/eta_rules/vr<VRANK>/table_<models>.md (one file per model set,
#                so agents running different models never write the same file)
#
#   MODELS="ds-r1-qwen-7b ds-r1-llama-8b" GPUS="0 1 2 3" ./scripts/full_pipeline.sh
#   MODELS=ds-r1-qwen-14b GPUS="0 1" SLOTS=1 MINFREE=40000 ./scripts/full_pipeline.sh
#   STEPS="score" ./scripts/full_pipeline.sh          # any subset of the steps
set -u
KV_ROOT="${KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KV_PY="${KV_PY:-python}"
MODELS="${MODELS:-ds-r1-qwen-7b ds-r1-llama-8b}"
DATASETS="${DATASETS:-MATH50 AIME24 GPQA50c}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49}"
GPUS="${GPUS:-0 1 2 3}"
CALIB_GPU="${CALIB_GPU:-${GPUS%% *}}"
VRANK="${VRANK:-16}"
SLOTS="${SLOTS:-2}"
MINFREE="${MINFREE:-20000}"
STEPS="${STEPS:-capture stats rules sweep score}"
OUT="$KV_ROOT/results/eta_rules/vr$VRANK"
LOG="$KV_ROOT/logs/eta_rules"; mkdir -p "$LOG" "$OUT"
HB="$LOG/full_pipeline_vr$VRANK.log"
say () { echo "[$(date -Iseconds)] $*" | tee -a "$HB"; }
want () { [[ " $STEPS " == *" $1 "* ]]; }
cd "$KV_ROOT"

for M in $MODELS; do
  for T in $DATASETS; do
    if want capture; then
      say "capture $M/$T (gpu $CALIB_GPU)"
      CUDA_VISIBLE_DEVICES=$CALIB_GPU $KV_PY analysis/eta_rules_calibrate.py capture --model "$M" --task "$T" \
          >> "$LOG/calib_$M.log" 2>&1 || { say "ABORT: capture failed for $M/$T (see $LOG/calib_$M.log)"; exit 1; }
    fi
    if want stats; then
      say "stats $M/$T at value rank $VRANK (gpu $CALIB_GPU)"
      CUDA_VISIBLE_DEVICES=$CALIB_GPU $KV_PY analysis/eta_rules_calibrate.py stats --model "$M" --task "$T" \
          --vrank "$VRANK" --ngrid 2 --out "$OUT" \
          >> "$LOG/calib_$M.log" 2>&1 || { say "ABORT: stats failed for $M/$T (see $LOG/calib_$M.log)"; exit 1; }
    fi
  done
done

if want rules; then
  $KV_PY analysis/eta_rules_calibrate.py rules --models $MODELS --tasks $DATASETS --out "$OUT" >> "$HB" 2>&1 \
    || { say "ABORT: rules stage failed (see $HB)"; exit 1; }
  say "tables: $(ls "$OUT/tables" | tr '\n' ' ')"
fi

if want sweep; then
  $KV_PY analysis/eta_rules_queue.py --models $MODELS --datasets $DATASETS --seeds $SEEDS \
      --tables_dir "$OUT/tables" --vrank "$VRANK" | tee -a "$HB"
  say "sweep: supervisor up (VRANK=$VRANK, GPUs [$GPUS], $SLOTS slot(s)/GPU, minfree ${MINFREE}MiB)"
  MODELS="$MODELS" DATASETS="$DATASETS" SEEDS="$SEEDS" GPUS="$GPUS" SLOTS="$SLOTS" MINFREE="$MINFREE" \
    VRANK="$VRANK" TABLES="$OUT/tables" KV_ROOT="$KV_ROOT" KV_PY="$KV_PY" \
    "$KV_ROOT/scripts/supervise_eta_tables.sh" \
    || { say "ABORT: supervisor exited non-zero (see $LOG/supervisor_vr$VRANK.log)"; exit 1; }
fi

if want score; then
  TAB="$OUT/table_$(echo $MODELS | tr ' ' '+').md"
  $KV_PY analysis/eta_rules_table.py --models $MODELS --datasets $DATASETS --vrank "$VRANK" \
      --eta_dir "$OUT" --out "$TAB" >> "$HB" 2>&1
  say "scored -> $TAB"
fi
say "DONE ($STEPS)"
