#!/usr/bin/env bash
# Harness cells for the eta-rule study: one cell per distinct per-(layer, KV
# head) eta table per (model, dataset), k seeds each, protocol identical to the
# value-codec cells (keys rank 8 + --oval_recon, values PCA rank 8), with
# --eta_table <json> in place of --eta.
#
# Sibling of codec_reasoning_sweep.sh: same scheduler (SLOTS per GPU, MINFREE
# gate, SETTLE), same per-seed completeness rule, its OWN queue file and logs,
# so it never touches the codec sweep's queue. The queue is built by
# analysis/eta_rules_queue.py, which matches the output tag exactly
# (..._etab<hash>_q0_recon1_vr8_vpca-pQ2) so eta-table cells and fixed-eta
# cells are never counted as each other.
#
#   KV_ROOT, KV_PY, FREEKV_DIR   as in codec_reasoning_sweep.sh
#   MODELS     default "ds-r1-qwen-7b ds-r1-llama-8b"
#   DATASETS   default "MATH50 AIME24 GPQA50c"   (queue order; MATH50 first)
#   SEEDS      default "42 43 44 45 46 47 48 49"
#   RULES      default "r1 r2 r3 r4 r5 r6"
#   TABLES     default $KV_ROOT/results/eta_rules/tables
#   GPUS, SLOTS, MINFREE, SETTLE   scheduler
set -u
KV_ROOT="${KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KV_PY="${KV_PY:-python}"
FREEKV_DIR="${FREEKV_DIR:-$KV_ROOT/third_party/FreeKV}"
cd "$FREEKV_DIR/accuracy" 2>/dev/null || { echo "no FreeKV at $FREEKV_DIR -- run ./setup.sh"; exit 1; }

export PYTHONUNBUFFERED=1
export OVAL_QUANT=0                    # bfloat16 records (and the _q0 tag)
export KVC_MAX_TOKENS=32768
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODELS="${MODELS:-ds-r1-qwen-7b ds-r1-llama-8b}"
DATASETS="${DATASETS:-MATH50 AIME24 GPQA50c}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49}"
RULES="${RULES:-r1 r2 r3 r4 r5 r6}"
TABLES="${TABLES:-$KV_ROOT/results/eta_rules/tables}"
GPUS="${GPUS:-0 1 2 3}"
LOG="$KV_ROOT/logs/eta_rules"; OUT="$KV_ROOT/results/codec"
mkdir -p "$LOG" "$OUT"
Q="$LOG/jobs.txt"

$KV_PY "$KV_ROOT/analysis/eta_rules_queue.py" --models $MODELS --datasets $DATASETS --seeds $SEEDS \
    --rules $RULES --tables_dir "$TABLES" --root "$OUT" --write "$Q" || exit 1
echo "=== $(wc -l < "$Q") seed-cells queued  $(date -Iseconds)"

run_job () {
  local m d s h tab g
  IFS='|' read -r m d s h tab <<< "$1"; g=$2
  echo "=== START $m/$d/s$s/etab$h gpu=$g $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES=$g $KV_PY -u -m eval.reasoning.pred \
      --model "$m" --dataset "$d" \
      --GQA_policy avgSM --spec_ret_steps 2 --spec_ret_corr 0.9 \
      --temperature 0.6 --top_p 0.95 --max_gen 16384 \
      --budget 2048 --sink 512 --recent 512 --page_size 32 \
      --seed "$s" --out_root_dir "$OUT" \
      --method spec_ret --page_rep oval --eta_table "$tab" --oval_rank 8 --oval_recon \
      --oval_vrank 8 --vbasis pca \
      > "$LOG/cell_${m}_${d}_s${s}_etab${h}.log" 2>&1
  echo "=== DONE  $m/$d/s$s/etab$h exit=$? $(date -Iseconds)"
}

SLOTS="${SLOTS:-2}"
MINFREE="${MINFREE:-20000}"
SETTLE="${SETTLE:-90}"
SLOTLIST=""
for g in $GPUS; do
  for i in $(seq 1 "$SLOTS"); do SLOTLIST="$SLOTLIST $g:$i"; done
done
echo "=== scheduler: $SLOTS slot(s) x GPUs [$GPUS], minfree ${MINFREE}MiB"

declare -A BUSY
while [ -s "$Q" ]; do
  for slot in $SLOTLIST; do
    g="${slot%%:*}"
    [ -n "${BUSY[$slot]:-}" ] && kill -0 "${BUSY[$slot]}" 2>/dev/null && continue
    BUSY[$slot]=""
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g")
    [ "$free" -lt "$MINFREE" ] && continue
    job=$(head -1 "$Q"); [ -z "$job" ] && break
    sed -i '1d' "$Q"
    run_job "$job" "$g" &
    BUSY[$slot]=$!
    sleep "$SETTLE"
  done
  sleep 45
done
wait
echo "=== ETA-TABLE SWEEP COMPLETE  $(date -Iseconds)"
