#!/usr/bin/env bash
# Decoupled-rank ablation: SCORING rank fixed, RECONSTRUCTION rank swept.
#
# --oval_rank sets the scoring rank; --oval_recon_rank sets the codec rank.
# Records are stored at max(score, recon) and the basis is eigenvalue-ordered,
# so each consumer takes an exact top-r truncation of the stored record
# (verified: truncating rank-32 to top-8 is bit-identical to building at 8).
#
# ETA IS PER TASK (ETAMAP). Note: at eta>0 the basis is the eigenbasis of the
# WHITENED Gram, so a top-r truncation is still exactly the rank-r record at
# that eta, but it is optimal in the output-coupled metric rather than in key
# MSE -- the rank axis is read against that metric, not against key error.
#
#   SCORE_RANK  default 8
#   RECON_RANKS default "1 4 8 16 32"   (32 clamps to page_size-1 = 31 = exact)
#   SEEDS       default 42..49          (k=8 -> avg@8 and pass@8)
set -u
KV_ROOT="${KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KV_PY="${KV_PY:-python}"
FREEKV_DIR="${FREEKV_DIR:-$KV_ROOT/third_party/FreeKV}"
cd "$FREEKV_DIR/accuracy" || { echo "no FreeKV at $FREEKV_DIR"; exit 1; }
export PYTHONUNBUFFERED=1 OVAL_QUANT=0 KVC_MAX_TOKENS=32768
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

M="${MODEL:?set MODEL}"
DATASETS="${DATASETS:-MATH50 AIME24 GPQA50c}"
SCORE_RANK="${SCORE_RANK:-8}"
RECON_RANKS="${RECON_RANKS:-1 4 8 16 32}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49}"
GPUS="${GPUS:-0 1 2 3}"
# per-task eta: "<DATASET>:<eta> ..." ; falls back to ETA_DEFAULT
ETAMAP="${ETAMAP:-MATH50:0.25 AIME24:1.0 GPQA50c:0.25}"
ETA_DEFAULT="${ETA_DEFAULT:-0.0}"
eta_for () {
  local d="$1" kv
  for kv in $ETAMAP; do
    [ "${kv%%:*}" = "$d" ] && { echo "${kv##*:}"; return; }
  done
  echo "$ETA_DEFAULT"
}
NSEED=$(echo $SEEDS | wc -w)
LOG="$KV_ROOT/logs"; OUT="$KV_ROOT/results/codec"
RES="$OUT/${M}-spec_ret"; mkdir -p "$LOG" "$OUT"
Q="$LOG/codec_recon_rank_jobs.txt"; : > "$Q"

for D in $DATASETS; do
  E=$(eta_for "$D")
  for R in $RECON_RANKS; do
    have=$(ls "$RES"/${D}-*oval_r${SCORE_RANK}_eta${E}_q0_recon1_cr${R}-*.jsonl 2>/dev/null | wc -l)
    if [ "$have" -ge "$NSEED" ]; then echo "   skip $D eta=$E cr=$R (already $have/$NSEED)"
    else for S in $SEEDS; do echo "$D|$S|$R|$E" >> "$Q"; done; fi
  done
done
echo "=== $(wc -l < "$Q") cells queued for $M  score_rank=$SCORE_RANK  etamap=[$ETAMAP]  $(date -Iseconds)"

run_job () {
  local d s r e g; IFS='|' read -r d s r e <<< "$1"; g=$2
  echo "=== START $M/$d/eta$e/s$s/cr$r gpu=$g $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES=$g $KV_PY -u -m eval.reasoning.pred \
      --model "$M" --dataset "$d" \
      --GQA_policy avgSM --spec_ret_steps 2 --spec_ret_corr 0.9 \
      --temperature 0.6 --top_p 0.95 --max_gen 16384 \
      --budget 2048 --sink 512 --recent 512 --page_size 32 \
      --seed "$s" --out_root_dir "$OUT" \
      --method spec_ret --page_rep oval --eta "$e" \
      --oval_rank "$SCORE_RANK" --oval_recon --oval_recon_rank "$r" \
      > "$LOG/recrank_${M}_${d}_eta${e}_s${s}_cr${r}.log" 2>&1
  echo "=== DONE  $M/$d/eta$e/s$s/cr$r exit=$? $(date -Iseconds)"
}

SLOTS="${SLOTS:-1}"; MINFREE="${MINFREE:-40000}"; SETTLE="${SETTLE:-60}"
SLOTLIST=""; for g in $GPUS; do for i in $(seq 1 "$SLOTS"); do SLOTLIST="$SLOTLIST $g:$i"; done; done
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
    sed -i '1d' "$Q"; run_job "$job" "$g" & BUSY[$slot]=$!
    sleep "$SETTLE"
  done
  sleep 45
done
wait
echo "=== RECON-RANK SWEEP COMPLETE for $M  $(date -Iseconds)"
