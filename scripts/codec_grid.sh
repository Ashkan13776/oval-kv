#!/usr/bin/env bash
# OVAL records used as a KEY CODEC -- long-reasoning rank grid.
#
# Normally the records only SELECT pages and the surviving pages are attended
# with original keys. With --oval_recon the selected pages are instead rebuilt
# from the records (mu + coef @ basis^T), so the records are the only copy of
# the keys. Values, sink and recent tokens stay exact, as in every baseline.
# Reconstructing only the SELECTED pages is equivalent to storing every page
# compressed -- attention never touches an unselected page.
#
# Config is the paper's reasoning protocol, identical in every cell. Rank and
# the codec on/off flag are the ONLY things that vary.
#
# eval.reasoning.pred resumes: a complete cell exits at once, a partial one
# continues from where it stopped.
set -u
KV_ROOT="${KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KV_PY="${KV_PY:-python}"
FREEKV_DIR="${FREEKV_DIR:-$KV_ROOT/third_party/FreeKV}"
cd "$FREEKV_DIR/accuracy" || exit 1
export PYTHONUNBUFFERED=1
export OVAL_QUANT=0                   # bf16 records
export KVC_MAX_TOKENS=32768           # short prompt + 16384 generated
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$KV_PY
LOG=$KV_ROOT/logs; OUT=$KV_ROOT/results/codec
mkdir -p "$LOG" "$OUT"

MODELS="${MODELS:-ds-r1-qwen-7b ds-r1-llama-8b ds-r1-qwen-14b}"
DATASETS="${DATASETS:-MATH50 AIME24 GPQA50c}"
RANKS="${RANKS:-1 4 8 16 32}"
GPUS="${GPUS:-0 1 2 3}"

Q="$KV_ROOT/logs/codec_grid_jobs.txt"; : > "$Q"
for M in $MODELS; do
  for D in $DATASETS; do
    echo "$M|$D|full|--method full"                                             >> "$Q"
    echo "$M|$D|sel_r8|--method spec_ret --page_rep oval --eta 0.0 --oval_rank 8" >> "$Q"
    for R in $RANKS; do
      echo "$M|$D|codec_r${R}|--method spec_ret --page_rep oval --eta 0.0 --oval_rank $R --oval_recon" >> "$Q"
    done
  done
done
# mechanism controls: selection-only at every rank on one cell. Rank changes
# BOTH selection and reconstruction; these separate the two.
if [ "${MECHANISM:-1}" = "1" ]; then
  for R in 1 4 16 32; do
    echo "ds-r1-qwen-7b|MATH50|sel_r${R}|--method spec_ret --page_rep oval --eta 0.0 --oval_rank $R" >> "$Q"
  done
fi
echo "=== $(wc -l < "$Q") cells queued  $(date -Iseconds)"

run_job () {   # $1 = "model|dataset|tag|flags", $2 = gpu
  local m d tag flags g
  IFS='|' read -r m d tag flags <<< "$1"; g=$2
  echo "=== START $m/$d/$tag gpu=$g $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES=$g $PY -u -m eval.reasoning.pred \
      --model "$m" --dataset "$d" \
      --GQA_policy avgSM --spec_ret_steps 2 --spec_ret_corr 0.9 \
      --temperature 0.6 --top_p 0.95 --max_gen 16384 \
      --budget 2048 --sink 512 --recent 512 --page_size 32 \
      --seed 42 --out_root_dir "$OUT" $flags \
      > "$LOG/codec_${m}_${d}_${tag}.log" 2>&1
  echo "=== DONE  $m/$d/$tag exit=$? $(date -Iseconds)"
}

declare -A BUSY
while [ -s "$Q" ]; do
  for g in $GPUS; do
    [ -n "${BUSY[$g]:-}" ] && kill -0 "${BUSY[$g]}" 2>/dev/null && continue
    BUSY[$g]=""
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    [ "$u" -gt 2000 ] && continue
    job=$(head -1 "$Q"); [ -z "$job" ] && break
    sed -i '1d' "$Q"
    run_job "$job" "$g" &
    BUSY[$g]=$!
    sleep 25
  done
  sleep 45
done
wait
echo "=== CODEC GRID COMPLETE $(date -Iseconds)"
