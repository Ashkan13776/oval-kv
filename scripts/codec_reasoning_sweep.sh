#!/usr/bin/env bash
# Long-reasoning sweep for the OVAL key codec: eta x seed grid at rank 8.
#
# Produces avg@k / pass@k over k seeds for
#   dense    FullKV                                     (DENSE=1)
#   codec    OVAL retrieval + compression, --oval_recon (always)
# at every eta requested. Selection-only (retrieval without compression) is
# opt-in via WITH_SEL=1.
#
# Only (dataset, eta) pairs that do NOT already have all SEEDS on disk are
# queued, so the script is safe to re-run, safe to run alongside another
# scheduler, and never lets two processes write the same .jsonl.
#
# Config is the paper's reasoning protocol and is identical in every cell;
# eta, seed and the codec flag are the only things that vary.
#
#   KV_ROOT      repo root            (default: this script's parent)
#   KV_PY        python to use        (default: python)
#   FREEKV_DIR   patched FreeKV tree  (default: $KV_ROOT/third_party/FreeKV)
#   MODEL        ds-r1-qwen-7b | ds-r1-llama-8b | ds-r1-qwen-14b
#   DATASETS     default "MATH50 AIME24 GPQA50c"
#   ETAS         default "0.0 0.25 0.5 0.75 1.0"
#   SEEDS        default "42 43 44 45 46 47 48 49"   (k = number of seeds)
#   GPUS         default "0 1 2 3"
#   DENSE=1      also run the FullKV control
#   WITH_SEL=1   also run selection-only (no compression)
set -u
KV_ROOT="${KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KV_PY="${KV_PY:-python}"
FREEKV_DIR="${FREEKV_DIR:-$KV_ROOT/third_party/FreeKV}"
cd "$FREEKV_DIR/accuracy" || { echo "no FreeKV at $FREEKV_DIR -- run ./setup.sh"; exit 1; }

export PYTHONUNBUFFERED=1
export OVAL_QUANT=0                    # bfloat16 records
export KVC_MAX_TOKENS=32768            # short prompt + 16384 generated
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

M="${MODEL:?set MODEL, e.g. MODEL=ds-r1-llama-8b}"
DATASETS="${DATASETS:-MATH50 AIME24 GPQA50c}"
ETAS="${ETAS:-0.0 0.25 0.5 0.75 1.0}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49}"
GPUS="${GPUS:-0 1 2 3}"
NSEED=$(echo $SEEDS | wc -w)

LOG="$KV_ROOT/logs"; OUT="$KV_ROOT/results/codec"
RES="$OUT/${M}-spec_ret"; RESF="$OUT/${M}-full"
mkdir -p "$LOG" "$OUT"
Q="$LOG/codec_reasoning_jobs.txt"; : > "$Q"

for D in $DATASETS; do
  if [ "${DENSE:-0}" = "1" ]; then
    have=$(ls "$RESF"/${D}-*.jsonl 2>/dev/null | wc -l)
    if [ "$have" -ge "$NSEED" ]; then echo "   skip $D dense (already $have/$NSEED)"
    else for S in $SEEDS; do echo "$D|$S|-|dense|--method full" >> "$Q"; done; fi
  fi
  for E in $ETAS; do
    if [ "${WITH_SEL:-0}" = "1" ]; then
      have=$(ls "$RES"/${D}-*oval_r8_eta${E}_q0-pQ2*.jsonl 2>/dev/null | wc -l)
      if [ "$have" -ge "$NSEED" ]; then echo "   skip $D eta=$E sel (already $have/$NSEED)"
      else for S in $SEEDS; do
        echo "$D|$S|$E|sel|--method spec_ret --page_rep oval --eta $E --oval_rank 8" >> "$Q"
      done; fi
    fi
    have=$(ls "$RES"/${D}-*oval_r8_eta${E}_q0_recon1*.jsonl 2>/dev/null | wc -l)
    if [ "$have" -ge "$NSEED" ]; then echo "   skip $D eta=$E codec (already $have/$NSEED)"
    else for S in $SEEDS; do
      echo "$D|$S|$E|codec|--method spec_ret --page_rep oval --eta $E --oval_rank 8 --oval_recon" >> "$Q"
    done; fi
  done
done
echo "=== $(wc -l < "$Q") cells queued for $M  $(date -Iseconds)"

run_job () {
  local d s e tag flags g
  IFS='|' read -r d s e tag flags <<< "$1"; g=$2
  echo "=== START $M/$d/s$s/eta$e/$tag gpu=$g $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES=$g $KV_PY -u -m eval.reasoning.pred \
      --model "$M" --dataset "$d" \
      --GQA_policy avgSM --spec_ret_steps 2 --spec_ret_corr 0.9 \
      --temperature 0.6 --top_p 0.95 --max_gen 16384 \
      --budget 2048 --sink 512 --recent 512 --page_size 32 \
      --seed "$s" --out_root_dir "$OUT" $flags \
      > "$LOG/codec_${M}_${d}_s${s}_eta${e}_${tag}.log" 2>&1
  echo "=== DONE  $M/$d/s$s/eta$e/$tag exit=$? $(date -Iseconds)"
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
echo "=== SWEEP COMPLETE for $M  $(date -Iseconds)"
