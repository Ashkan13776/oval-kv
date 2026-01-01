#!/usr/bin/env bash
# Finish the three cells that the sweep could not complete.
#
# Each died with "CUDA error: the launch timed out and was terminated" at the
# same problem on every attempt. All three failures happened while TWO cells
# ran concurrently, so this runs them ONE AT A TIME on a single GPU: if
# contention or thermals were contributing, they pass now; if a specific
# problem is pathological, they fail at the same index and we have our answer.
#
# pred.py resumes per problem, so each picks up where it stopped.
set -u
KV_ROOT="${KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KV_PY="${KV_PY:-python}"
FREEKV_DIR="${FREEKV_DIR:-$KV_ROOT/third_party/FreeKV}"
cd "$FREEKV_DIR/accuracy" || exit 1
export PYTHONUNBUFFERED=1 OVAL_QUANT=0 KVC_MAX_TOKENS=32768
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G="${GPU:-0}"; OUT="$KV_ROOT/results/codec"; L="$KV_ROOT/logs"

run () {  # run <dataset> <seed> <eta>
  local d=$1 s=$2 e=$3
  echo "=== STUCK $d/s$s/eta$e START $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES=$G $KV_PY -u -m eval.reasoning.pred \
      --model ds-r1-qwen-14b --dataset "$d" \
      --GQA_policy avgSM --spec_ret_steps 2 --spec_ret_corr 0.9 \
      --temperature 0.6 --top_p 0.95 --max_gen 16384 \
      --budget 2048 --sink 512 --recent 512 --page_size 32 \
      --seed "$s" --out_root_dir "$OUT" \
      --method spec_ret --page_rep oval --eta "$e" --oval_rank 8 --oval_recon \
      > "$L/stuck_${d}_s${s}_eta${e}.log" 2>&1
  echo "=== STUCK $d/s$s/eta$e DONE exit=$? $(date -Iseconds)"
}

run MATH50  47 0.5
run AIME24  46 0.75
run GPQA50c 46 0.25
echo "=== STUCK CELLS ALL ATTEMPTED $(date -Iseconds)"
