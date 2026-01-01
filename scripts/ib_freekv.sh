#!/bin/bash
# InfiniteBench in FreeKV's harness. Arms: quest (= FreeKV's own page rep) and
# oval (ours). Same model, same harness, same protocol -- the within-harness
# control that the LongBench v2 14B cell showed is essential.
#
# Budget matches their long-input convention (their LongBench2 command):
#   sink 128 + recent 128 + budget 1792 = B 2048,  tau 0.8, page 32, greedy
# Protocol matches our LOCKS-side InfiniteBench runs: 11 tasks, seeded-random
# 50 records (seed 42), reference prompts, middle truncation.
set -u
KV_ROOT="${KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KV_PY="${KV_PY:-python}"
FREEKV_DIR="${FREEKV_DIR:-$KV_ROOT/third_party/FreeKV}"

cd $FREEKV_DIR/accuracy || exit 1
export HF_HOME=$HOME/.cache/huggingface PYTHONUNBUFFERED=1
# accuracy path stores bf16 records; this also fixes the _q0 tag in output paths
export OVAL_QUANT=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$KV_PY
L=$KV_ROOT/logs
M="${MODEL:-llama-3.1-chat-8b}"
for REP in ${REPS:-quest oval}; do
  EXTRA=""; [ "$REP" = "oval" ] && EXTRA="--eta ${ETA:-0.25} --oval_rank ${RANK:-8}"
  echo "=== IB $REP START $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES=0 $PY -m eval.InfiniteBench.pred \
    --model "$M" --method spec_ret --page_rep "$REP" $EXTRA \
    --GQA_policy avgSM --spec_ret_steps 2 --spec_ret_corr 0.8 \
    --sink 128 --recent 128 --budget 1792 \
    --out_root_dir eval/InfiniteBench/res > "$L/ib_${REP}.log" 2>&1
  echo "=== IB $REP DONE rc=$? $(date -Iseconds)"
done
echo "=== IB ALL DONE $(date -Iseconds)"
