#!/bin/bash
# MVP: decode latency on the SAME LongBench-v2 prompts the accuracy harness
# scores, so one benchmark supplies both axes of the trade-off figure.
#
# One median-length record per length bucket. Latency is near-deterministic
# given (model, prompt length, budget) -- it does not depend on content -- so a
# single record per bucket is sound here, unlike accuracy.
#
# EXPECTED: Qwen-2.5-14B is capped at max_len 65536 with head/tail truncation,
# and the medium/long medians are 139K/402K tokens, so BOTH arrive at the model
# as exactly 64K. Their latencies should come out equal; that is a property of
# the truncation, not a measurement error. Only `short` (~27K) differs.
#
# BUDGET: the perf path CARVES sink/window out of --budget (opposite of the
# accuracy path). 2048/32 = 64 pages; topk 32 + 16 sink + 16 window = 64 pages
# = B = 2048, matching the accuracy runs' 128+128+1792.
set -u
KV_ROOT="${KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KV_PY="${KV_PY:-python}"
FREEKV_DIR="${FREEKV_DIR:-$KV_ROOT/third_party/FreeKV}"

cd $FREEKV_DIR || exit 1
export PYTHONPATH=$FREEKV_DIR/source
export HF_HOME=${HF_HOME} PYTHONUNBUFFERED=1
export FREEKV_WORKSPACE_MIB=${FREEKV_WORKSPACE_MIB:-512}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Upstream hardcodes a 6 GiB GPU page pool; Qwen-14B's weights (28 GB) plus that
# pool do not fit beside another user on this card. B=2048 needs ~0.4 GiB, so
# 2 GiB is ample. NOTE: this changes the offloading regime, so these numbers are
# NOT comparable to the Fig-6 runs (which used the 6 GiB default) -- only the
# arm-vs-arm comparison inside this experiment is valid.
export FREEKV_GPU_POOL_GIB=${FREEKV_GPU_POOL_GIB:-2}
PY=$KV_PY
OUT=$KV_ROOT/results/freekv/lb2_latency
mkdir -p "$OUT"
M=qwen-2.5-chat-14b

# bucket:median-record-index (median by context chars, computed over the bucket)
declare -A IDX=( [short]=148 [medium]=190 [long]=39 )

run() {  # run <bucket> <arm> <extra...>
  local b=$1 arm=$2; shift 2
  local tag="${M}_${b}_${arm}"
  [ -f "$OUT/$tag.done" ] && { echo "=== SKIP $tag"; return; }
  echo "=== LB2LAT $tag  $(date -Iseconds)"
  FREEKV_RUN_TAG="$tag" CUDA_VISIBLE_DEVICES=${LB2_GPU:-0} $PY source/pred.py \
      --model "$M" --dataset longbench2 --lb2_length "$b" \
      --data_idx "${IDX[$b]}" --max_length 65536 --max_gen 128 \
      --warmup 1 --budget 2048 --sink 512 --recent 512 \
      --recall_impl cuda_cpy "$@" > "$OUT/$tag.log" 2>&1 \
    && touch "$OUT/$tag.done" \
    && echo "    $(grep -E 'prompt_length=' "$OUT/$tag.log" | head -1 | grep -oE 'prompt_length=[0-9]+')  $(grep -E 'Avg TBT' "$OUT/$tag.log" | tail -1)" \
    || echo "    FAILED (see $OUT/$tag.log)"
}

for B in short medium long; do
  run "$B" freekv --spec_ret --corr 0.8
  run "$B" oval   --spec_ret --corr 0.8 --page_rep locks --eta 0.25 --locks_rank 8
done
echo "=== LB2LAT ALL DONE $(date -Iseconds)"
