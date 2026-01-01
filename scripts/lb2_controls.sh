#!/bin/bash
# Controls for the Qwen-2.5-14B LongBench v2 cell.
#
# Our OVAL arms came in 0.4-1.2 BELOW the paper's FullKV (33.40), whereas on
# Llama-8B and Qwen-7B our method came in ABOVE it. Before concluding anything
# about the method on this model we need OUR OWN baselines in THIS harness,
# rather than comparing against the paper's published column:
#
#   1. page_rep=quest  -> this IS FreeKV's own page representation. If our run
#      reproduces their 34.19, the harness is faithful and the OVAL deficit is
#      real. If it also lands ~32-33, everything is shifted and the deficit is
#      a harness/protocol artifact, not the method.
#
#   2. method=full     -> FullKV, their 33.40. Same question for the dense
#      reference.
#
# Everything else is held identical to the OVAL arms: B=2048 as 128+128+1792,
# tau=0.8, page 32, avgSM, spec_ret_steps 2, 64K truncation, greedy.
set -u
KV_ROOT="${KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KV_PY="${KV_PY:-python}"
FREEKV_DIR="${FREEKV_DIR:-$KV_ROOT/third_party/FreeKV}"

cd $FREEKV_DIR/accuracy || exit 1
export HF_HOME=$HOME/.cache/huggingface
export PYTHONUNBUFFERED=1
export KVC_MAX_TOKENS=65536
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$KV_PY
LOG=$KV_ROOT/logs
R=eval/LongBench2/results

run() {  # run <tag> <expected-file-substring> <args...>
  local tag=$1 pat=$2; shift 2
  local f
  f=$(ls $R/*"$pat"*.jsonl 2>/dev/null | head -1)
  local n=0; [ -n "$f" ] && n=$(grep -c . "$f")
  if [ "$n" -ge 503 ]; then echo "=== LB2CTL $tag SKIP (already $n)"; return; fi
  [ -n "$f" ] && { echo "=== LB2CTL $tag removing partial ($n)"; rm -f "$f"; }
  echo "=== LB2CTL $tag START $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES=0 $PY -m eval.LongBench2.pred \
    --model qwen-2.5-chat-14b "$@" \
    --out_root_dir eval/LongBench2/res > "$LOG/lb2_ctl_$tag.log" 2>&1
  local rc=$?
  f=$(ls $R/*"$pat"*.jsonl 2>/dev/null | head -1)
  n=0; [ -n "$f" ] && n=$(grep -c . "$f")
  echo "=== LB2CTL $tag DONE rc=$rc records=$n $(date -Iseconds)"
}

# 1. FreeKV's own page representation (quest min/max digest) -- expect ~34.19
run quest "avgSM-pQ2" --method spec_ret --page_rep quest \
    --GQA_policy avgSM --spec_ret_steps 2 --spec_ret_corr 0.8 \
    --sink 128 --recent 128 --budget 1792

# 2. FullKV dense reference -- expect ~33.40
run full "qwen-2.5-chat-14b-full" --method full

echo "=== LB2CTL ALL DONE $(date -Iseconds)"
