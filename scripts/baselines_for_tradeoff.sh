#!/bin/bash
# Baselines needed to turn the accuracy-vs-latency plot into a real frontier.
#
# The plot needs BOTH axes for the same (method, model). Today only FreeKV and
# OVAL have both. This script fills the gaps, cheapest-first:
#
#   1. RaaS latency on Llama-3.1-8B  (~15 min) -- the perf sweep ran RaaS on
#      Qwen only, so RaaS has no x-coordinate on the Llama panel.
#   2. ArkVale accuracy on InfiniteBench (~16 h) -- ArkVale already HAS a
#      latency point (76.86 ms); it is missing only the y-coordinate, so this
#      single run converts a dashed rule into a plotted method.
#   3. RaaS accuracy on InfiniteBench (~16 h) -- pairs with (1).
#   4. FullKV on InfiniteBench (~16 h) -- the dense ceiling. Also settles which
#      tasks beyond kv_retrieval are DEAD on this model (FullKV <= sparse), the
#      open question from the kv-retr retraction.
#
# Everything in 2-4 is held identical to the two arms already run: B=2048 as
# 128+128+1792, tau 0.8, page 32, avgSM, spec_ret_steps 2, greedy, 11 tasks,
# seeded-random 50 records seed 42.
set -u
KV_ROOT="${KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KV_PY="${KV_PY:-python}"
FREEKV_DIR="${FREEKV_DIR:-$KV_ROOT/third_party/FreeKV}"

K=$KV_ROOT
PY=$KV_PY
L=$K/logs
export HF_HOME=$HOME/.cache/huggingface PYTHONUNBUFFERED=1
# accuracy path stores bf16 records; this also fixes the _q0 tag in output paths
export OVAL_QUANT=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=llama-3.1-chat-8b

# ---- 1. RaaS latency, Llama, long-input + long-gen, bs1 -------------------
echo "=== [1/4] RaaS latency START $(date -Iseconds)"
MODELS="$M" BSZS=1 FIG6_GPU=0 bash $K/env/fig6_latency_all.sh \
  > $L/fig6_raas_llama.log 2>&1
echo "=== [1/4] RaaS latency DONE rc=$? $(date -Iseconds)"

# ---- 2-4. accuracy arms ---------------------------------------------------
cd $K/vendor/FreeKV/accuracy || exit 1
acc() {  # acc <tag> <args...>
  local tag=$1; shift
  echo "=== ACC $tag START $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES=0 $PY -m eval.InfiniteBench.pred \
    --model "$M" "$@" --out_root_dir eval/InfiniteBench/res \
    > "$L/ib_${tag}.log" 2>&1
  echo "=== ACC $tag DONE rc=$? $(date -Iseconds)"
}

SEL="--GQA_policy avgSM --sink 128 --recent 128 --budget 1792"
acc arkvale --method arkv --page_rep arkv $SEL
acc raas    --method raas $SEL
acc full    --method full
echo "=== BASELINES ALL DONE $(date -Iseconds)"
