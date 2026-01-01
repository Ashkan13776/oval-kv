#!/bin/bash
# Figure 6 of the FreeKV paper: end-to-end latency, FreeKV vs ArkVale.
#
# Their setup (Sec 5.3): A100 40GB + AMD 7302 via PCIe Gen4, Qwen-2.5-7B and
# Llama-3.1-8B, long-input (32K in / 512 out) and long-generation (600 in /
# 16K out), B=2048, S=W=512, tau=0.8 long-input / 0.9 long-generation.
#
# DEVIATIONS, all forced and all recorded:
#  * Hardware: RTX PRO 6000 Blackwell 97GB, not A100 40GB. Their headline is an
#    OFFLOADING result; with 97GB the memory pressure their design relieves is
#    much weaker, so absolute speedups are not expected to transfer.
#  * Only ArkVale is reproducible. The released perf path implements
#    arkvale/torch_cpy/cuda_cpy only -- ShadowKV, InfiniGen, RaaS and
#    RazorAttention are in the paper's Fig 6 but not in the artifact.
#  * spec_ret required a fix to their thread-pool handoff (see
#    notes/freekv-port.md); without it FreeKV's arm emits corrupted text.
#
# BUDGET: the perf path and the accuracy path use OPPOSITE conventions for
# --budget. Here sink/window are carved OUT of the budget:
#   page_budget = 2048/32 = 64 ; topk = (64-1) - 16 - (16-1) = 32
#   attended    = 32 + 16 + 16 = 64 pages = 2048 tokens = the paper's B
# so the perf-path DEFAULTS are already correct and we pass them explicitly.
set -u
KV_ROOT="${KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KV_PY="${KV_PY:-python}"
FREEKV_DIR="${FREEKV_DIR:-$KV_ROOT/third_party/FreeKV}"

cd $FREEKV_DIR || exit 1
export PYTHONPATH=$FREEKV_DIR/source
export HF_HOME=${HF_HOME}
export PYTHONUNBUFFERED=1
# their fixed 64MiB FlashInfer workspace is too small for 16K-token generation
export FREEKV_WORKSPACE_MIB=${FREEKV_WORKSPACE_MIB:-512}
PY=$KV_PY
OUT=$KV_ROOT/results/freekv/fig6
mkdir -p "$OUT"

MODELS="${MODELS:-qwen-2.5-chat-7b llama-3.1-chat-8b}"
BSZS="${BSZS:-1 2 4}"

run() {   # run <model> <scenario> <method> <bsz> <extra...>
  local m=$1 sc=$2 meth=$3 b=$4; shift 4
  local tag="${m}_${sc}_${meth}_bs${b}"
  local f="$OUT/$tag.log"
  if [ -f "$OUT/$tag.done" ]; then echo "=== SKIP $tag"; return; fi
  echo "=== FIG6 $tag  $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES=${FIG6_GPU:-0} $PY source/pred.py \
      --model "$m" --data_idx 0 --warmup 1 \
      --budget 2048 --sink 512 --recent 512 \
      --repeat_bsz "$b" "$@" > "$f" 2>&1 \
    && touch "$OUT/$tag.done" \
    && echo "    $(grep -E 'Avg TBT' "$f" | tail -1)" \
    || echo "    FAILED (see $f)"
}

for M in $MODELS; do
  for B in $BSZS; do
    # long-input: 32K prompt, 512 generated, tau=0.8
    LI="--dataset gov_report --max_gen 512"
    run "$M" longinput arkvale   "$B" $LI --recall_impl arkvale
    run "$M" longinput raas      "$B" $LI --recall_impl cuda_cpy --method raas
    run "$M" longinput freekv    "$B" $LI --recall_impl cuda_cpy --spec_ret --corr 0.8
    run "$M" longinput locks0    "$B" $LI --recall_impl cuda_cpy --spec_ret --corr 0.8 --page_rep locks --eta 0.0
    run "$M" longinput locks50   "$B" $LI --recall_impl cuda_cpy --spec_ret --corr 0.8 --page_rep locks --eta 0.5
    run "$M" longinput locks100  "$B" $LI --recall_impl cuda_cpy --spec_ret --corr 0.8 --page_rep locks --eta 1.0
    # long-generation: 600 prompt, 16K generated, tau=0.9
    LG="--dataset lgbench --max_gen 16384"
    run "$M" longgen   arkvale   "$B" $LG --recall_impl arkvale
    run "$M" longgen   raas      "$B" $LG --recall_impl cuda_cpy --method raas
    run "$M" longgen   freekv    "$B" $LG --recall_impl cuda_cpy --spec_ret --corr 0.9
    run "$M" longgen   locks0    "$B" $LG --recall_impl cuda_cpy --spec_ret --corr 0.9 --page_rep locks --eta 0.0
    run "$M" longgen   locks50   "$B" $LG --recall_impl cuda_cpy --spec_ret --corr 0.9 --page_rep locks --eta 0.5
    run "$M" longgen   locks100  "$B" $LG --recall_impl cuda_cpy --spec_ret --corr 0.9 --page_rep locks --eta 1.0
  done
done
echo "=== FIG6 ALL DONE  $(date -Iseconds)"
