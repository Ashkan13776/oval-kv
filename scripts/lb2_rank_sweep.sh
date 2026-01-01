#!/bin/bash
# LongBench v2 eta sweep, Qwen-2.5-14B-Instruct, OVAL page basis.
#
# Completes the eta column for Table 2's third model. eta=0.25 is already done
# (Overall 32.21); this adds 0.0 / 0.5 / 0.75 / 1.0.
#
# Protocol is the paper's LongBench v2 setting, matching their README command:
#   --sink 128 --recent 128 --budget 1792   -> B = 2048 total
#   tau = 0.8 (long-input), page size 32, avgSM, spec_ret_steps 2, greedy
#   inputs truncated to 64K  ("For all methods, we truncated the inputs to 64K")
#
# NOTE their append_to_json_file APPENDS -- each cell's output file is removed
# before its run, or a rerun silently produces 1006 mixed records.
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

for E in "${ETAS:-0.0 0.5 0.75 1.0}"; do :; done
ETAS="${ETAS:-0.25}"
RANKS="${RANKS:-4 16}"

for R in $RANKS; do
 for E in $ETAS; do
  OUT="eval/LongBench2/results/qwen-2.5-chat-14b-spec_ret-s128-r128-1.00-manual-p32-b1792-avgSM-oab_r${R}_eta${E}_q0-pQ2-llb0-pQcs0.8-skl1.jsonl"
  n=$([ -f "$OUT" ] && grep -c . "$OUT" || echo 0)
  if [ "$n" -ge 503 ]; then
    echo "=== LB2RANK rank=$R eta=$E SKIP (already $n records)"
    continue
  fi
  # partial or stale file: remove, since their writer appends
  [ -f "$OUT" ] && { echo "=== LB2RANK rank=$R eta=$E removing partial ($n records)"; rm -f "$OUT"; }
  echo "=== LB2RANK rank=$R eta=$E START $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES=0 $PY -m eval.LongBench2.pred \
    --model qwen-2.5-chat-14b --method spec_ret \
    --page_rep oval --eta "$E" --oval_rank "$R" \
    --GQA_policy avgSM --spec_ret_steps 2 --spec_ret_corr 0.8 \
    --sink 128 --recent 128 --budget 1792 \
    --out_root_dir eval/LongBench2/res \
    > "$LOG/lb2_qwen14b_r${R}_eta${E}.log" 2>&1
  rc=$?
  got=$([ -f "$OUT" ] && grep -c . "$OUT" || echo 0)
  echo "=== LB2RANK rank=$R eta=$E DONE rc=$rc records=$got $(date -Iseconds)"
done
 done
echo "=== LB2RANK ALL DONE $(date -Iseconds)"
