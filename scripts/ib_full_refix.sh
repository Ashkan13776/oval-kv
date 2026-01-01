#!/bin/bash
# Re-run FullKV on InfiniteBench after the tuple-KV cache fix.
#
# The first FullKV run scored 9.39 because kvc/patch/tuple_kv_cache.py only
# wrote the KV cache on the int8 path; in bf16 every decode step saw an empty
# cache. Prefill was correct, so the first token was right and the rest was
# garbage. The old predictions are kept under BROKEN-full-precachefix/ as
# evidence, not deleted.
set -u
KV_ROOT="${KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KV_PY="${KV_PY:-python}"
FREEKV_DIR="${FREEKV_DIR:-$KV_ROOT/third_party/FreeKV}"

K=$KV_ROOT
export HF_HOME=$HOME/.cache/huggingface PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $K/vendor/FreeKV/accuracy || exit 1
echo "=== FULLKV refix START $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=0 $KV_PY \
  -m eval.InfiniteBench.pred --model llama-3.1-chat-8b --method full \
  --out_root_dir eval/InfiniteBench/res > $K/logs/ib_full_refix.log 2>&1
echo "=== FULLKV refix DONE rc=$? $(date -Iseconds)"
