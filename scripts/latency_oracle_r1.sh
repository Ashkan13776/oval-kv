#!/usr/bin/env bash
# Decode time of OVAL "oracle" (sweep: decode the sequence once per grid eta
# {0, .25, .5, .75, 1}) vs OVAL rule 1 (decode once with the rule-1 table) on
# a few example sequences, in FreeKV's efficient engine. Same setup as
# scripts/latency_rule1.sh (budget 2048 incl. 512 sink + 512 recent, pages of
# 32, rank 8, greedy, batch 1, one warmup round) except that every run stops
# at its natural EOS (no --ignore_eos, max 16384 tokens), so each eta decodes
# the sequence to whatever length it produces. Per-step times go to
# results/eta_rules/latency/seq/<model>_AIME24_p<idx>_<arm>.json.
#
#   GPU=0 MODELS=ds-r1-qwen-7b PROBS="0 8" ./scripts/latency_oracle_r1.sh
set -u
KV_ROOT="${KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KV_PY="${KV_PY:-python}"
_ENV="$(cd "$(dirname "$(command -v "$KV_PY")")/.." && pwd)"
export PATH="$_ENV/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-$_ENV}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
FREEKV_DIR="${FREEKV_DIR:-$KV_ROOT/third_party/FreeKV}"
MODELS="${MODELS:-ds-r1-qwen-7b ds-r1-llama-8b}"
PROBS="${PROBS:-0 8 1}"                    # AIME24 problems: short, medium, long answers
ETAS="${ETAS:-0.0 0.25 0.5 0.75 1.0}"
GPU="${GPU:-0}"
TASK=AIME24
LOG="$KV_ROOT/logs/latency/seq"; OUT="$KV_ROOT/results/eta_rules/latency/seq"
mkdir -p "$LOG" "$OUT"

cd "$FREEKV_DIR" || exit 1
export PYTHONPATH="$FREEKV_DIR/source" PYTHONUNBUFFERED=1
export FREEKV_WORKSPACE_MIB="${FREEKV_WORKSPACE_MIB:-512}" OVAL_QUANT="${OVAL_QUANT:-0}"

COMMON="--dataset $TASK --max_gen 16384 --warmup 1 --page_size 32 --sink 512 --recent 512
        --budget 2048 --recall_impl cuda_cpy --spec_ret --corr 0.9 --page_rep oval --oval_rank 8"

run () {   # run <model> <prob> <arm> <args...>; retried: see latency_rule1.sh
  local m=$1 p=$2 arm=$3; shift 3
  local tag="${m}_${TASK}_p${p}_${arm}" f="$LOG/${m}_${TASK}_p${p}_${arm}.log" try
  if [ -f "$OUT/$tag.json" ] && [ -z "${FORCE:-}" ]; then echo "=== SKIP $tag"; return; fi
  echo "=== $tag gpu=$GPU $(date -Iseconds)"
  for try in $(seq 1 "${RETRIES:-6}"); do
    if CUDA_VISIBLE_DEVICES=$GPU $KV_PY source/pred.py --model "$m" --data_idx "$p" $COMMON \
         --tbt_out "$OUT/$tag.json.tmp" "$@" > "$f" 2>&1; then
      mv "$OUT/$tag.json.tmp" "$OUT/$tag.json"
      echo "    $(grep -E 'Avg TBT' "$f" | tail -1)  (attempt $try)"
      return
    fi
    cp "$f" "$f.fail$try"; echo "    attempt $try failed; retrying"
  done
  echo "    FAILED after ${RETRIES:-6} attempts (see $f)"
}

for M in $MODELS; do
  for P in $PROBS; do
    run "$M" "$P" r1 --eta_table "$KV_ROOT/results/eta_rules/tables/${M}_${TASK}_r1.json"
    for E in $ETAS; do run "$M" "$P" "eta$E" --eta "$E"; done
  done
done
echo "=== ORACLE/R1 DONE $(date -Iseconds)"
