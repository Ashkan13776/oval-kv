#!/usr/bin/env bash
# Decode latency of eta rule 1 (per-(layer, KV head) eta table) against the
# other arms, in FreeKV's efficient engine (third_party/FreeKV/source).
#
# Scenario: long generation, as in the reasoning evals. AIME24 problem 0, the
# harness prompt ("<think>\n" appended for ds-r1, no chat template), exactly
# MAX_GEN decoded tokens for every arm (--ignore_eos), greedy, batch 1, one
# warmup round, then a timed round. Metric: pred.py's "Avg TBT (decode)", the
# mean ms per decode step of the timed round excluding its first step; the
# per-step times go to results/eta_rules/latency/<model>_<arm>.json.
#
# Budget semantics are the perf engine's for every arm: --budget 2048 is the
# TOTAL (64 pages of 32 = top-k + 16 sink + 16 recent), unlike the accuracy
# harness, where sink and recent sit outside the budget.
#
# Arms (ARMS overrides):
#   full      --budget 0: no eviction or recall, every KV on the GPU
#   arkvale   ArkVale recall
#   raas      RaaS dropping
#   freekv    FreeKV: quest min/max pages, speculative retrieval, corr 0.9
#   oval_fix  FreeKV + OVAL pages, rank 8, the fixed AIME24 eta (1.0)
#   oval_r1   FreeKV + OVAL pages, rank 8, rule-1 table for <model>/AIME24
#
# eta enters only the page-record build at page creation; the per-step scoring
# kernel and attention never read it, so oval_fix vs oval_r1 isolates that cost.
# The accuracy runs also rebuild the selected keys and values from rank-8
# records (key/value codec); the perf engine has no kernel for that, so these
# arms recall exact KV for the selected pages.
#
#   GPU=0 ./scripts/latency_rule1.sh             # all arms, both models
#   MODELS=ds-r1-qwen-7b ARMS="freekv oval_r1" MAX_GEN=256 ./scripts/latency_rule1.sh
set -u
KV_ROOT="${KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KV_PY="${KV_PY:-python}"
# flashinfer JITs its kernels at first use: it needs the env's ninja and nvcc
_ENV="$(cd "$(dirname "$(command -v "$KV_PY")")/.." && pwd)"
export PATH="$_ENV/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-$_ENV}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
FREEKV_DIR="${FREEKV_DIR:-$KV_ROOT/third_party/FreeKV}"
MODELS="${MODELS:-ds-r1-qwen-7b ds-r1-llama-8b}"
ARMS="${ARMS:-full arkvale raas freekv oval_fix oval_r1}"
MAX_GEN="${MAX_GEN:-16384}"
GPU="${GPU:-0}"
FIXED_ETA="${FIXED_ETA:-1.0}"             # the AIME24 fixed-eta baseline
TASK=AIME24
LOG="$KV_ROOT/logs/latency"; OUT="$KV_ROOT/results/eta_rules/latency"
mkdir -p "$LOG" "$OUT"

cd "$FREEKV_DIR" || exit 1
export PYTHONPATH="$FREEKV_DIR/source"
export PYTHONUNBUFFERED=1
export FREEKV_WORKSPACE_MIB="${FREEKV_WORKSPACE_MIB:-512}"
export OVAL_QUANT="${OVAL_QUANT:-0}"      # bf16 records, as in the accuracy runs

COMMON="--dataset $TASK --data_idx 0 --max_gen $MAX_GEN --ignore_eos --warmup 1
        --page_size 32 --sink 512 --recent 512"

run () {   # run <model> <arm> <args...>
  local m=$1 arm=$2; shift 2
  local tag="${m}_${arm}" f="$LOG/${m}_${arm}.log"
  [ "$MAX_GEN" != 16384 ] && tag="${tag}_g${MAX_GEN}" && f="$LOG/${tag}.log"
  if [ -f "$OUT/$tag.json" ] && [ -z "${FORCE:-}" ]; then echo "=== SKIP $tag (done)"; return; fi
  echo "=== $tag gpu=$GPU $(date -Iseconds)"
  # The engine intermittently hits a device-side index assert in the FIRST
  # generation of a process (the warmup round; plain FreeKV too, independent of
  # OVAL). It never hit a timed round in ~25 test runs, so a crashed run is
  # simply re-run; only a run that completes writes its timings.
  local try
  for try in $(seq 1 "${RETRIES:-6}"); do
    if CUDA_VISIBLE_DEVICES=$GPU $KV_PY source/pred.py --model "$m" $COMMON \
         --tbt_out "$OUT/$tag.json.tmp" "$@" > "$f" 2>&1; then
      mv "$OUT/$tag.json.tmp" "$OUT/$tag.json"
      echo "    $(grep -E 'Avg TBT' "$f" | tail -1)  (attempt $try)"
      return
    fi
    cp "$f" "$f.fail$try"
    echo "    attempt $try failed ($(grep -m1 -oE 'Assertion|Error[^:]*' "$f" || echo '?')); retrying"
  done
  echo "    FAILED after ${RETRIES:-6} attempts (see $f)"
}

for M in $MODELS; do
  TAB="$KV_ROOT/results/eta_rules/tables/${M}_${TASK}_r1.json"
  for A in $ARMS; do
    case $A in
      full)     run "$M" full     --budget 0 ;;
      arkvale)  run "$M" arkvale  --budget 2048 --recall_impl arkvale ;;
      raas)     run "$M" raas     --budget 2048 --recall_impl cuda_cpy --method raas ;;
      freekv)   run "$M" freekv   --budget 2048 --recall_impl cuda_cpy --spec_ret --corr 0.9 ;;
      oval_fix) run "$M" oval_fix --budget 2048 --recall_impl cuda_cpy --spec_ret --corr 0.9 \
                                  --page_rep oval --oval_rank 8 --eta "$FIXED_ETA" ;;
      oval_r1)  run "$M" oval_r1  --budget 2048 --recall_impl cuda_cpy --spec_ret --corr 0.9 \
                                  --page_rep oval --oval_rank 8 --eta_table "$TAB" ;;
      *) echo "unknown arm $A"; exit 1 ;;
    esac
  done
done
echo "=== LATENCY DONE $(date -Iseconds)"
