#!/usr/bin/env bash
# Piece 6: OVAL's full pipeline on long context (docs/6-long-context.md).
#
#   retrieval  rank-KRANK OVAL page scores (avgSM / top-k / spec_ret)  --page_rep oval --oval_rank KRANK
#   keys       selected pages rebuilt from the same rank-KRANK records --oval_recon
#   values     selected pages rebuilt from rank-VRANK PCA records   --oval_vrank VRANK --vbasis pca
#   eta        one per (layer, KV head), rule 1, calibrated per (model, benchmark)
#              on held-out long-context data                       --eta_table
#
# Steps (each resumable; a finished job leaves logs/long_context/done/<job>.ok):
#   1. calib   rule-1 stats per (model, benchmark), streamed     analysis/eta_rules_calibrate_lc.py stats
#   2. rules   eta tables -> results/long_context/eta_rules/tables
#   3. run     LongBench v2 (503 questions, the paper's protocol) and LongGenBench
#              (400 items, FreeKV's command, seeded, LGB_SHARDS shards per model)
#   4. judge   LongGenBench accuracy with the Qwen3-32B judge (vLLM, env $JUDGE_PY)
#   5. score   results/long_context/table.md (+ copies of the raw outputs)
#
# Jobs are queued over GPUS (SLOTS per GPU), longest first; a slot is used only
# when its GPU has MINFREE MiB free (other users' jobs share these cards).
#
#   KV_PY=... GPUS="0 1" scripts/long_context_pipeline.sh
#   MODELS="qwen-2.5-chat-7b" STEPS="calib rules" scripts/long_context_pipeline.sh
set -u
KV_ROOT="${KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KV_PY="${KV_PY:-python}"
JUDGE_PY="${JUDGE_PY:-$HOME/.conda/envs/lgbjudge/bin/python}"
FK="${FREEKV_DIR:-$KV_ROOT/third_party/FreeKV}/accuracy"
MODELS="${MODELS:-qwen-2.5-chat-7b llama-3.1-chat-8b qwen-2.5-chat-14b}"
BENCHES="${BENCHES:-LongBench2 LongGenBench}"
KRANK="${KRANK:-16}"
VRANK="${VRANK:-32}"
SEED="${SEED:-42}"
TOPP="${TOPP:-0.95}"         # LongGenBench top-p: 0.95 as Table 2's OVAL cells (FreeKV's README command leaves 1.0)
LGB_SHARDS="${LGB_SHARDS:-4}"
GPUS="${GPUS:-0 1}"
SLOTS="${SLOTS:-1}"          # jobs per GPU (decode is latency-bound; 7B/8B fit two per 97 GB card)
MINFREE="${MINFREE:-0}"     # floor on top of the per-job need_mib gate
STEPS="${STEPS:-calib rules run judge score}"
OUT="$KV_ROOT/results/long_context"
ETA="$OUT/eta_rules/kr${KRANK}_vr${VRANK}"     # stats and tables per (key rank, value rank)
TABLES="$ETA/tables"
LOG="$KV_ROOT/logs/long_context"; TAG="kr${KRANK}_vr${VRANK}_topp${TOPP}"; DONE="$LOG/done_$TAG"; HB="$LOG/pipeline.log"
mkdir -p "$LOG" "$DONE" "$OUT"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" PYTHONUNBUFFERED=1 OVAL_QUANT=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True KVC_MAX_MEMORY="${KVC_MAX_MEMORY:-90GiB}"
say () { echo "[$(date -Iseconds)] $*" | tee -a "$HB"; }
want () { [[ " $STEPS " == *" $1 "* ]]; }

OVAL="--method spec_ret --page_rep oval --oval_rank $KRANK --oval_recon --oval_vrank $VRANK --vbasis pca --GQA_policy avgSM --spec_ret_steps 2 --page_size 32"
LB2_ARGS="--spec_ret_corr 0.8 --sink 128 --recent 128 --budget 1792 --out_root_dir eval/LongBench2/res"
LGB_ARGS="--spec_ret_corr 0.9 --sink 512 --recent 512 --budget 1024 --max_gen 16000 --temperature 0.95 --top_p $TOPP --seed $SEED --out_root_dir eval/LongGenBench/res"
table () { echo "$TABLES/${1}_${2}_r1.json"; }

# ---------------------------------------------------------------- jobs ----
# job names: calib:<M>:<B>   lb2:<M>   lgb:<M>:<k>   judge:<M>
job_cmd () {
  local j=$1; IFS=: read -r kind M x <<< "$j"
  case $kind in
    calib) echo "cd $KV_ROOT && $KV_PY analysis/eta_rules_calibrate_lc.py stats --model $M --bench $x --rank $KRANK --vrank $VRANK --out $ETA" ;;
    lb2)   echo "cd $FK && KVC_MAX_TOKENS=65536 $KV_PY -m eval.LongBench2.pred --model $M $OVAL --eta_table $(table $M LongBench2) $LB2_ARGS" ;;
    lgb)   echo "cd $FK && KVC_MAX_TOKENS=32768 LGB_SHARD=$x/$LGB_SHARDS $KV_PY $KV_ROOT/scripts/long_context_lgb_pred.py --model $M $OVAL --eta_table $(table $M LongGenBench) $LGB_ARGS" ;;
    judge) echo "cd $FK && LGB_JUDGE_TP=1 LGB_JUDGE_MAXLEN=${LGB_JUDGE_MAXLEN:-8192} $JUDGE_PY eval/LongGenBench/eval.py --data $LOG/judge_$TAG/$M.jsonl --csv $LOG/judge_$TAG/$M.csv" ;;
  esac
}

need_mib () {        # free memory a job needs before it may start on a GPU
  case $1 in
    lgb:qwen-2.5-chat-14b:*|calib:qwen-2.5-chat-14b:*) echo 45000 ;;
    lb2:qwen-2.5-chat-14b)                             echo 62000 ;;
    lb2:*)                                             echo 40000 ;;
    judge:*)                                           echo 80000 ;;
    *)                                                 echo 25000 ;;
  esac
}
gpu_free () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }

declare -A RUNNING   # "gpu.slot" -> pid
declare -A JOBOF     # pid -> job
run_queue () {       # run_queue job...  (SLOTS jobs per GPU, in order)
  local q=("$@") j g s k pid need
  for j in "${q[@]}"; do
    [ -f "$DONE/${j//:/_}.ok" ] && { say "skip $j (done)"; continue; }
    while :; do
      for s in $(seq 1 "$SLOTS"); do for g in $GPUS; do
        k="$g.$s"; pid=${RUNNING[$k]:-}
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then reap "$k"; pid=""; fi
        need=$(need_mib "$j"); [ "$MINFREE" -gt "$need" ] && need=$MINFREE
        if [ -z "$pid" ] && [ "$(gpu_free "$g")" -ge "$need" ]; then
          launch "$j" "$k"; sleep 60; continue 4      # let it allocate before the next free-memory check
        fi
      done; done
      sleep 30
    done
  done
  for k in "${!RUNNING[@]}"; do wait "${RUNNING[$k]}"; reap "$k"; done
}
launch () {
  local j=$1 k=$2 lf="$LOG/${1//:/_}.log"
  say "START $j gpu=${k%.*} slot=${k#*.}"
  ( export CUDA_VISIBLE_DEVICES=${k%.*}; eval "$(job_cmd "$j")" >> "$lf" 2>&1; echo "rc=$?" >> "$lf" ) &
  RUNNING[$k]=$!; JOBOF[$!]=$j
}
reap () {
  local k=$1 pid=${RUNNING[$1]} j lf rc
  j=${JOBOF[$pid]}; lf="$LOG/${j//:/_}.log"
  rc=$(tail -1 "$lf" | sed -n 's/^rc=//p')
  if [ "$rc" = "0" ] && verify "$j"; then touch "$DONE/${j//:/_}.ok"; say "DONE  $j gpu=${k%.*}"
  else say "FAIL  $j gpu=${k%.*} rc=${rc:-?} (see $lf; rerun resumes)"; fi
  unset "RUNNING[$k]"
}
verify () {          # a job is done only if its output is complete
  local kind M x; IFS=: read -r kind M x <<< "$1"
  case $kind in
    calib) [ -s "$ETA/${M}_${x}.json" ] ;;
    lgb)   grep -q "COMPLETE" "$LOG/${1//:/_}.log" ;;
    *)     true ;;
  esac
}

# --------------------------------------------------------------- steps ----
if want calib; then
  jobs=()
  for M in $(echo $MODELS | tr ' ' '\n' | tac); do for B in $BENCHES; do jobs+=("calib:$M:$B"); done; done
  say "calib: ${#jobs[@]} jobs on GPUs [$GPUS]"
  run_queue "${jobs[@]}"
fi

if want rules; then
  $KV_PY "$KV_ROOT/analysis/eta_rules_calibrate_lc.py" rules --models $MODELS --benches $BENCHES --out "$ETA" | tee -a "$HB"
fi

if want run; then
  for M in $MODELS; do for B in $BENCHES; do
    [ -s "$(table $M $B)" ] || { say "ABORT: missing eta table $(table $M $B)"; exit 1; }
  done; done
  jobs=()
  # largest model's LongGenBench shards (16K-token generations) first, then LongBench v2 for every
  # model (64K prefills, ~1-2 h each), so a complete LongBench v2 column lands early; then the
  # remaining LongGenBench shards. LB2_EARLY=0 restores LongGenBench-all-first.
  first=1
  for M in $(echo $MODELS | tr ' ' '\n' | tac); do
    [[ " $BENCHES " == *" LongGenBench "* ]] && for k in $(seq 0 $((LGB_SHARDS - 1))); do jobs+=("lgb:$M:$k"); done
    if [ "$first" = 1 ] && [ "${LB2_EARLY:-1}" = 1 ]; then
      for M2 in $(echo $MODELS | tr ' ' '\n' | tac); do [[ " $BENCHES " == *" LongBench2 "* ]] && jobs+=("lb2:$M2"); done
    fi
    first=0
  done
  [ "${LB2_EARLY:-1}" = 1 ] || for M in $(echo $MODELS | tr ' ' '\n' | tac); do [[ " $BENCHES " == *" LongBench2 "* ]] && jobs+=("lb2:$M"); done
  say "run: ${#jobs[@]} jobs on GPUs [$GPUS] x $SLOTS slots, KRANK=$KRANK VRANK=$VRANK, LongGenBench top-p $TOPP seed $SEED"
  run_queue "${jobs[@]}"
fi

lgb_out () {         # canonical LongGenBench output for model $1
  ls "$FK"/eval/LongGenBench/results_short/"$1"-spec_ret-*oval_r${KRANK}_etab*_recon1_vr${VRANK}_vpca*-t0.95-topp${TOPP}-Mg16000.jsonl 2>/dev/null \
    | grep -v part | head -1
}

if want judge; then
  JD="$LOG/judge_$TAG"; mkdir -p "$JD"
  jobs=()
  for M in $MODELS; do
    [[ " $BENCHES " == *" LongGenBench "* ]] || continue
    c=$(ls "$FK"/eval/LongGenBench/results_short/"$M"-spec_ret-*oval_r${KRANK}_etab*_vr${VRANK}_vpca*-topp${TOPP}-*.part0of${LGB_SHARDS}.json 2>/dev/null | head -1)
    [ -n "$c" ] || { say "judge: no LongGenBench output for $M"; continue; }
    canon=${c/.part0of${LGB_SHARDS}.json/.jsonl}
    [ -s "$canon" ] || (cd "$FK" && $KV_PY "$KV_ROOT/scripts/long_context_lgb_pred.py" --merge "$canon" "$LGB_SHARDS") | tee -a "$HB"
    # the judge writes its verdicts back into the file it reads: give it a copy
    [ -s "$JD/$M.jsonl" ] || cp "$canon" "$JD/$M.jsonl"
    jobs+=("judge:$M")
  done
  MINFREE_SAVE=$MINFREE; SLOTS_SAVE=$SLOTS; SLOTS=1   # Qwen3-32B bf16 on one card (need_mib 80000)
  run_queue "${jobs[@]}"
  MINFREE=$MINFREE_SAVE; SLOTS=$SLOTS_SAVE
fi

if want score; then
  $KV_PY "$KV_ROOT/analysis/score_long_context.py" --models $MODELS --krank $KRANK --vrank $VRANK --fk "$FK" \
      --topp $TOPP --judge_dir "$LOG/judge_$TAG" --out "$OUT" | tee -a "$HB"
fi
say "DONE ($STEPS)"
