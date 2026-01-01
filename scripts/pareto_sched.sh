#!/usr/bin/env bash
# Slot-based scheduler: TWO jobs per GPU instead of one.
#
# Each cell uses ~17 GiB of a 49 GiB card, so the original gate ("skip this GPU
# if anything at all is on it") left ~32 GiB idle per card. Two jobs fit with
# ~15 GiB headroom. Decode is memory-bandwidth bound rather than SM bound, so
# two co-resident jobs do not halve each other's speed -- expect ~1.5x
# throughput, not 2x, which still turns 22 days of queue into roughly 14.
#
# Pops are serialised with flock. Two schedulers doing `head -1` + `sed -i 1d`
# without a lock will hand the same cell to two GPUs, which has already
# happened once on this project.
set -u
cd /home/shahbaa/kv-compression/vendor/FreeKV/accuracy || exit 1
export PATH=$HOME/miniconda3/envs/freekv/bin:$PATH
export HF_HOME=$HOME/.cache/huggingface
export PYTHONUNBUFFERED=1 OVAL_QUANT=0 KVC_MAX_TOKENS=32768
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$HOME/miniconda3/envs/freekv/bin/python
LOG=/home/shahbaa/kv-compression/logs
OUT=/home/shahbaa/kv-compression/results/freekv/codec
M=ds-r1-qwen-7b; D=MATH50
Q=${QFILE:-/home/shahbaa/kv-compression/scratchpad/pareto_jobs.txt}
LK="$Q.lock"
SLOTS_PER_GPU=${SLOTS_PER_GPU:-2}
GPUS=${GPUS:-"0 1 2 3"}
# Never co-schedule onto a card another user is on, and leave room for a
# second job: a card already past this many MiB gets no new work from us.
MEM_CEIL=${MEM_CEIL:-34000}

pop () { flock 9; head -1 "$Q"; sed -i '1d' "$Q"; } 9>"$LK"

run_job () {
  local tag B s flags g
  IFS='|' read -r tag B s flags <<< "$1"; g=$2
  echo "=== START $tag/B$B/s$s gpu=$g $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES=$g $PY -u -m eval.reasoning.pred \
      --model "$M" --dataset "$D" --GQA_policy avgSM \
      --temperature 0.6 --top_p 0.95 --max_gen 16384 \
      --budget "$B" --sink 512 --recent 512 --page_size 32 \
      --seed "$s" --out_root_dir "$OUT" $flags \
      > "$LOG/par_${tag}_B${B}_s${s}.log" 2>&1
  echo "=== DONE  $tag/B$B/s$s exit=$? $(date -Iseconds)"
}

declare -A PIDS   # "gpu:slot" -> pid
while :; do
  launched=0
  for g in $GPUS; do
    for sl in $(seq 1 "$SLOTS_PER_GPU"); do
      key="$g:$sl"
      [ -n "${PIDS[$key]:-}" ] && kill -0 "${PIDS[$key]}" 2>/dev/null && continue
      PIDS[$key]=""
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
      [ "$u" -gt "$MEM_CEIL" ] && continue
      job=$(pop); [ -z "$job" ] && continue
      run_job "$job" "$g" & PIDS[$key]=$!
      launched=1
      sleep 25          # stagger model loads so two never allocate at once
    done
  done
  if [ ! -s "$Q" ]; then
    busy=0
    for k in "${!PIDS[@]}"; do
      [ -n "${PIDS[$k]}" ] && kill -0 "${PIDS[$k]}" 2>/dev/null && busy=1
    done
    [ "$busy" = 0 ] && break
  fi
  [ "$launched" = 0 ] && sleep 40
done
wait
echo "=== SWEEP COMPLETE $(date -Iseconds)"
