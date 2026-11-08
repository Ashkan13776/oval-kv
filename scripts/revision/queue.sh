#!/usr/bin/env bash
# Resumable GPU queue for the revision decodes.
#
# JOBS file: one job per line, "<job_id>|<harness args>" (args for eval.reasoning.pred,
# without CUDA_VISIBLE_DEVICES). A job is skipped when logs/revision/done/<job_id> exists;
# the marker is written only when the harness exits 0. Lines starting with # are ignored.
#
#   GPUS="0 2 3" SLOTS=2 MINFREE=20000 JOBS=scripts/revision/jobs_A3.txt scripts/revision/queue.sh
set -u
source "$HOME/oval-kv/scripts/revision/env.sh"
JOBS="$(readlink -f "${JOBS:?JOBS=<file>}")"
GPUS="${GPUS:-0 1 2 3}"; SLOTS="${SLOTS:-2}"; MINFREE="${MINFREE:-20000}"; SETTLE="${SETTLE:-90}"
GPN="${GPN:-1}"   # GPUs per job (2 for 14B)
DONE="$KV_ROOT/logs/revision/done"; CLOG="$KV_ROOT/logs/revision/cells"; mkdir -p "$DONE" "$CLOG"
CMDLOG="$KV_ROOT/logs/revision/commands.log"
cd "$KV_ROOT/third_party/FreeKV/accuracy"

run_job () {   # $1 id, $2 args, $3 gpu list
  local id="$1" args="$2" g="$3"
  echo "$(date -Iseconds) CUDA_VISIBLE_DEVICES=$g $KV_PY -u -m eval.reasoning.pred $args" >> "$CMDLOG"
  echo "=== START $id gpu=$g $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES=$g $KV_PY -u -m eval.reasoning.pred $args > "$CLOG/$id.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] && date -Iseconds > "$DONE/$id"
  echo "=== DONE  $id exit=$rc $(date -Iseconds)"
}

# slot list: with GPN=2, pair consecutive GPUs
SLOTLIST=""
if [ "$GPN" = 2 ]; then
  set -- $GPUS
  while [ $# -ge 2 ]; do for i in $(seq 1 "$SLOTS"); do SLOTLIST="$SLOTLIST $1,$2:$i"; done; shift 2; done
else
  for g in $GPUS; do for i in $(seq 1 "$SLOTS"); do SLOTLIST="$SLOTLIST $g:$i"; done; done
fi
echo "=== queue $JOBS on slots [$SLOTLIST] minfree ${MINFREE}MiB $(date -Iseconds)"

mapfile -t LINES < <(grep -v '^\s*#' "$JOBS" | grep -v '^\s*$')
declare -A BUSY
i=0
while :; do
  # advance past finished jobs
  while [ $i -lt ${#LINES[@]} ] && [ -f "$DONE/${LINES[$i]%%|*}" ]; do i=$((i+1)); done
  if [ $i -ge ${#LINES[@]} ]; then
    for s in $SLOTLIST; do [ -n "${BUSY[$s]:-}" ] && wait "${BUSY[$s]}" 2>/dev/null; done
    echo "=== queue empty $(date -Iseconds)"; break
  fi
  launched=0
  for slot in $SLOTLIST; do
    [ $i -ge ${#LINES[@]} ] && break
    [ -n "${BUSY[$slot]:-}" ] && kill -0 "${BUSY[$slot]}" 2>/dev/null && continue
    gl="${slot%%:*}"; ok=1
    for g in ${gl//,/ }; do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g")
      [ "$free" -lt "$MINFREE" ] && ok=0
    done
    [ $ok = 1 ] || continue
    line="${LINES[$i]}"; i=$((i+1))
    [ -f "$DONE/${line%%|*}" ] && continue
    run_job "${line%%|*}" "${line#*|}" "$gl" &
    BUSY[$slot]=$!; launched=1
    sleep "$SETTLE"
  done
  [ $launched = 0 ] && sleep 30
done
