#!/usr/bin/env bash
# Keep the codec sweep running until every cell is complete.
#
# WHY THIS EXISTS, rather than chaining sweeps:
# codec_reasoning_sweep.sh uses ONE fixed queue file, truncates it at startup
# (`: > "$Q"`) and consumes it with `sed -i '1d'`. Two overlapping sweeps
# therefore destroy each other's queue. This supervisor holds an flock and is
# the only thing allowed to launch a sweep, so that can never happen.
#
# It also guards the failure mode we already hit once: when a cell dies on
# startup (a bad dataset path), the queue drains at ~25s per job and all 40
# cells "complete" in minutes having computed nothing. If a launch produces no
# new records, this stops rather than burning the queue.
set -u
KV_ROOT="${KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
M="${MODEL:-ds-r1-qwen-14b}"
DATASETS="${DATASETS:-MATH50 AIME24 GPQA50c}"
ETAS="${ETAS:-0.0 0.25 0.5 0.75 1.0}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49}"
RES="$KV_ROOT/results/codec/${M}-spec_ret"
HB="$KV_ROOT/logs/supervisor.log"
mkdir -p "$KV_ROOT/logs" "$RES"

exec 9>"$KV_ROOT/logs/.supervisor.lock"
flock -n 9 || { echo "another supervisor holds the lock; exiting"; exit 0; }

declare -A NPROB=( [MATH50]=50 [AIME24]=30 [GPQA50c]=50 )
say () { echo "[$(date -Iseconds)] $*" | tee -a "$HB"; }

# A cell counts as COMPLETE only when its .jsonl holds every problem. Counting
# files instead would call a cell done the moment it writes its first record.
complete_cells () {
  local n=0 d e f want
  for d in $DATASETS; do
    want=${NPROB[$d]}
    for e in $ETAS; do
      for f in "$RES"/${d}-*oval_r8_eta${e}_q0_recon1*.jsonl; do
        [ -e "$f" ] || continue
        [ "$(grep -c . "$f" 2>/dev/null || echo 0)" -ge "$want" ] && n=$((n+1))
      done
    done
  done
  echo "$n"
}
total_records () { cat "$RES"/*.jsonl 2>/dev/null | grep -c . || echo 0; }

NSEED=$(echo $SEEDS | wc -w); NETA=$(echo $ETAS | wc -w)
NDS=$(echo $DATASETS | wc -w); TARGET=$((NDS * NETA * NSEED))
say "supervisor up: target $TARGET complete cells for $M"

fails=0; last_rec=$(total_records); last_change=$(date +%s)
while :; do
  done_n=$(complete_cells)
  if [ "$done_n" -ge "$TARGET" ]; then
    say "ALL COMPLETE: $done_n/$TARGET cells"; exit 0
  fi

  if pgrep -f "[c]odec_reasoning_sweep.sh" >/dev/null 2>&1; then
    rec=$(total_records); now=$(date +%s)
    if [ "$rec" -ne "$last_rec" ]; then
      last_rec=$rec; last_change=$now; fails=0
    elif [ $((now - last_change)) -gt 2700 ]; then
      # 45 min with no new record anywhere. A single hard problem can take
      # ~5 min, so this is well clear of normal variance.
      u0=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0)
      u1=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1)
      say "STALL: no new records for 45min (gpu util ${u0}% / ${u1}%), cells $done_n/$TARGET"
      last_change=$now
    fi
    sleep 120; continue
  fi

  # No sweep running and work remains -> launch one.
  before=$(total_records)
  say "launching sweep ($done_n/$TARGET complete, $(( TARGET - done_n )) to go)"
  MODEL="$M" DATASETS="$DATASETS" ETAS="$ETAS" SEEDS="$SEEDS" \
    GPUS="${GPUS:-0 1}" KV_ROOT="$KV_ROOT" KV_PY="${KV_PY:-python}" \
    "$KV_ROOT/scripts/codec_reasoning_sweep.sh" \
    >> "$KV_ROOT/logs/sweep_run.log" 2>&1
  after=$(total_records)
  if [ "$after" -le "$before" ]; then
    fails=$((fails+1))
    say "sweep exited producing NO new records (strike $fails/3)"
    if [ "$fails" -ge 3 ]; then
      say "ABORT: three launches in a row computed nothing. Not burning the queue."
      say "       check \$KV_ROOT/logs/codec_${M}_*.log for the per-cell error."
      exit 1
    fi
    sleep 60
  else
    fails=0
  fi
done
