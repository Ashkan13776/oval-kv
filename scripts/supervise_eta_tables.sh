#!/usr/bin/env bash
# Keep the eta-table sweep running until every cell is complete.
#
# Sibling of supervise_sweep.sh with the same guards -- an flock so only one
# instance ever launches a sweep, the "launch produced no new records" strike
# counter, the 45-minute stall restart that kills only OUR harness processes
# (pattern anchored on the interpreter path and --eta_table) -- but its own
# lock, queue, logs and record glob (*_etab*), so it never interferes with the
# codec supervisor. Completeness is decided by analysis/eta_rules_queue.py,
# the same code that builds the queue.
set -u
KV_ROOT="${KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KV_PY="${KV_PY:-python}"
MODELS="${MODELS:-ds-r1-qwen-7b ds-r1-llama-8b}"
DATASETS="${DATASETS:-MATH50 AIME24 GPQA50c}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49}"
RULES="${RULES:-r1 r2 r3 r4 r5 r6}"
TABLES="${TABLES:-$KV_ROOT/results/eta_rules/tables}"
GPUS="${GPUS:-0 1 2 3}"
LOG="$KV_ROOT/logs/eta_rules"; HB="$LOG/supervisor.log"
mkdir -p "$LOG"

exec 9>"$LOG/.supervisor_eta.lock"
flock -n 9 || { echo "another eta-table supervisor holds the lock; exiting"; exit 0; }

say () { echo "[$(date -Iseconds)] $*" | tee -a "$HB"; }
todo () { $KV_PY "$KV_ROOT/analysis/eta_rules_queue.py" --models $MODELS --datasets $DATASETS \
            --seeds $SEEDS --rules $RULES --tables_dir "$TABLES" --root "$KV_ROOT/results/codec" --count; }
done_n () { $KV_PY "$KV_ROOT/analysis/eta_rules_queue.py" --models $MODELS --datasets $DATASETS \
            --seeds $SEEDS --rules $RULES --tables_dir "$TABLES" --root "$KV_ROOT/results/codec" --complete; }
total_records () { cat "$KV_ROOT"/results/codec/*-spec_ret/*_etab*.jsonl 2>/dev/null | grep -c . ; return 0; }
PRED_PAT="^${KV_PY} -u -m eval.reasoning.pred .*--eta_table"

say "eta-table supervisor up: $(todo) seed-cells to run, $(done_n) complete"
fails=0; last_rec=$(total_records); last_change=$(date +%s)
while :; do
  left=$(todo)
  if [ "${left:-1}" -eq 0 ]; then
    say "ALL COMPLETE: $(done_n) seed-cells"; exit 0
  fi
  if pgrep -f "[e]ta_table_sweep.sh" >/dev/null 2>&1; then
    sleep 120; continue
  fi
  before=$(total_records)
  say "launching eta-table sweep ($left seed-cells to go, $(done_n) complete)"
  MODELS="$MODELS" DATASETS="$DATASETS" SEEDS="$SEEDS" RULES="$RULES" TABLES="$TABLES" \
    GPUS="$GPUS" SLOTS="${SLOTS:-2}" MINFREE="${MINFREE:-20000}" SETTLE="${SETTLE:-90}" \
    KV_ROOT="$KV_ROOT" KV_PY="$KV_PY" \
    "$KV_ROOT/scripts/eta_table_sweep.sh" >> "$LOG/sweep_run.log" 2>&1 &
  SWEEP_PID=$!
  waiting_since=0
  while kill -0 "$SWEEP_PID" 2>/dev/null; do
    rec=$(total_records); now=$(date +%s)
    if [ "$rec" -ne "$last_rec" ]; then
      last_rec=$rec; last_change=$now; waiting_since=0
    else
      ours=$(pgrep -cf "$PRED_PAT" || true)
      if [ "${ours:-0}" -eq 0 ]; then
        busy=""
        for g in $GPUS; do
          u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo 0)
          [ "${u:-0}" -gt 2000 ] && busy="$busy $g(${u}MiB)"
        done
        if [ -n "$busy" ]; then
          if [ "$waiting_since" -eq 0 ]; then
            waiting_since=$now
            say "WAITING: GPUs busy with another user -->$busy. Holding off; will start automatically when free."
          elif [ $(( (now - waiting_since) % 3600 )) -lt 130 ]; then
            say "still waiting for$busy ($(( (now - waiting_since) / 60 ))min so far)"
          fi
          last_change=$now
        fi
      elif [ $((now - last_change)) -gt 2700 ]; then
        say "STALL: our cells are running but no new record for 45min -- restarting them"
        kill "$SWEEP_PID" 2>/dev/null
        pkill -P "$SWEEP_PID" 2>/dev/null
        pkill -f "$PRED_PAT" 2>/dev/null
        sleep 30
        pkill -9 -f "$PRED_PAT" 2>/dev/null
        last_change=$now
        break
      fi
    fi
    sleep 120
  done
  wait "$SWEEP_PID" 2>/dev/null || true
  after=$(total_records)
  if [ "$after" -le "$before" ]; then
    fails=$((fails+1))
    say "sweep exited producing NO new records (strike $fails/3)"
    if [ "$fails" -ge 3 ]; then
      say "ABORT: three launches in a row computed nothing. Not burning the queue."
      say "       check $LOG/cell_*.log for the per-cell error."
      exit 1
    fi
    sleep 60
  else
    fails=0
  fi
done
