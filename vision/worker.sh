#!/usr/bin/env bash
# One worker per GPU: pop the first line of vision/queue/jobs.txt (under a lock) and run it
# with CUDA_VISIBLE_DEVICES=$1; idle-poll when the queue is empty. Log: vision/logs/worker$1.log
cd "$(dirname "$0")/.." && source vision/env.sh
Q=vision/queue/jobs.txt; touch $Q
while true; do
  job=$(flock $Q.lock -c "head -1 $Q; sed -i 1d $Q")
  if [ -z "$job" ]; then sleep 30; continue; fi
  echo "[$(date '+%F %T')] GPU$1 START $job" >> vision/queue/done.txt
  CUDA_VISIBLE_DEVICES=$1 bash -c "$job"; rc=$?
  echo "[$(date '+%F %T')] GPU$1 END rc=$rc $job" >> vision/queue/done.txt
done
