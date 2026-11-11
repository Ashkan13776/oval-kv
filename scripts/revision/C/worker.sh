#!/usr/bin/env bash
# Block C queue worker, one per GPU (as vision/worker.sh): pop the first line of
# logs/revc/queue/jobs.txt under a lock, run it on GPU $1, log START/END to done.txt.
cd "$(dirname "$0")/../../.." && source scripts/revision/C/env.sh
Q=logs/revc/queue/jobs.txt; mkdir -p logs/revc/queue logs/revc/jobs; touch $Q
while true; do
  job=$(flock $Q.lock -c "head -1 $Q; sed -i 1d $Q")
  if [ -z "$job" ]; then sleep 30; continue; fi
  echo "[$(date '+%F %T')] GPU$1 START $job" >> logs/revc/queue/done.txt
  CUDA_VISIBLE_DEVICES=$1 bash -c "$job"; rc=$?
  echo "[$(date '+%F %T')] GPU$1 END rc=$rc $job" >> logs/revc/queue/done.txt
done
