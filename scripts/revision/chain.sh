#!/usr/bin/env bash
# Runs the revision decode blocks back to back on all GPUs, in the order of the brief:
#   A3 Qwen-7B (already queued separately) -> A3 Llama-8B -> A1 -> A2 (MATH500 row) -> B.
# Each stage waits for the tables it needs, regenerates its job file, then runs queue.sh.
set -u
source "$HOME/oval-kv/scripts/revision/env.sh"; cd "$KV_ROOT"
L=logs/revision/chain.log; say () { echo "[$(date -Iseconds)] $*" >> $L; }
GPUS="${GPUS:-0 1 2 3}"
say "waiting for the A3 Qwen-7B queue"
until grep -q "queue empty" logs/revision/queue_A3_qwen7b.log; do sleep 300; done
say "waiting for Llama-8B r_v=16 tables"
until grep -q DONE logs/revision/calib_llama_vr16.log; do sleep 300; done
$KV_PY scripts/revision/a3_tables.py >> $L 2>&1
python3 scripts/revision/make_jobs.py A3 ds-r1-llama-8b > scripts/revision/jobs_A3_llama8b.txt
say "A3 Llama-8B: $(wc -l < scripts/revision/jobs_A3_llama8b.txt) cells"
GPUS="$GPUS" SLOTS=2 JOBS=scripts/revision/jobs_A3_llama8b.txt scripts/revision/queue.sh >> logs/revision/queue_A3_llama8b.log 2>&1
say "waiting for disjoint tables"
until grep -q DONE logs/revision/calib_disjoint.log; do sleep 300; done
for B in A1 A2math B; do
  python3 scripts/revision/make_jobs.py $B > scripts/revision/jobs_$B.txt
  say "$B: $(wc -l < scripts/revision/jobs_$B.txt) cells"
  GPUS="$GPUS" SLOTS=2 JOBS=scripts/revision/jobs_$B.txt scripts/revision/queue.sh >> logs/revision/queue_$B.log 2>&1
done
say "ALL DONE"
