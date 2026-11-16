#!/usr/bin/env bash
# Equal-bytes arm: FreeKV selection (spec_ret + quest digest) on an int8 KV cache (--kv8).
# Waits for a GPU with NO compute process from any user, smoke-tests 1 problem x seed 42,
# then runs seeds 42-49 on MATH50 (one cell per free GPU). Resumable: pred.py resumes per
# problem; done/<seed> is written only on exit 0.
set -u
R=$HOME/oval-kv-main/results/revision/equal_bytes
PY=$HOME/miniconda3/envs/ovalkv/bin/python
cd $HOME/oval-kv-main/third_party/FreeKV/accuracy
export PATH=$HOME/miniconda3/envs/ovalkv/bin:$PATH CUDA_HOME=$HOME/miniconda3/envs/ovalkv TORCH_CUDA_ARCH_LIST=8.6
export OVAL_QUANT=0 KVC_MAX_TOKENS=32768 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
mkdir -p $R/logs $R/done $R/claim
COMMON="--model ds-r1-qwen-7b --dataset MATH50 --budget 2048 --sink 512 --recent 512 --page_size 32 --GQA_policy avgSM --spec_ret_steps 2 --spec_ret_corr 0.9 --temperature 0.6 --top_p 0.95 --max_gen 16384 --method spec_ret --page_rep quest --kv8 --out_root_dir $R/runs_kv8"
GPUS="${GPUS:-0 1 2 3}"
free_gpu () {  # first GPU with no compute process (any user) and not claimed by us
  local busy g u
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  for g in $GPUS; do
    [ -f $R/claim/gpu$g ] && kill -0 "$(cat $R/claim/gpu$g)" 2>/dev/null && continue
    u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $g)
    echo "$busy" | grep -q "$u" && continue
    echo $g; return 0
  done; return 1
}
run () {  # $1 gpu $2 tag $3 extra args
  echo $BASHPID > $R/claim/gpu$1
  local t0=$(date +%s)
  echo "$(date -Iseconds) gpu=$1 CUDA_VISIBLE_DEVICES=$1 $PY -u -m eval.reasoning.pred $COMMON $3" >> $R/commands.log
  CUDA_VISIBLE_DEVICES=$1 $PY -u -m eval.reasoning.pred $COMMON $3 > $R/logs/$2.log 2>&1
  local rc=$?
  echo "$(date -Iseconds) $2 exit=$rc wall=$(( $(date +%s)-t0 ))s" >> $R/commands.log
  rm -f $R/claim/gpu$1; return $rc
}
# ---- smoke: 1 problem x seed 42 ----
if [ ! -f $R/done/smoke ]; then
  until g=$(free_gpu); do sleep 60; done
  run $g smoke "--seed 42 --data_idx 0" || { echo "SMOKE FAILED"; exit 1; }
  f=$(ls $R/res-test/ds-r1-qwen-7b-spec_ret/*.jsonl | head -1)
  $PY - "$f" <<'PY' >> $R/smoke.txt || { echo "SMOKE PARSE FAILED"; exit 1; }
import json,re,sys
e=[json.loads(l) for l in open(sys.argv[1]) if l.strip()][-1]
m=re.search(r"\\boxed{((?:[^{}]|\{[^{}]*\})*)}",e["pred"])
assert m, "no boxed answer"
print(f"smoke qid={e['qid']} output_len={e['output_len']} boxed={m.group(1)!r} answer={e['answer']!r}")
PY
  date -Iseconds > $R/done/smoke
fi
# ---- full: seeds 42-49, one cell per free GPU ----
for S in 42 43 44 45 46 47 48 49; do
  [ -f $R/done/s$S ] && continue
  until g=$(free_gpu); do sleep 60; done
  ( run $g s$S "--seed $S" && date -Iseconds > $R/done/s$S ) &
  sleep 120
done
wait
echo "ALL DONE $(date -Iseconds)"
