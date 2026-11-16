#!/usr/bin/env bash
# Iso-memory (128 frames), Qwen2.5-VL-7B, Video-MME Long (900 q): launch everything in the background.
# Resumable: rerun this script; finished questions are skipped (run_qwen25vl.py), finished embeddings kept.
#   1. bash scripts/revision/C/iso128_fetch.sh   (once: 128-frame PNG caches in tmpfs)
#   2. bash scripts/revision/C/iso128.sh         (this: vision-side workers + harness shards)
# Arms / flags: gen_jobs_c1.py iso128 (= full_n64 / oval_n64 / vz50_full with --frames 128).
# Placement (4x A6000, measured peaks: prep ~10 GB with the LM parked on the CPU, OVAL 19.9, FullKV 19.0,
# VisionZip 16.6 GB): prep worker + OVAL shard on GPUs 0, 1, 3; FullKV + VisionZip on GPU 2.
set -u
cd "$(dirname "$0")/../../.." && source scripts/revision/C/env.sh
export QWEN_CACHE_ROOT=/dev/shm/oval_iso128 KVC_MAX_TOKENS=32768   # prefill ~25.9K tokens (env.sh: 16384)
L=logs/revc/iso128; mkdir -p $L
OUT=results/revision/C/runs/iso128_qwen25vl
PREP_GPUS=(0 1 3)
ARMSPEC="full_n128:0:1 oval_n128:0:3 vz50_full_n128:0.5:1"
for k in 0 1 2; do
  CUDA_VISIBLE_DEVICES=${PREP_GPUS[$k]} nohup ~/qwenvl/bin/python scripts/revision/C/iso128_prep.py --out $OUT \
    --arms $ARMSPEC --worker $k --nworkers 3 >> $L/prep_w$k.log 2>&1 < /dev/null &
  echo "prep w$k GPU${PREP_GPUS[$k]} pid $!"
done
run () {  # gpu arm nshards shard
  J=$(python scripts/revision/C/gen_jobs_c1.py iso128 --arms $2 --nshards $3 | sed -n "$(($4 + 1))p")
  CUDA_VISIBLE_DEVICES=$1 nohup bash -c "$J" < /dev/null > /dev/null 2>&1 &
  echo "$2 s$4/$3 GPU$1 pid $!"
}
run 0 oval_n128 3 0
run 1 oval_n128 3 1
run 3 oval_n128 3 2
run 2 full_n128 1 0
run 2 vz50_full_n128 1 0
