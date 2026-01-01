#!/bin/bash
# Wait for GPU 0 to have room, then run the LongBench-v2 latency MVP.
#
# Qwen-2.5-14B needs ~28 GB of weights + ~2 GB pool + ~2 GB prefill activations.
# With another user holding 62 GB the run OOMs during prefill (788 MiB short).
# Poll rather than squat: we never allocate until the space is actually there,
# so nobody else is crowded out while we wait.
set -u
KV_ROOT="${KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KV_PY="${KV_PY:-python}"
FREEKV_DIR="${FREEKV_DIR:-$KV_ROOT/third_party/FreeKV}"

NEED_MIB=${NEED_MIB:-36000}
for _ in $(seq 1 480); do           # 480 x 90s = 12h
  free=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "$free" -ge "$NEED_MIB" ]; then
    echo "=== GPU0 has ${free}MiB free (need ${NEED_MIB}) -- starting $(date -Iseconds)"
    exec bash $KV_ROOT/env/lb2_latency_mvp.sh
  fi
  sleep 90
done
echo "=== gave up waiting after 12h $(date -Iseconds)"
