#!/usr/bin/env bash
# Value-codec rank sweep (results/codec/value_rank_qwen7b.md, docs/value-codec.md).
#   keys:   rank 8 for scoring and reconstruction (--oval_recon), eta fixed per
#           dataset by ETAMAP -- identical to the keys-only codec protocol
#   values: plain PCA (--vbasis pca) at each rank in VRANKS; no value eta
#   k = 8 seeds per cell. Rank 64 is not run: at page size 32 a centred page
#   has rank <= 31, so 32 and 64 both clamp to 31 (lossless) and are identical.
# Ranks run one after another, each under scripts/supervise_sweep.sh
# (per-seed resume, stall restart). Run detached:
#   setsid nohup scripts/value_rank_sweep.sh > logs/value_rank_sweep.out 2>&1 < /dev/null &
set -u
KV_ROOT="${KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KV_PY="${KV_PY:-python}"
export KV_ROOT KV_PY
export MODEL="${MODEL:-ds-r1-qwen-7b}"
export ETAMAP="${ETAMAP:-MATH50:0.25 AIME24:1.0 GPQA50c:0.25}" VBASIS=pca
export SEEDS="${SEEDS:-42 43 44 45 46 47 48 49}"
export GPUS="${GPUS:-0 1 2 3}" SLOTS="${SLOTS:-2}" MINFREE="${MINFREE:-20000}" SETTLE="${SETTLE:-90}"
for V in ${VRANKS:-16 8 4 1 32}; do
  VRANK=$V "$KV_ROOT/scripts/supervise_sweep.sh" || exit 1
done
