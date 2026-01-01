#!/usr/bin/env bash
# Fetch the upstream trees and install OVAL into FreeKV.
#
# FreeKV is not vendored here: our change to it is ~500 lines, so we ship a
# patch plus the new modules and apply them to a pinned upstream checkout.
set -euo pipefail
KV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP="$KV_ROOT/third_party"; mkdir -p "$TP"

FREEKV_REPO=https://github.com/sjtu-zhao-lab/FreeKV.git
FREEKV_PIN=2c8a7d25c9f3c7c15ce15b2f84cd03f477bd7469
IB_REPO=https://github.com/OpenBMB/InfiniteBench.git

if [ ! -d "$TP/FreeKV/.git" ]; then
  echo "== cloning FreeKV @ $FREEKV_PIN"
  git clone --recurse-submodules "$FREEKV_REPO" "$TP/FreeKV"
fi
git -C "$TP/FreeKV" checkout -q "$FREEKV_PIN"
git -C "$TP/FreeKV" submodule update --init --recursive

echo "== applying patch"
git -C "$TP/FreeKV" apply --3way "$KV_ROOT/patches/freekv.patch"

echo "== installing OVAL modules"
install -Dm644 "$KV_ROOT/oval/page_basis.py"   "$TP/FreeKV/accuracy/kvc/patch/locks_rep.py"
install -Dm644 "$KV_ROOT/oval/page_summary.py" "$TP/FreeKV/source/freekv/locks_summary.py"
install -Dm644 "$KV_ROOT/oval/raas.py"         "$TP/FreeKV/source/freekv/raas.py"
install -Dm644 "$KV_ROOT/oval/locks_estimate.cu" \
               "$TP/FreeKV/source/freekv_cpp/src/locks_estimate.cu"
install -Dm644 "$KV_ROOT/oval/infinitebench_pred.py" \
               "$TP/FreeKV/accuracy/eval/InfiniteBench/pred.py"

if [ ! -d "$TP/InfiniteBench/.git" ]; then
  echo "== cloning InfiniteBench (official scorers)"
  git clone --depth 1 "$IB_REPO" "$TP/InfiniteBench"
fi

cat <<MSG

Done. Next:
  export KV_ROOT="$KV_ROOT"
  export KV_PY=\$(which python)          # a torch>=2.8 / CUDA 12.8 env
  ./scripts/build_freekv.sh              # builds FreeKV's CUDA extension
  ./scripts/ib_freekv.sh                 # InfiniteBench: FreeKV vs OVAL
  python analysis/all_tables.py          # score everything found on disk
MSG
