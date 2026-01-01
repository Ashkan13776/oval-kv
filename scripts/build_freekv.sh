#!/bin/bash
# Attempt to build FreeKV's CUDA extension on Blackwell (sm_120).
#
# The pinned submodules (cutlass 2024-04, flashinfer 2024-06, raft 2024-04) all
# predate Blackwell, and their requirements.txt pins torch 2.5.1 which has no
# sm_120 kernels at all. Two things make this plausible anyway:
#   - FreeKV's own 12 .cu files contain no wgmma/TMA/mbarrier/sm_90 code
#   - their CMakeLists uses CMAKE_CUDA_ARCHITECTURES=native, so it targets
#     sm_120 only and never instantiates Hopper-specific templates
# We build against CUDA 12.8 (the first toolkit with sm_120, and still new
# enough to compile 2024-era CUTLASS) rather than the system CUDA 13.0.
set -eu
KV_ROOT="${KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KV_PY="${KV_PY:-python}"
FREEKV_DIR="${FREEKV_DIR:-$KV_ROOT/third_party/FreeKV}"

# The torch env and the CUDA toolkit are TWO different things and are not
# always the same prefix: the torch env here has no nvcc, and nvcc 12.8 lives
# in a separate conda env. Resolve them independently.
#   KV_PY    -> python to build against   (its prefix becomes ENVDIR)
#   CUDA_DIR -> prefix containing bin/nvcc (>= 12.8, first toolkit with sm_120)
ENVDIR="${ENVDIR:-$(dirname "$(dirname "$(readlink -f "$KV_PY")")")}"
if [ -z "${CUDA_DIR:-}" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_DIR="$(dirname "$(dirname "$(command -v nvcc)")")"
  elif [ -x "$ENVDIR/bin/nvcc" ]; then
    CUDA_DIR="$ENVDIR"
  else
    echo "no nvcc found. Set CUDA_DIR to a CUDA >= 12.8 prefix (needs bin/nvcc)." >&2
    exit 1
  fi
fi
CUDA="$CUDA_DIR"
export CUDA_HOME=$CUDA
export PATH=$ENVDIR/bin:$CUDA/bin:$PATH
export TORCH_CUDA_ARCH_LIST="12.0"
cd $FREEKV_DIR/source
echo "=== nvcc: $(nvcc --version | tail -2 | head -1)"
echo "=== python: $($ENVDIR/bin/python -V)"
echo "=== torch: $($ENVDIR/bin/python -c 'import torch;print(torch.__version__, torch.version.cuda)')"
echo "=== building freekv_cpp ==="
$ENVDIR/bin/pip install -e . --no-build-isolation 2>&1 | tail -60
