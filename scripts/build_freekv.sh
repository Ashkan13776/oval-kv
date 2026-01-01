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

ENVDIR=${CONDA_ENV}
CUDA=${CONDA_ENV}          # provides nvcc 12.8
export CUDA_HOME=$CUDA
export PATH=$ENVDIR/bin:$CUDA/bin:$PATH
export TORCH_CUDA_ARCH_LIST="12.0"
cd $FREEKV_DIR/source
echo "=== nvcc: $(nvcc --version | tail -2 | head -1)"
echo "=== python: $($ENVDIR/bin/python -V)"
echo "=== torch: $($ENVDIR/bin/python -c 'import torch;print(torch.__version__, torch.version.cuda)')"
echo "=== building freekv_cpp ==="
$ENVDIR/bin/pip install -e . --no-build-isolation 2>&1 | tail -60
