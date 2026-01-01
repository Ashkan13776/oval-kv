#!/bin/bash
KV_ROOT="${KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KV_PY="${KV_PY:-python}"
FREEKV_DIR="${FREEKV_DIR:-$KV_ROOT/third_party/FreeKV}"

export ENVDIR=${CONDA_ENV}
export CUDA=$KV_ROOT/vendor/cuda128
export CUDA_HOME=$CUDA CUDAToolkit_ROOT=$CUDA
export PATH=$ENVDIR/bin:$CUDA/bin:$PATH
export TORCH_CUDA_ARCH_LIST="12.0"
cd /tmp/fkbuild && cmake --build . --target freekv_cpp -j 16
