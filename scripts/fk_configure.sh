#!/bin/bash
set -eu
KV_ROOT="${KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KV_PY="${KV_PY:-python}"
FREEKV_DIR="${FREEKV_DIR:-$KV_ROOT/third_party/FreeKV}"

export ENVDIR=${CONDA_ENV}
export CUDA=$KV_ROOT/vendor/cuda128
export CUDA_HOME=$CUDA CUDAToolkit_ROOT=$CUDA
export PATH=$ENVDIR/bin:$CUDA/bin:$PATH
export TORCH_CUDA_ARCH_LIST="12.0"
rm -rf /tmp/fkbuild && mkdir -p /tmp/fkbuild && cd /tmp/fkbuild
cmake $FREEKV_DIR/source/freekv_cpp \
  -DCMAKE_PREFIX_PATH=$ENVDIR/lib/python3.10/site-packages/torch/share/cmake \
  -DBUILD_SHARED_LIBS=ON -GNinja \
  -DCUDAToolkit_ROOT=$CUDA \
  -DPython_EXECUTABLE=$ENVDIR/bin/python \
  -DPython3_EXECUTABLE=$ENVDIR/bin/python \
  -DPYTHON_EXECUTABLE=$ENVDIR/bin/python \
  -DPython_ROOT_DIR=$ENVDIR -DPython3_ROOT_DIR=$ENVDIR \
  -DPython_INCLUDE_DIR=$ENVDIR/include/python3.10 \
  -DPython3_INCLUDE_DIR=$ENVDIR/include/python3.10 2>&1 | tail -4
