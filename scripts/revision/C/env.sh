# Block C env: vision/env.sh (ovalkv: torch 2.8.0+cu128, transformers 4.45.2) plus the
# system gcc for flashinfer's JIT (the base conda's cross gcc fails on ::timespec_get).
source "$(dirname "${BASH_SOURCE[0]}")/../../../vision/env.sh"
export CC=/usr/bin/gcc CXX=/usr/bin/g++
unset CUDAHOSTCXX
export VMME_ROOT="${VMME_ROOT:-$HOME/vmme}"
