# Environment for every revision run (same stack as the Tab. 1 Qwen-7B runs).
# FREEKV_ENV: the conda env with the FreeKV harness (torch 2.5.1+cu121, transformers 4.45.2).
export KV_ROOT="${KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FREEKV_ENV="${FREEKV_ENV:-$HOME/miniconda3/envs/freekv}"
export PATH=$FREEKV_ENV/bin:$PATH
export CUDA_HOME=$FREEKV_ENV
export TORCH_CUDA_ARCH_LIST=8.6
export KV_PY=$FREEKV_ENV/bin/python
export OVAL_QUANT=0 KVC_MAX_TOKENS=32768
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
