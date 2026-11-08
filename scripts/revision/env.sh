# Environment for every revision run (same stack as the Tab. 1 Qwen-7B runs).
export KV_ROOT=$HOME/oval-kv
export PATH=$HOME/miniconda3/envs/freekv/bin:$PATH
export CUDA_HOME=$HOME/miniconda3/envs/freekv
export TORCH_CUDA_ARCH_LIST=8.6
export KV_PY=$HOME/miniconda3/envs/freekv/bin/python
export OVAL_QUANT=0 KVC_MAX_TOKENS=32768
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
