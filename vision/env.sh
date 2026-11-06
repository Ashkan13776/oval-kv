# Source before any run: the ovalkv env (torch 2.8 / cu128, transformers 4.45.2) as if activated.
export PATH=$HOME/miniconda3/envs/ovalkv/bin:$PATH
export CUDA_HOME=$HOME/miniconda3/envs/ovalkv
export TORCH_CUDA_ARCH_LIST=8.6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OVAL_QUANT=0
export KVC_MAX_TOKENS=16384   # sequences are ~14K tokens; slab size only
