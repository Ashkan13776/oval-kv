#!/bin/bash
# Output-aware page basis inside FreeKV's own harness -- Table 3 protocol.
#
# We do NOT re-run their baselines; their published Table 3 columns stand as
# the reference. We run only our page representation, swept over eta:
#
#   eta=0     rank-8 key PCA          (the LOCKS summary, not FreeKV's)
#   eta>0     output-aware mix        (the idea under test)
#   eta=1     output-only basis
#
# Everything else is FreeKV's, verbatim from accuracy/README.md:
#   --method spec_ret --GQA_policy avgSM --spec_ret_steps 2 --spec_ret_corr 0.9
#   --temperature 0.6 --top_p 0.95 --max_gen 16384
#   seeds 42..49  (k=8; the paper uses 8 samples because reasoning output is
#                  highly seed-sensitive)
#   budget 2048 (paper's B), sink 512, recent 512, page_size 32
#
# Datasets: the shipped files ARE the paper's subsets -- aime24 (30 problems,
# "the entire AIME24 dataset"), math50 (50), gpqa50c (50). Note the paper calls
# the latter two MATH500 and GPQA; --dataset MATH500 points at a file that does
# not exist in the artifact.
#
# NOTE on spec_ret: the CUDA perf path (source/pred.py) has a genuine race on
# this hardware -- same config twice gives different corruption. It is NOT used
# here: accuracy/ is pure PyTorch and never imports freekv_cpp. Verified by
# grepping the import graph; see notes/freekv-port.md.
#
# Resumable: pred.py skips a (model, dataset, eta, seed) cell whose output
# .jsonl already exists.
set -u
KV_ROOT="${KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KV_PY="${KV_PY:-python}"
FREEKV_DIR="${FREEKV_DIR:-$KV_ROOT/third_party/FreeKV}"

cd $FREEKV_DIR/accuracy || exit 1
export HF_HOME=${HF_HOME}
export PYTHONUNBUFFERED=1
# accuracy path stores bf16 records; this also fixes the _q0 tag in output paths
export OVAL_QUANT=0
PY=$KV_PY
OUT=$KV_ROOT/results/freekv/sweep

MODELS="${MODELS:-ds-r1-qwen-7b ds-r1-llama-8b ds-r1-qwen-14b}"
ETAS="${ETAS:-0.0 0.25 0.5 0.75}"
DATASETS="${DATASETS:-AIME24 MATH50 GPQA50c}"
SEEDS="${SEEDS:-42 43 44}"

for M in $MODELS; do
  for E in $ETAS; do
    for D in $DATASETS; do
      for S in $SEEDS; do
        echo "=== FKSWEEP $M eta=$E $D seed=$S  $(date -Iseconds)"
        $PY -u -m eval.reasoning.pred \
          --model "$M" --dataset "$D" \
          --method spec_ret --page_rep oval --eta "$E" \
          --GQA_policy avgSM --spec_ret_steps 2 --spec_ret_corr 0.9 \
          --temperature 0.6 --top_p 0.95 --max_gen 16384 \
          --budget 2048 --sink 512 --recent 512 --page_size 32 \
          --seed "$S" --out_root_dir "$OUT" \
          || echo "=== FKSWEEP $M eta=$E $D seed=$S FAILED"
      done
      echo "=== FKSWEEP CELL DONE $M eta=$E $D  $(date -Iseconds)"
    done
  done
  echo "=== FKSWEEP MODEL DONE $M  $(date -Iseconds)"
done
echo "=== FKSWEEP ALL DONE  $(date -Iseconds)"
