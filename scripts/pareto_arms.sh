#!/usr/bin/env bash
# New baselines at the EXISTING config -- B=2048, sink 512, recent 512, 8 seeds,
# the same settings the seven arms already on disk were run under, which are the
# paper's. The Pareto front is then across METHODS at matched budget: they all
# attend the same number of tokens, and differ in the per-page digest they must
# store to decide which, and in what that decision costs at decode.
#
# No budget sweep here. Varying the budget would make these numbers
# incomparable with the ones already reported; that sweep is kept separately in
# scratchpad/budget_sweep_deferred.txt as a supplementary curve if wanted.
#
# THREE FAMILIES, kept distinct so the comparison stays fair:
#  A. scoring rules, --method quest: score with the CURRENT query, no
#     speculative retrieval -- how Loki/SparQ/InfLLM/H2O/SnapKV actually work,
#     and it isolates the scoring rule from the retrieval machinery.
#  B. published systems, --method spec_ret: FreeKV and ArkVale ARE speculative
#     retrieval; ours is run the same way, matching the existing numbers.
#  C. non-retrieval references: the two window controls.
set -u
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49}"
B="${B:-2048}"
SR="--spec_ret_steps 2 --spec_ret_corr 0.9"
A_ARMS="loki sparq infllm knorm centroid expatt h2o tova snapkv random q4key shadowkv rocketkv"

for S in $SEEDS; do
  # --- A: scoring rules at matched machinery and matched budget ------------
  for a in $A_ARMS; do
    echo "$a|$B|$S|--method quest --page_rep $a"
  done
  echo "ours_q|$B|$S|--method quest --page_rep oval --eta 0.0 --oab_rank 8"
  # --- B: published systems ------------------------------------------------
  echo "pyramid|$B|$S|--method spec_ret --page_rep quest --pyramid 0.5 $SR"
  # --- C: controls ----------------------------------------------------------
  # StreamingLLM: no retrieval at all, sink+recent only. The floor.
  echo "stream|0|$S|--method quest --page_rep quest"
  # Sliding window: surrender the budget, spend it on a wider recent window,
  # so it attends the SAME token count as every other arm. Answers "is
  # retrieval better than just a bigger window?", which random cannot.
  echo "sliding|0|$S|--method quest --page_rep quest --recent $((512+B))"
done

# Ceilings last: they do a dense pass over every key at every decode step, so
# they cost ~2-3x a normal cell. Not competitors -- bounds.
for S in $SEEDS; do
  echo "oracle|$B|$S|--method quest --page_rep oracle"
  echo "oracle_prev|$B|$S|--method spec_ret --page_rep oracle $SR"
done
