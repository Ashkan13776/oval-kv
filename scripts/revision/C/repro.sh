#!/usr/bin/env bash
# Reproduction gate: the 60 questions of the first 20 Long videos (601-620) with the flags of
# vision/runs/row1_full_n64 and vision/runs/row3_ovalfull_n64 (their args.json), then compare.
set -u
cd "$(dirname "$0")/../../.."
source scripts/revision/C/env.sh
Q=$(cat logs/revc/repro_qids.txt)
C="--protocol stream --frames 64 --durations long --budget 1792 --sink 128 --recent 128 --page_size 32 \
   --GQA_policy avgSM --skip_layer 1 --sparsity 1 --last_layer_budget 0 --corr_group avg --batch_size 1"
O=results/revision/C/runs
CUDA_VISIBLE_DEVICES=${G1:-0} python vision/run_videomme.py $C --method full \
    --qids $Q --out $O/repro_row1_full_n64 > logs/revc/repro_row1.log 2>&1 &
CUDA_VISIBLE_DEVICES=${G2:-1} python vision/run_videomme.py $C --method spec_ret --page_rep oval --oval_rank 8 \
    --oval_recon --oval_vrank 16 --vbasis pca --eta_table results/vision/eta/tables/llava-video-7b_VCAL_r1.json \
    --spec_ret_steps 2 --spec_ret_corr 0.8 --qids $Q --out $O/repro_row3_ovalfull_n64 > logs/revc/repro_row3.log 2>&1 &
wait
python vision/compare.py vision/runs/row1_full_n64 $O/repro_row1_full_n64
python vision/compare.py vision/runs/row3_ovalfull_n64 $O/repro_row3_ovalfull_n64
