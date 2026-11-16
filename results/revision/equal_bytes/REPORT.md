# Equal-bytes baselines vs OVAL (MATH50, R1-Distill Qwen-7B, k=8)

Status 2026-11-11 18:35 EST: one arm (int8 KV + FreeKV selection) running; two arms not runnable
without new code. No accuracy numbers yet -- report only completed cells (k=8, n=50 each).

Tree: `~/oval-kv-main` = git worktree of `origin/main` @ 501d2d3, FreeKV pinned 2c8a7d2 + `patches/freekv.patch`
(`setup.sh`). Env `~/miniconda3/envs/ovalkv` (torch 2.8.0+cu128), PATH/CUDA_HOME set to that env (flashinfer JIT needs ninja).
`OVAL_QUANT=0 KVC_MAX_TOKENS=32768 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Arm 2 -- 8-bit KV + FreeKV retrieval (launched)

```
python -u -m eval.reasoning.pred --model ds-r1-qwen-7b --dataset MATH50 --seed S \
  --budget 2048 --sink 512 --recent 512 --page_size 32 --GQA_policy avgSM \
  --spec_ret_steps 2 --spec_ret_corr 0.9 --temperature 0.6 --top_p 0.95 --max_gen 16384 \
  --method spec_ret --page_rep quest --kv8 --out_root_dir results/revision/equal_bytes/runs_kv8
```
Same flags as the Pareto FreeKV arm (`spec_ret`, `page_rep quest`, avgSM, pQ2, corr 0.9, skl1) plus `--kv8`.
Driver: `run_kv8.sh` (nohup; waits for GPUs with no compute process of any user; smoke first; resumable,
`done/s<seed>` on exit 0). Every launch is in `commands.log`.
NOTE: `--kv8` is NOT in the output filename, so it MUST keep its own `--out_root_dir`.

What `--kv8` does (kvc/patch/llama.py, utils.asym_quant_int8):
- keys AND values of EVERY token (sink, paged, recent) stored int8, asymmetric, per token per KV head over
  head_dim (min/max -> scale, zero point), scale + zero in the model dtype (2 B each);
  the whole cache is dequantized each step before attention; the newest token is attended exact once.
- Quest min/max digests: written from exact keys at prefill, from the (dequantized) cache keys in decode.
- Side effect: `--kv8` loads the model in float16 instead of bfloat16 (eval/util.py). Weights/activation
  dtype therefore also differs from the bf16 FreeKV and OVAL arms -- a confound to state.

Smoke (problem 0, seed 42, GPU 0): exit 0, boxed answer parsed (`(3, \dfrac{\pi}{2})` vs gold
`\left( 3, \frac{\pi}{2} \right)`), 1236 tokens, 80.3 s incl. ~20 s flashinfer JIT.
Measured speed of the full cells (first 6-15 problems): 107-209 s/problem (mean ~165), ~2.3-2.9 h per cell.

## Memory model (supplement: one KV head/layer, L=3745, d=128, 256 B/token/tensor in 16-bit)

- FreeKV: 2*3745*256 + 101*512 B = 1923.0 KiB. OVAL (bf16 records): 1236.5 KiB.
- kv8 as run: 3745 tokens * 2 tensors * (128 + 2 + 2) B + 101 digests * 512 B
  = 988,680 + 51,712 B = **1016.0 KiB** (82% of OVAL, 53% of FreeKV).
  (If sink/recent were kept exact like OVAL -- not what the harness does --
  1025*512 + 2720*264 + 101*512 B = 1264.25 KiB.)

## Arm 1 -- OVAL with quantized records (OVAL_QUANT=1): NOT runnable

In the accuracy harness `OVAL_QUANT` only changes the output tag (`_q1`, eval/util.py:182); the records in
`kvc/patch/oval.py` are explicitly "UNQUANTIZED (model dtype)", and dynamic_attention reconstructs from them.
The int4/int8 packing exists only in the performance engine (`oval/page_summary.py::quantize/dequantize`):
basis int4 per-column scale, coef int8 per-token scale, mean int8 per-page scale, bf16 scales inline;
key record r=8: r(d/2+2)+m(r+2)+(d+2) = 978 B/page (vs 2816 B bf16). A run with OVAL_QUANT=1 would be
the bf16 OVAL computation under a different filename, so it was not launched.
Needed: apply quantize->dequantize to the key AND value records in kvc/patch/oval.py at page build
(value record rv=16 in the same format would be 1762 B/page). Projected memory if done the same way:
512.5 KiB exact + 85*(978+1762) B + 16*978 B = 755.2 KiB (not a run; formula only).

## Arm 3 -- shared low-rank K+V with reconstruction: NOT available

`--page_rep loki` is a shared per-(layer, head) PCA basis (rank hard-coded 16, from the sequence's own
keys) used for SCORING only; attention reads exact keys and values. No shared-basis reconstruction mode
exists for keys or values (`--oval_vrank/--vbasis pca` is page-local). Needed: a shared K and V basis
(calibrated offline or from prefill), per-token coefficients stored instead of paged K/V, reconstruction of
selected pages in dynamic_attention (as the oval_recon branch does), and a rank flag.
Equal bytes: 512.5 KiB + 2720*4r + 2*(128r+128)*2 B <= 1236.5 KiB  ->  r <= 65; r=64 gives 1225.0 KiB.
