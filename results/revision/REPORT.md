# OVAL revision experiments — report

Machine: 4x RTX A6000 (48 GB). Stack for all runs so far: torch 2.5.1+cu121, transformers 4.45.2 (`freekv` conda env), FreeKV @ 2c8a7d25 + `patches/freekv.patch` installed by `./setup.sh` into `third_party/FreeKV`. Unit tests (`analysis/test_page_basis.py`, `analysis/test_eta_rules.py`) pass.

## Discrepancies

1. **Tab. 5 KV-memory column vs. `vision/memory.py`** (C4). Running the repo's byte model on the Tab. 5 run directories (`~/oval-vision/vision/runs/row*_n64`) gives, model-level including the dense layer 0:
   - FullKV 100.0%
   - OVAL 53.0% (per layer 51.3%)
   - FreeKV 103.0%
   - Quest 103.0%
   - 4-bit keys 67.6%
   - SnapKV 17.6%
   - OVAL keys-only 69.1%
   
   Tab. 5 prints 74.3 / 149.3 / 149.3 / 70.2 / 21.1. The ratios are not constant (1.40, 1.45, 1.04, 1.20), so the column was not produced by this byte model with any single scaling. Nothing in the repo reproduces it. Not yet resolved; the author's source for the column is needed.

## Open questions
1. **GPQA Diamond is gated** (`Idavidrein/gpqa`, HTTP 401), and there is no Hugging Face token on this machine.
   - Blocked: A1 for GPQA, the GPQA row of A2, and the pooled table.
   - Conservative choice: A1 and A2 run MATH500 and AIME24 only until a token with GPQA access is provided.
2. **Llama-8B r_v=16 tables behind Tab. 1 are not on this machine**, and neither are the calibration captures for any model.
   - Llama-8B is recalibrated here with the Tab. 1 protocol (`scripts/revision/a_calib_llama_vr16.sh`, seed 7). Whether the result equals the table behind Tab. 1 cannot be checked without that table's hash.
   - For Qwen-7B and Qwen-14B, the rebuilt rule-1 tables reproduce the deployed hashes exactly (A3 below).
3. **Block C is parked.** It needs disk (about 34 GB free; Qwen2.5-VL-7B alone is about 16 GB), access to LongVideoBench (gated, hundreds of GB), and an engineering decision on InfiniPot-V, ReKV and VisionZip. Only the GPU-free C4 check has been done.
4. **Block B, retrieval-only arm.** It uses the same per-head AIME24 table as OVAL, so the arm differs from OVAL only in compression. No retrieval-only per-head run existed before.

## Timing (smoke test)
Qwen-7B, AIME24 problem 0, seed 42, OVAL Tab. 1 configuration, one A6000: 4,418 generated tokens in 179 s, i.e. 40.5 ms per token, end to end with exit 0. Extrapolated per seed (one slot): MATH500 about 2 h, GPQA about 4-5 h, AIME24 about 4 h. A3 on Qwen-7B (72 seed-cells) over 6 slots takes about 2-2.5 days.

## Block status
| Block | Status |
|---|---|
| A5 figures (F1-F3) | done: `A/figures/` (PDF + CSV). F1 is for Qwen-7B/14B; Llama-8B is added after its r_v=16 calibration. |
| A3 tables | done for Qwen-7B and Qwen-14B (`A/tables/`); rule-1 hashes reproduced 6/6; Llama-8B after calibration |
| A3 decodes, Qwen-7B (72 cells) | running on GPUs 0, 2, 3 (`scripts/revision/queue.sh`, `jobs_A3_qwen7b.txt`) |
| Llama-8B r_v=16 calibration | running on GPU 1 |
| A1 disjoint calibration (MATH500, AIME24; Qwen-7B, Llama-8B) | queued on GPU 1 after the Llama calibration; problems in `A/calib_data/MANIFEST.json` |
| A3 Llama-8B -> A1 -> A2 (MATH500 row) -> B decodes | chained (`scripts/revision/chain.sh`) |
| A4 | not started; needs the stack check |
| C1-C3, C5 | parked (see Open questions) |

## A5 results
F3, the held-out output error of the rule's eta_h relative to the 21-point grid minimum (value rank 8, layers >= 1, three tasks pooled):

| Model | Heads | Median | 90th percentile | Max |
|---|---|---|---|---|
| Qwen-7B | 324 | 0.08% | 3.21% | 23.8% |
| Llama-8B | 744 | 0.44% | 7.62% | 140.4% |

The maxima are the rare non-convex heads.
