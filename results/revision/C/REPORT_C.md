# Block C (vision) — report

Branch `revision-c` (cut from `vision` at `bb7ce66`), machine **mint06**. Brief:
`docs/agents/revision-block-c.md` on `main`. C4 is out of scope (authors), not run.

## Blockers

- None at the moment. (No HF token: LongVideoBench and MLVU are gated, so C2 runs on
  LVBench, as the brief decides.)

## Discrepancies

1. **The six eta calibration clips were not recorded.** STATUS.md says only "6 YouTube clips from
   LLaVA-Video-178K `2_3_m_youtube_v0_1` archive 98" (34 clips). `llava-video-7b_VCAL.json`
   has no clip names. I recovered them exactly (`scripts/revision/C/identify_calib.py`):
   - re-ran the LLaVA capture and the endpoint stats on every candidate (layers 2, 14, 26);
   - searched the subsets that match the per-clip window signature in the json.

   Exactly one subset per split reproduces the committed pooled losses, with deviation 0.0
   (`calib_clip_identification.txt`; runner-up subsets are 5-9% off):
   - calibrate: `ytb_-lYWmYJ2gsY`, `ytb_GzuT7VIBK4Y`, `ytb_RP5mput8V9A`, `ytb_XkNYStGhGQ8`
   - test: `ytb_gWn_9tVg_-Q`, `ytb_swo-I2Bl3zY`

   That is their ASCII sort order, as `sorted(glob)` in `vision/calibrate_capture.py`.
   The Qwen2.5-VL table uses the same six clips in the same order.

## Progress

| step | status |
|---|---|
| Environment check | done (below) |
| Env rebuild | done (stack identical to Tab. 5; `STACK.md`) |
| Reproduce 20 Long videos of `row1_full_n64` and `row3_ovalfull_n64` | **PASS, bit-exact** (below) |
| C1 Qwen2.5-VL-7B on Video-MME | integration tested, eta calibrated; **done**, all 7 arms x 2,700 |
| C2 LVBench | **done**, both models, 7 arms x 1,549 |
| C5 analysis | eta heatmaps done; evidence position and selection maps **done** (both models) |
| C3 video baselines | implemented and smoke-tested; Video-MME timing queued |
| C4 | out of scope (authors) |

## Environment check (2026-11-08)

- **GPUs:** 4x NVIDIA RTX A6000 (49,140 MiB each), driver 580.178.04. **Not shared** at
  check time: 15 MiB used per card, no other users' processes. (Between 2026-11-05 09:30
  and 14:50 another user's training held ~11 GB on every GPU of this machine. The Tab. 5
  machine had ~28 GB taken per GPU.)
- **CPU / RAM:** 128 cores, 251 GB.
- **Disk:** one shared 3.5 TB filesystem. 26 GB free at check time. With the user's
  approval I deleted my own unrelated data (old projects, conda envs and caches, the
  reasoning calibration traces, a cached R1-Distill-Llama-8B), which gave 317 GB free. Video-MME videos (94 GB)
  are kept unzipped in `~/vmme/videos` because the Qwen2.5-VL frames need a different
  resolution from the LLaVA-Video cache.
- **Cached at check time:** nothing for this block (no Video-MME, no LLaVA-Video,
  no Qwen2.5-VL, no LVBench). Everything is downloaded from ungated repos:
  `lmms-lab/LLaVA-Video-7B-Qwen2` (done), `lmms-lab/Video-MME` (annotations done, videos
  downloading), `Qwen/Qwen2.5-VL-7B-Instruct`, `lmms-lab/LVBench` + `zai-org/LVBench`.

## Reproduction gate (2026-11-08): PASS, bit-exact

Questions: the 60 questions of the first 20 Long videos (601-620). Flags: the `args.json` of the
two Tab. 5 run dirs. Driver: `scripts/revision/C/repro.sh`. Runs:
`results/revision/C/runs/repro_row1_full_n64`, `results/revision/C/runs/repro_row3_ovalfull_n64`.
Comparison: `vision/compare.py`, plus margins.

| arm | input hashes (frames / pixels / ids) | same letter | same text | logit margin (top-1 minus top-2) | acc. on these 60 (Tab. 5 run / here) |
|---|---|---|---|---|---|
| FullKV-64 (`row1_full_n64`) | identical 60/60 | 60/60 | 60/60 | identical 60/60 (max diff 0.0) | 53.33 / 53.33 |
| OVAL (`row3_ovalfull_n64`) | identical 60/60 | 60/60 | 60/60 | identical 60/60 (max diff 0.0) | 48.33 / 48.33 |

The Tab. 5 reference lines for these questions came from the first machine (no `host` field).
The stack is the same (`STACK.md`), and the outputs agree bit for bit.

## C1: Qwen2.5-VL-7B-Instruct on Video-MME

**Integration** (`scripts/revision/C/run_qwen25vl.py`, `qwen_prep.py`):
- **Vision side, once per video, in the official HF stack** (transformers 4.51.3): the 64 frames
  (same indices as the LLaVA cache), the Qwen2.5-VL processor and vision encoder, and
  `get_rope_index`. It stores the stream-protocol prefill (system prompt + video up to
  `<|vision_end|>`) as embeddings plus 3-D M-RoPE positions. Every arm reads the same stored
  file, so frames and token counts are identical across arms (hashes in every answer line).
- **Language model in the harness** (transformers 4.45): the Qwen2-architecture LM of the same
  checkpoint, loaded as `Qwen2ForCausalLM`. The harness's RoPE helper is swapped for M-RoPE in
  the prefill (sections 16/24/24); decode tokens use 1-D RoPE at the Qwen text position, which
  is exact because t = h = w for text.
- **Prompt:** lmms-eval's Qwen2.5-VL setting: system "You are a helpful assistant.", user = video + text,
  the default `videomme_doc_to_text` post-prompt ("Answer with the option's letter from the given
  choices directly."). There is no LLaVA time instruction; Qwen encodes time in its temporal positions.
- **Resolution:** `max_pixels = min_pixels = 420*28*28` (fixed area, aspect kept by `smart_resize`).
  16:9 videos become 420x756 frames, giving 32 temporal groups x 405 = **12,960 visual tokens**
  (LLaVA-Video: 13,440). Prefill 12,976 tokens. The exact count per video is in every answer line.
- **Unit test** (`test_mrope.py`, output `test_mrope.txt`): ALL PASS.
  1. Prefill M-RoPE is bitwise equal to HF `apply_multimodal_rotary_pos_emb`.
  2. Decode M-RoPE is bitwise equal to 1-D RoPE; flashinfer agrees to bf16 rounding.
  3. The harness's cached keys match HF's post-M-RoPE keys: relative error 3e-4 / 1e-2 / 1e-3
     at layers 1 / 14 / 27, bf16 drift.
  4. Brief's check: page logits rebuilt at r = 31 match the exact ones, max |diff| 4e-4 to 9e-4
     on logits up to 62.6. eta = 1 alone is not full-rank by construction, so eta 0 and 0.5
     were tested.
- **Harness vs HF on one video (3 questions):** same answers; margins agree to bf16
  (e.g. 1.75 vs 1.75; 1.625 vs 1.75).

**eta_h calibration** (`qwen_calibrate_capture.py`, `calib_qwen.sh`):
- Same six clips and order as the LLaVA table, same config (3 windows x 96 prefill queries,
  sink/recent 128, budget 1792, r 8, r_v 16, 21-point grid).
- Queries are rotated with their own 3-D positions.
- Table `results/revision/C/eta/tables/qwen2.5-vl-7b_VCAL_r1.json`, **hash 486493e2**.

| model | eta_h mean / median (layers >= 1) | heads at 0 / at 1 | held-out output error: rule 1 / per-head grid optimum | mean regret |
|---|---|---|---|---|
| LLaVA-Video-7B (Tab. 5, 42bf0455) | 0.663 / 0.711 | 5% / 13% | 0.0369 / 0.0363 | 0.00061 |
| Qwen2.5-VL-7B (486493e2) | 0.401 / 0.427 | 15% / 0% | 0.0770 / 0.0765 | 0.00056 |

**Timing (20 Long videos = 60 questions per arm; GPUs shared with my own embedding jobs, so an upper bound):**

| arm | s / question | peak GB | extrapolated, 2,700 questions |
|---|---|---|---|
| FullKV-64 | 7.9 | 16.9 | 5.9 GPU-h |
| FullKV-32 (N/2) | 5.8 | 15.5 | 4.3 GPU-h |
| Quest | 8.0 | 16.9 | 6.0 GPU-h |
| FreeKV | 11.7 | 16.9 | 8.7 GPU-h |
| 4-bit keys | 12.2 | 16.9 | 9.2 GPU-h |
| SnapKV† | 5.9 | 19.6 | 4.5 GPU-h |
| OVAL | 15.2 | 17.3 | 11.4 GPU-h |

Total about 50 GPU-h, i.e. about 13-16 h on 4 GPUs, well under the 5-day cut. **No subset: all 2,700.**
The timing questions are rerun inside the full runs; tables use the full-run shard files only.

**Sanity check (first 100 Long questions):**
- Streamed dense: **51.0%**.
- Question-in-prompt dense (official HF `generate`, unpatched, question inside the prompt): **49.0%**.
- **Agreement, streamed vs question-in-prompt: 98 / 100** predicted letters identical.
- Extra check, harness stream vs HF stream (official HF model, same stored embeddings, question
  streamed as decode tokens): 100 / 100 identical output strings, 51.0% both.
- Runs: `runs/c1_qwen25vl/sanity_{stream_full,prompt_hf,hfstream}_n64`.

**Results (all 7 arms complete, 2,700 / 2,700 questions each).**
Table: `tables/c1_qwen25vl_videomme.md` (+ `_acc.csv`, `_paired.csv`, `latex/c1_qwen25vl_videomme.tex`).

| arm | overall | short | medium | long |
|---|---|---|---|---|
| FullKV-64 | 63.70 [61.85, 65.56] | 75.11 [72.22, 77.89] | 63.78 [60.67, 66.89] | 52.22 [49.00, 55.44] |
| FullKV-32 (N/2) | 61.67 [59.78, 63.59] | 72.67 [69.67, 75.56] | 61.11 [58.00, 64.11] | 51.22 [47.89, 54.56] |
| OVAL | 61.93 [60.00, 63.85] | 72.56 [69.67, 75.33] | 62.22 [59.00, 65.33] | 51.00 [47.67, 54.33] |
| FreeKV | 62.52 [60.63, 64.41] | 73.78 [70.89, 76.44] | 61.22 [57.89, 64.44] | 52.56 [49.33, 55.78] |
| Quest | 63.04 [61.18, 64.93] | 74.22 [71.33, 77.00] | 62.22 [59.00, 65.56] | 52.67 [49.44, 55.89] |
| 4-bit keys | 63.11 [61.26, 64.96] | 74.78 [72.00, 77.56] | 63.11 [60.00, 66.11] | 51.44 [48.11, 54.67] |
| SnapKV† | 57.59 [55.74, 59.48] | 64.33 [61.33, 67.22] | 59.00 [55.67, 62.22] | 49.44 [46.22, 52.78] |

Paired difference vs FullKV-64 (pp, 95% CI):

| arm | overall | short | medium | long |
|---|---|---|---|---|
| FullKV-32 | -2.04 [-3.33, -0.78] | -2.44 [-4.56, -0.33] | -2.67 [-5.22, -0.22] | -1.00 [-3.00, +1.00] |
| OVAL | -1.78 [-2.85, -0.70] | -2.56 [-4.22, -1.00] | -1.56 [-3.67, +0.44] | -1.22 [-2.89, +0.44] |
| FreeKV | -1.19 [-2.22, -0.11] | -1.33 [-3.00, +0.33] | -2.56 [-4.67, -0.44] | +0.33 [-1.33, +2.00] |
| Quest | -0.67 [-1.63, +0.33] | -0.89 [-2.56, +0.67] | -1.56 [-3.44, +0.33] | +0.44 [-1.11, +2.00] |
| 4-bit keys | -0.59 [-1.33, +0.15] | -0.33 [-1.56, +0.89] | -0.67 [-2.00, +0.67] | -0.78 [-2.11, +0.44] |
| SnapKV† | -6.11 [-7.59, -4.67] | -10.78 [-13.56, -8.00] | -4.78 [-7.33, -2.33] | -2.78 [-5.00, -0.56] |

Paired difference vs FullKV-32 (pp, 95% CI):

| arm | overall | short | medium | long |
|---|---|---|---|---|
| OVAL | +0.26 [-1.07, +1.59] | -0.11 [-2.00, +1.78] | +1.11 [-1.44, +3.78] | -0.22 [-2.44, +2.11] |
| FreeKV | +0.85 [-0.45, +2.15] | +1.11 [-0.89, +3.00] | +0.11 [-2.44, +2.78] | +1.33 [-1.00, +3.67] |
| Quest | +1.37 [+0.07, +2.63] | +1.56 [-0.44, +3.56] | +1.11 [-1.33, +3.56] | +1.44 [-0.78, +3.78] |
| 4-bit keys | +1.44 [+0.15, +2.78] | +2.11 [-0.11, +4.22] | +2.00 [-0.33, +4.44] | +0.22 [-2.11, +2.56] |
| SnapKV† | -4.07 [-5.59, -2.52] | -8.33 [-11.33, -5.44] | -2.11 [-4.89, +0.67] | -1.78 [-4.11, +0.56] |

95% CIs: bootstrap over videos (10k). †: reported, never ranked.
Reading: on Qwen2.5-VL-7B, OVAL is below FreeKV, Quest and 4-bit keys on Video-MME and is not
distinguishable from FullKV-32 (unlike LLaVA-Video, Tab. 5). All differences between the four
ranked compressed methods are within about 1.2 pp; none of their long-video gaps to FullKV-64 is significant.

## C2: LVBench (frames only, 64 frames, stream protocol)

**LLaVA-Video timing** (first 3 questions of each of the first 20 LVBench videos = 60 per arm; the
stream protocol re-prefills every question, so cost per question does not depend on how many
questions a video has; GPUs shared):

| arm | s / question | peak GB | extrapolated, 1,549 questions |
|---|---|---|---|
| FullKV-64 | 7.0 | 16.9 | 3.0 GPU-h |
| FullKV-32 | 3.3 | 16.8 | 1.4 GPU-h |
| Quest | 7.2 | 16.9 | 3.1 GPU-h |
| FreeKV | 6.3 | 17.8 | 2.7 GPU-h |
| 4-bit keys | 6.8 | 16.9 | 2.9 GPU-h |
| SnapKV† | 5.6 | 19.5 | 2.4 GPU-h |
| OVAL | 11.4 | 17.4 | 4.9 GPU-h |

Total about 20 GPU-h (~5-6 h on 4 GPUs): no subset, all 1,549 questions, queued after C1.
**Qwen2.5-VL-7B timing** (same 60 questions; 12,960 visual tokens per video; GPUs shared):

| arm | s / question | peak GB | extrapolated, 1,549 questions |
|---|---|---|---|
| FullKV-64 | 3.6 | 16.9 | 1.5 GPU-h |
| FullKV-32 | 2.4 | 15.5 | 1.0 GPU-h |
| Quest | 4.1 | 16.9 | 1.7 GPU-h |
| FreeKV | 4.7 | 16.9 | 2.0 GPU-h |
| 4-bit keys | 5.0 | 16.9 | 2.1 GPU-h |
| SnapKV† | 3.9 | 19.6 | 1.7 GPU-h |
| OVAL | 6.0 | 17.3 | 2.6 GPU-h |

Total about 13 GPU-h (vision encoder precomputed, so cheaper than LLaVA): no subset, all 1,549
questions, queued after the LLaVA C2 runs.

**LLaVA-Video-7B on LVBench: complete (all 7 arms x 1,549 questions).** Table:
`tables/c2_lvbench_llava.md` (+ `_acc.csv`, `_paired.csv`, `latex/c2_lvbench_llava.tex`), with
breakdowns by question type and video type.

| arm | overall | paired vs FullKV-64 | paired vs FullKV-32 |
|---|---|---|---|
| FullKV-64 | 41.64 [38.93, 44.29] | | +2.26 [+0.19, +4.22] |
| FullKV-32 (N/2) | 39.38 [36.59, 42.13] | -2.26 [-4.22, -0.19] | |
| OVAL | 40.87 [38.10, 43.64] | -0.77 [-2.35, +0.78] | +1.48 [-0.59, +3.54] |
| FreeKV | 42.16 [39.31, 44.95] | +0.52 [-1.13, +2.20] | +2.78 [+0.79, +4.73] |
| Quest | 41.32 [38.52, 44.12] | -0.32 [-1.78, +1.16] | +1.94 [-0.06, +3.85] |
| 4-bit keys | 42.35 [39.53, 45.14] | +0.71 [-0.45, +1.92] | +2.97 [+0.85, +5.02] |
| SnapKV† | 36.22 [33.72, 38.69] | -5.42 [-7.69, -3.05] | -3.16 [-5.40, -0.88] |

95% CIs: bootstrap over videos (103 videos, 10k). Reading: none of the four ranked compressed
methods differs significantly from FullKV-64; OVAL is the lowest of them by point estimate and,
unlike FreeKV and 4-bit keys, is not significantly above FullKV-32. No per-question-type gap of
any ranked method to FullKV-64 is significant.

**Qwen2.5-VL-7B on LVBench: complete (all 7 arms x 1,549 questions).** Table:
`tables/c2_lvbench_qwen25vl.md` (+ csv, `latex/c2_lvbench_qwen25vl.tex`).

| arm | overall | paired vs FullKV-64 | paired vs FullKV-32 |
|---|---|---|---|
| FullKV-64 | 41.25 [39.05, 43.55] | | +2.84 [+1.04, +4.67] |
| FullKV-32 (N/2) | 38.41 [35.86, 40.98] | -2.84 [-4.67, -1.04] | |
| OVAL | 39.70 [37.27, 42.17] | -1.55 [-2.98, -0.12] | +1.29 [-0.44, +3.09] |
| FreeKV | 38.48 [35.98, 41.05] | -2.78 [-4.36, -1.17] | +0.06 [-1.75, +1.89] |
| Quest | 38.99 [36.51, 41.52] | -2.26 [-3.95, -0.59] | +0.58 [-1.18, +2.33] |
| 4-bit keys | 40.99 [38.79, 43.30] | -0.26 [-1.57, +1.06] | +2.58 [+0.56, +4.64] |
| SnapKV† | 36.09 [33.44, 38.84] | -5.16 [-7.36, -2.97] | -2.32 [-4.25, -0.33] |

Reading: here OVAL, FreeKV and Quest are all significantly below FullKV-64; OVAL has the smallest
gap of the three selection methods (4-bit keys, a quantization method, is not significantly below
FullKV-64). By question type, OVAL has no significant gap to FullKV-64 in any type; FreeKV and
Quest are significantly below on entity recognition, event understanding and reasoning.

## C5: analysis

**eta_h on video** (`scripts/revision/C/c5_eta_heatmap.py`, no GPU; figure
`figures/c5_eta_heatmaps.pdf` + `.csv`, pairs `tables/c5_eta_pairs.csv`). All five tables are
28 layers x 4 KV heads; statistics over layers >= 1.

| table | median eta_h | at 0 | at 1 |
|---|---|---|---|
| R1-Distill Qwen-7B, MATH500 (vr16, 0c99a26a) | 0.54 | 33% | 11% |
| R1-Distill Qwen-7B, AIME24 (vr16) | 0.33 | 41% | 11% |
| R1-Distill Qwen-7B, GPQA (vr16) | 0.37 | 39% | 10% |
| LLaVA-Video-7B-Qwen2, video (42bf0455) | 0.71 | 5% | 13% |
| Qwen2.5-VL-7B, video (486493e2) | 0.43 | 15% | 0% |

Pearson r over heads: the three reasoning tables agree with each other (0.81-0.87). LLaVA-Video
vs the reasoning tables: 0.26-0.32, mean eta +0.20 to +0.27. Qwen2.5-VL vs reasoning: -0.06 to
0.02, mean shift -0.06 to +0.01. LLaVA-Video vs Qwen2.5-VL: 0.03.
One sentence: visual prefill moves eta_h; the per-head pattern of the reasoning tables does not
carry over to either video table, and for LLaVA-Video (same LM family as R1-Distill Qwen-7B)
it shifts eta_h up by about 0.2-0.27 and nearly removes heads at eta 0.

**Evidence position on LVBench** (`c5_evidence.py`; `c5_evidence.md`, `tables/c5_evidence_*.csv`,
`figures/c5_evidence_buckets.pdf` + `.csv`). Bucket = midpoint of the annotated span over the video
length, in thirds; 2 questions with spans annotated "None" excluded; CIs bootstrap over questions.
LLaVA-Video-7B (complete; 1,547 questions):

| arm | first third (n=708) | middle (n=520) | last third (n=319) |
|---|---|---|---|
| FullKV-64 | 44.49 [40.82, 48.16] | 40.77 [36.54, 45.00] | 36.99 [31.97, 42.32] |
| FreeKV | 45.90 [42.23, 49.58] | 39.42 [35.19, 43.65] | 38.56 [33.23, 43.89] |
| OVAL | 43.93 [40.25, 47.60] | 38.65 [34.42, 42.88] | 37.93 [32.60, 43.26] |
| paired OVAL - FullKV-64 | -0.56 [-2.82, +1.69] | -2.12 [-4.42, +0.19] | +0.94 [-2.82, +4.70] |
| paired FreeKV - FullKV-64 | +1.41 [-0.85, +3.67] | -1.35 [-3.85, +1.15] | +1.57 [-1.88, +5.02] |

No position bucket shows a significant gap to FullKV-64. Caveat: with 64 uniformly sampled frames,
838 / 1,547 annotated spans (54%) contain no sampled frame at all, so the bucket analysis measures
where the question points, not whether the frames are kept. FullKV-64: 43.02 on spans with >= 1
sampled frame vs 40.57 on spans with none.

Qwen2.5-VL-7B (complete; 1,547 questions):

| arm | first third (n=708) | middle (n=520) | last third (n=319) |
|---|---|---|---|
| FullKV-64 | 42.94 [39.41, 46.61] | 41.92 [37.69, 46.15] | 36.68 [31.66, 42.01] |
| FreeKV | 42.09 [38.42, 45.76] | 37.50 [33.46, 41.54] | 32.29 [27.27, 37.62] |
| OVAL | 42.51 [38.84, 46.19] | 39.62 [35.58, 43.85] | 33.86 [28.84, 39.18] |
| paired OVAL - FullKV-64 | -0.42 [-2.68, +1.84] | -2.31 [-4.81, +0.19] | -2.82 [-6.27, +0.31] |
| paired FreeKV - FullKV-64 | -0.85 [-3.25, +1.69] | -4.42 [-7.31, -1.54] | -4.39 [-8.15, -0.94] |

On Qwen2.5-VL the losses of both methods grow with evidence position (later evidence = larger
loss); for FreeKV the middle and last thirds are significant, for OVAL none of the thirds is.
FullKV-64: 45.13 on spans with >= 1 sampled frame vs 38.07 on spans with none.

**Per-frame selection** (`c5_selection.py`; `c5_selection.md`, `figures/c5_selection_{map,line}.pdf`
+ `.csv`). OVAL decode-time page selections logged read-only during real runs (outputs verified
bit-identical to unlogged runs), Video-MME Long, 900 / 900 questions per model. Selected fraction =
share of a frame's pages chosen, averaged over decode steps, layers and heads. Qwen2.5-VL temporal
groups (2 frames each) are shown at frame resolution by repetition. CIs bootstrap over videos.

| model | mean, all frames | first 4 + last 4 frames | frames 4-59 | ratio ends / middle | frame 0 | frame 63 |
|---|---|---|---|---|---|---|
| LLaVA-Video-7B | 12.31% | 27.14% | 10.19% | 2.66 [2.62, 2.71] | 25.47% | 44.73% |
| Qwen2.5-VL-7B | 14.10% | 24.79% | 12.58% | 1.97 [1.92, 2.02] | 36.54% | 20.64% |

Selection concentrates at the ends of the video: LLaVA-Video mostly on the last frames (closest to
the question), Qwen2.5-VL mostly on the first frames; the middle of the video gets about 10-13%.

## C3: video baselines (implemented and smoke-tested; full runs not started)

Official code exists for all three (cloned read-only to `~/c3src`; commits recorded at run time).
None of it runs unmodified on our stack, so each method's core step is ported into the harness.
What is ported, and what deviates:

- **VisionZip** (dvlab-research/VisionZip). Upstream covers CLIP (LLaVA-1.5/NeXT) and Qwen2.5-VL
  *images* only. Its video branch is broken (`Qwen2_5_VL/qwen2_5vl_visionzip.py:1865`), and it
  has no SigLIP path.
  - Ported: the Qwen2.5-VL scoring rule, applied to video tokens.
    - Score = attention received at the last ViT block: head-mean, summed over queries, then
      averaged over each 2x2 merge unit.
    - Global top-k for the dominant tokens; contextual tokens fixed at 5%.
    - Dominant fraction: 45% for the 50% arm (the README's setting) and 20% for 25%.
    - Contextual merge: targets at a uniform stride; each remaining token joins its most
      cosine-similar target by key; merged token = target + mean of the tokens assigned to it.
    - M-RoPE positions of kept tokens are preserved.
  - LLaVA-Video (SigLIP, no CLS): the same CLS-free score, taken at the encoder layer LLaVA
    reads. This is our adaptation; the repo has no SigLIP version.
  - Then FullKV on the pruned cache, and OVAL at budget 2048 on the 50%-pruned cache.
- **InfiniPot-V** (aiha-lab/InfiniPot-V; needs transformers 4.57). Ported the two scores:
  - TaR = negative cosine similarity between post-RoPE keys at the same spatial index in the
    newest 25% of frames vs older frames; the query frames are always kept.
  - VaN = value norm, per layer and KV head.
  - 50/50 split; TaR picks are forced in, the rest filled by VaN.
  - Applied during block-wise prefill (blocks of frames, compress after each block), every
    layer, question excluded.
- **ReKV** (Becomebright/ReKV; LLaVA-OV only, 1-D RoPE). Ported its retrieval step:
  - One block per frame: 210 tokens for LLaVA-Video, 405 tokens (one temporal group) for Qwen.
  - Representative key = mean key, flattened over heads; query = question tokens averaged.
  - One dot product per layer, top-k blocks, retrieved once after the question is streamed;
    that fixed set is then used for answering.
  - Budget matched to 1792 tokens: 8 frames (1680) for LLaVA, 4 groups (1620) for Qwen, plus
    the 128 sink and 128 recent.
  - Deviation: our cache holds post-RoPE keys and keeps original positions. ReKV re-applies
    contiguous RoPE, which has no M-RoPE equivalent.

Implementation status (2026-11-08), every arm smoke-tested on Video-MME 601-1..3:
- Qwen: `visionzip.py` + `qwen_prep.py --vz` (pruned embeddings: exactly 6,480 / 3,240 visual tokens),
  run in the harness (FullKV, OVAL); `qwen_c3.py` (InfiniPot-V, ReKV) in HF on the stored prefill.
- LLaVA: `visionzip_llava.py` (SigLIP adaptation: exactly 6,720 / 3,360 visual tokens) in the harness;
  `llava_c3.py` (InfiniPot-V, ReKV) in HF with explicit positions.
- Budgets: InfiniPot-V "mem" = 16/12 Qwen groups (final 6,480 visual tokens) and 32/24 LLaVA frames
  (6,720); "2k" = 5/4 groups (2,025) and 10/8 frames (2,100). ReKV = 4 groups (1,620) and 8 frames
  (1,680), plus 128 sink and 128 recent.
- The HF-path arms are paired against a dense reference on the same path. For Qwen, the HF stream
  path matched the harness FullKV 100/100. For LLaVA, the HF dense path (`hf_full`, no compression)
  gave the harness answer on 2 of 3 smoke questions; the third flipped at margin 0.125 (bf16 kernel
  differences). So `hf_full` is run as LLaVA's paired reference for these arms.
- Job lines: `gen_jobs_c3.py`.

## Disk notes

- 2026-11-09: the shared disk dropped to 60 GB free (other users). After C1 and C2 finished, the
  Video-MME 32-frame Qwen2.5-VL embeddings (`~/vmme/qwen_emb/*_n32.pt`, 900 files, 40 GB) were deleted.
  No pending job uses them; they are regenerated exactly by `qwen_prep.py --frames 32`.

## Open questions

1. **"20 videos" for the reproduction.** I use the first 20 Long videos (601-620),
   which are all 60 of their questions, not 20 questions. 20 questions is what the
   earlier `diag_newmachine` check used. 60 is the stricter reading.
2. **FullKV N/2 for Qwen2.5-VL** is included in C1, although the brief's C1 arm list omits it
   (C2's list has it). It mirrors Tab. 5 row 2 and is cheap. N/2 is every other cached frame,
   as in Tab. 5, with the fps metadata halved so Qwen's temporal positions stay correct
   (16 groups, 6,480 visual tokens).
3. **Qwen2.5-VL resolution:** I fix the frame area at 420*28*28 px (min = max pixels), so every
   video lands near 13,440 visual tokens with its aspect ratio kept (12,960 for 16:9). A fixed
   784x420 resize would hit 13,440 exactly but distort the aspect ratio.
- **InfiniPot-V budget.** The brief asks for "a memory budget matched to OVAL's measured KV
  memory". OVAL stores every page (~50% bytes/page), so the match would let InfiniPot-V keep
  ~50% of the tokens, a different regime from the 2048-token attention budget of every other
  arm. Conservative option taken: run InfiniPot-V at the token count matching OVAL's KV bytes
  (stated per arm), and if time allows, also at 2048 tokens; both labelled.
- **ReKV and VisionZip on SigLIP/M-RoPE** are ports, not the authors' code paths (see C3);
  arms are labelled "(our port)".

