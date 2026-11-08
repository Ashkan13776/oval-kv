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
| C1 Qwen2.5-VL-7B on Video-MME | integration tested, eta calibrated; full runs **running** (all 2,700 questions, 7 arms) |
| C2 LVBench | not started |
| C5 analysis | not started |
| C3 video baselines | not started |
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
