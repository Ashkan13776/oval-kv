# Block C (vision) of the OVAL revision — brief for the agent on mint06

You are running **Block C only** on mint06 (4 GPUs). Blocks A and B run on mint04 with another
agent; do not touch them. You cannot message that agent; write everything to the repo.

## Where things are

- Repo: `https://github.com/Ashkan13776/oval-kv`. All vision code is on branch **`vision`**
  (`vision/`: `run_videomme.py`, `summarize.py`, `memory.py`, `frames.py`, `calibrate_capture.py`,
  `calibrate_stats.py`, `plot_token_selection.py`, `plot_frames_kept.py`, `queue/`, `worker.sh`).
  Read first: `STATUS.md` and `docs/agents/vision-videomme.md` on that branch (env, harness map,
  how the Tab. 5 rows were run, the shared-GPU OOM history).
- Tab. 5 run directories, all 2,700 Video-MME questions, are committed on `vision`:
  `vision/runs/row1_full_n64` (FullKV), `row2_full_n32` (FullKV N/2), `row3_ovalfull_n64` (OVAL),
  `row4_freekv_n64`, `row5_quest_n64`, `row6_q4key_n64` (4-bit keys), `row7_ovalkeys_n64`,
  `row8_snapkv_n64` (SnapKV, run **without eviction**: an upper bound, daggered and never ranked),
  `row9_sliding_n64`, plus sanity-check dirs (`gateB_*`, `fullkv64_question_in_prompt`).
- Video η table: `results/vision/eta/tables/llava-video-7b_VCAL_r1.json` (on `vision`).
- Work on a new branch **`revision-c`** cut from `vision`. All outputs go under
  `results/revision/C/` and `scripts/revision/C/`; report in `results/revision/C/REPORT_C.md`.
  Commit and push often (`git pull --no-rebase` first; never force-push, never delete branches).
- mint04 has one uncommitted change in `vision/summarize.py`: SnapKV is added as a daggered
  upper-bound row (`DAGGER = {"SnapKV"}`), excluded from medals, and every row is paired against
  both FullKV references. Apply the same rule in your tables.

## Do first on mint06

1. Check GPUs (are they shared? the Tab. 5 machine had another user's ~28 GB job on every GPU),
   free disk, and which models/datasets are cached. Video-MME and LLaVA-Video-7B-Qwen2 must be
   downloaded if absent; Qwen2.5-VL-7B is ~16 GB; LongVideoBench is gated and large. Put what
   you find, and anything you cannot get (HF token, gated access), under "Open questions".
2. Rebuild the env from `docs/agents/vision-videomme.md` and reproduce **20 videos of
   `row3_ovalfull_n64` and `row1_full_n64`** before anything new: same predictions expected
   (greedy). Record the stack in `results/revision/C/STACK.md`. If they differ, stop and report.

## Decisions already made by the authors (binding)

- All revision results go to the **supplementary**, not the main paper. Do not edit the paper.
- OVAL setting everywhere: rank-8 keys, rank-16 PCA values, per-head η_h. Call it OVAL, never
  "full pipeline". "Exact keys/values" describes only baselines, FullKV and sink/recent tokens.
- Figures: vector PDF + CSV, paper font (Times clone, e.g. Nimbus Roman), recessive axes;
  categorical blue `#2a78d6`, orange `#eb6834`; sequential single-hue blue ramp
  `#cde2fb → #0d366b` (no viridis/rainbow). F1-style η heatmaps must use the same ramp as
  the reasoning F1 (`analysis/fig_eta_revision.py` on `main`).
- Priority if time runs short (the authors want results within ~a week):
  **C1 on Video-MME → C2 on LVBench (both models, C1 arms) → C5 → C3.** Mark anything not reached as
  not run; never fill in.

## No Hugging Face token: LVBench replaces LongVideoBench (binding, 2026-11-08)

There is no HF token, so gated datasets are unavailable (LongVideoBench and MLVU/MVLU are
`gated=auto`, which still needs a logged-in account). **C2 runs on LVBench instead**, checked
ungated from mint04 on 2026-11-08:

- Videos: `lmms-lab/LVBench` (`video_chunks/videos_chunk_001..014.zip`, 61.8 GB total, ungated).
- Annotations: `zai-org/LVBench`, file `video_info.meta.jsonl` (ungated): 103 videos, about
  one hour each, **1,549 four-way multiple-choice questions**. Each question has `uid`,
  `question` (options inline), `answer`, `question_type` (six categories, e.g. key information
  retrieval) and **`time_reference`** (`MM:SS-MM:SS`, the moment the question refers to).
- Protocol unchanged (64 uniform frames, no subtitles, query-agnostic streaming, greedy bf16,
  budget 2048 = 1792 + 128 + 128). Both models, the C1 arms (FullKV, FullKV N/2, Quest, FreeKV,
  4-bit keys, SnapKV-dagger, OVAL) plus whatever C3 arms are working. Report overall accuracy,
  accuracy per `question_type`, and per video-type (`type` field).
- **C5 evidence-position analysis is back, on LVBench:** bucket questions by the midpoint of
  `time_reference` relative to `duration_minutes` (first / middle / last third) and report
  OVAL, FreeKV and FullKV accuracy per bucket with bootstrap CIs (resample questions). Also
  report how many of the 64 sampled frames fall inside each question's `time_reference`
  span: with hour-long videos many spans contain no sampled frame, so also report accuracy
  restricted to questions whose span contains at least one sampled frame.
- Check disk before downloading (62 GB zipped; unzip one chunk at a time if space is short).
- Use only ungated weights (Qwen2.5-VL-7B-Instruct, LLaVA-Video-7B-Qwen2; InternVL3-8B as the
  M-RoPE fallback). Anything that turns out to need a token goes under "Open questions"; never
  work around it. MLVU is not required.

## C4 is out of scope

The authors will fix the Tab. 5 memory column themselves. Do not run C4 and do not
change or report on that column; for new arms (C1/C3) report accuracy only.

---

## Original brief, verbatim (ground rules + Block C)

## Context and ground rules

Run three experiment blocks for the CVPR revision of OVAL (Output-Aware Local Page Bases for KV Cache Compression): A, per-head η\_h; B, an attention-budget sweep on AIME24; C, a vision section with a second video LLM, a second benchmark and video-specific baselines. Work in the OVAL repo (the code behind https://anonymous.4open.science/r/oval-kv-4424/), which patches FreeKV's accuracy harness.

**What OVAL is, in one paragraph.** Each completed 32-token KV page is stored as a rank-8 key basis (μ\_P, B\_P, C\_P) and a rank-16 PCA value basis. The key basis is the top eigenspace of M\_η, a trace-normalized mix of the page key covariance (η=0, page-local key PCA as in LOCKS) and the output-sensitivity matrix XᵀY G Yᵀ X (η=1). The same basis scores pages for retrieval (log-mass of the rebuilt keys) and rebuilds the selected pages' keys. η\_h is set per layer and KV head by a closed-form rule from output errors measured at η=0 and η=1 on six calibration traces (Eq. 12 of the paper).

**Repo facts to verify first** (taken from the paper's supplement; the PDF extraction may have dropped underscores from flag names, so confirm every flag against `patches/freekv.patch` and `--help` before use):

- Setup: `./setup.sh` clones FreeKV at commit `2c8a7d25c9f3c7c15ce15b2f84cd03f477bd7469` and applies `patches/freekv.patch`; set `KV_ROOT`, `KV_PY`, `TORCH_CUDA_ARCH_LIST` (8.6 on RTX A6000, 12.0 on RTX PRO 6000). CPU unit tests: `python analysis/test_page_basis.py && python analysis/test_eta_rules.py`.
- Drivers and scorers: `scripts/full_pipeline.sh` (Tab. 1), `scripts/supervise_sweep.sh` (shared η), `analysis/eta_rules_calibrate.py` + `scripts/supervise_eta_tables.sh` (η\_h tables), `analysis/eta_rules_table.py`, `analysis/score_avg_pass_at_k.py`, `scripts/pareto_arms.sh`, `analysis/paired.py`, `vision/run_videomme.py`, `vision/summarize.py`, `vision/memory.py`. Tables live in `tables/<model>_<task>_r1.json` (\[n\_layers\]\[n\_kv\_heads\]).
- OVAL retrieval switches: `--method spec_ret --page_rep oval --oval_rank 8 --GQA_policy avgSM --spec_ret_steps 2 --spec_ret_corr 0.9` (0.8 for long context). Compression: `--oval_recon --oval_vrank 16 --vbasis pca`. Per-head table: `--eta_table <T.json>`; one shared value: `--eta E`.
- Fixed in every run, never change: `--page_size 32 --skip_layer 1 --sparsity 1 --last_layer_budget 0 --corr_group avg --batch_size 1`, `OVAL_QUANT=0`, `KVC_MAX_TOKENS` 32768 (reasoning) or 65536 (LongBench v2).
- Reasoning protocol (Tab. S8): FreeKV's data files `math50.jsonl` (50), `aime24.jsonl` (30), `gpqa50c.jsonl` (50); T=0.6, top-p 0.95, up to 16,384 generated tokens, seeds 42–49, `--budget 2048` plus 512 exact sink and 512 exact recent tokens, layer 0 dense, bfloat16. Metrics avg@8 (steps of 0.25 on 50 problems, 0.42 on 30) and pass@8. Models: `deepseek-ai/DeepSeek-R1-Distill-{Llama-8B,Qwen-7B,Qwen-14B}`.

**Ground rules for every run**

1. Protocol is frozen: fixed switches, data files, decoding, seeds and budgets stay as in Tab. S8 unless a block below changes one of them explicitly.
2. All new outputs go under `results/revision/<block>/`; never overwrite an existing result file or table; new η tables get a suffix (`_disjoint`, `_argmin01`, `_layer`, `_pooled`).
3. Smoke test before each block: 1 problem × 1 seed (reasoning) or 20 videos (vision), scored end to end. Then time one full seed, write the extrapolated wall-clock into `REPORT.md`, and only then launch the rest.
4. Record the stack per run in `results/revision/STACK.md`: block, model, arm, GPU, driver, torch, transformers, flash-attn, flashinfer, repo commit. Keep all runs of one block on one stack; keep Block A on the stack that produced Tab. 1 for that model.
5. Drivers are resumable like the existing ones (skip outputs that exist, wait for free GPU memory) and run under `nohup` or `tmux`; log every command with its arguments.
6. Score with the existing scorers and keep per-problem, per-seed correctness files next to every aggregate. For every reported number compute a 95% bootstrap CI over problems (resample problems, average over seeds); for every difference you report, a paired bootstrap CI (same problems and seeds).
7. Never fabricate, interpolate or smooth a number. An incomplete run is reported as incomplete. A protocol decision this document does not cover goes into the "Open questions" list of `REPORT.md`; meanwhile take the most conservative option and say which.
8. Do not edit the FreeKV submodule beyond `patches/freekv.patch`; new code goes into the OVAL module or `scripts/revision/`; unit tests must still pass.

## Block C — Vision section

Deliverable: Tab. 5 extended to two video LLMs, two benchmarks and three new baselines, a KV-memory column that can be audited, and one analysis of where the selection goes. The current section has one model, one benchmark and four baselines, none of them video-specific.

**Protocol, unchanged from Sec. 4.4 and S4.2:** 64 uniformly sampled frames, no subtitles, query-agnostic (the video is cached before the question is seen; the question is then streamed through decoding), greedy bf16, 2048-token attention budget = 1792 selected + 128 exact sink + 128 exact recent, 32-token pages, dense first layer. OVAL: r=8, r\_v=16, per-head η\_h.

**C1. Second video LLM: Qwen2.5-VL-7B-Instruct** (different vision encoder and LM from LLaVA-Video-7B-Qwen2; 28 query heads, 4 KV heads, head dim 128). Two integration points to get right:

1. M-RoPE. The page basis is built on post-RoPE keys; with Qwen2.5-VL those carry the 3-D rotation. The method does not change, but the harness patch must capture keys after M-RoPE and the calibration script must re-rotate its query windows with the same 3-D positions. Unit-test by checking that the rebuilt page logits match the exact ones at r=31.
2. Token count. Fix 64 frames and choose `max_pixels` (temporal patch size 2, so 32 temporal groups) such that the visual token count lands near LLaVA-Video's 13,440; write the exact resolution and count into the report and use the identical frames and count for every arm.

If M-RoPE integration is not working after one working day, switch to InternVL3-8B (standard 1-D RoPE, Qwen2.5-7B LM, a different vision encoder) and say so in the report. Calibrate η\_h for the new model on the same six LLaVA-Video-178K clips as Sec. S4.4 (prefill captured, three windows of 96 prefill queries at T, T−1024, T−2048, 1792-token selection budget, 128 sink and recent). Before any arm, repeat the paper's sanity check: streamed dense versus question-in-prompt dense on the first 100 Long questions; report the agreement count. Arms on Video-MME (2,700 questions): FullKV, Quest, FreeKV, 4-bit keys, SnapKV† (as in Tab. 5), OVAL, plus the C3 baselines.

**C2. Second benchmark: LongVideoBench — REPLACED by LVBench (see the binding section above)** (validation split, multiple-choice, 1,337 questions, videos up to one hour), both models, all arms, same protocol. Its standard setup interleaves subtitles with the frames; run the frames-only variant and state it. Report overall accuracy and accuracy by the benchmark's duration groups. If time remains afterwards: MLVU dev, multiple-choice tasks only.

**C3. Video-specific baselines**, both models, both benchmarks:

- InfiniPot-V: query-agnostic KV compression during streaming, the closest protocol match. Use the authors' code if it loads these models; otherwise implement its two scores (temporal-axis redundancy and value norm) from the paper, at a memory budget matched to OVAL's measured KV memory. State exactly what was implemented.
- ReKV: retrieval of video KV entries with offloading, retrieval budget matched to our 2048 tokens. Official code if it integrates; otherwise its retrieval step in the harness.
- VisionZip (training-free visual-token reduction before the LM) at 50% and 25% of visual tokens; FastV (prune at layer 2, keep 50%) as the second pruning arm if VisionZip's encoder hooks do not apply to the new model.
- OVAL + VisionZip-50%: OVAL on the pruned cache at budget 2048. This is the "complementary" claim of the related-work section made concrete; report its memory under the same model as everything else.

**C4. Memory accounting — SKIPPED (authors handle it; ignore this item).** (no GPU). Rewrite the KV-memory column from a per-token byte model and print the breakdown for every arm: exact tokens (sink, recent, question) × 512 B; paged pages × bytes per page by arm — OVAL 2816 + 5376, 4-bit keys 2560 + 8192, Quest/FreeKV/ReKV 16384 + a 512 B min–max digest, SnapKV 16384 per retained token only, pruning arms 16384 per page over the pruned sequence. Tab. 5 charges Quest and FreeKV 149.3% of FullKV, but a 512 B digest per 32-token page is about 3%, and Tab. S3 charges Quest exactly that (+2.7%): find the cause in `vision/memory.py`, fix it, and re-derive OVAL's 74.3% the same way (50% per page plus 256 exact tokens gives about 51%). Any number that changes gets a line in the "Discrepancies" list of `REPORT.md`.

**C5. Analysis**, both models:

- Fig. 4-style per-frame selection maps averaged over the Long-split questions, plus a line plot of mean selected fraction against frame index (one line per model), so the start/end bias is a number, not a picture.
- Does the bias cost accuracy? LongVideoBench records the position of the referenced moment for each question. Bucket questions by the relative position of that moment (first, middle, last third of the video) and report OVAL, FreeKV and FullKV accuracy per bucket. A larger drop in the middle bucket than at the ends would mean the sink-like pattern hurts; report it either way.
- η\_h on video: the F1 heatmap for the video tables, next to the Qwen-7B reasoning table (LLaVA-Video-7B-Qwen2 shares the LM family), with one sentence on whether visual prefill moves η\_h.

**Order and smoke tests.** [C4 skipped.] C1 on 20 videos end to end, the sanity check, and the full Video-MME pass for the new model; then C3 on Video-MME; then C2 for everything; C5 last. Video-MME and LongVideoBench runs are prefill-dominated (one-letter answers), so time 20 videos per arm and extrapolate before scheduling.

**Deliverables and reporting rules** (apply to Block C; paths become `results/revision/C/`):

**Deliverables, all under `results/revision/`:**

| Item | Content |
| --- | --- |
| `REPORT.md` | one section per block: what ran, on which stack, every table with 95% bootstrap CIs, paired CIs for every difference claimed, links to figures, an "Open questions" list, a "Discrepancies" list at the top |
| `STACK.md` | one row per run: block, model, arm, GPU, torch, transformers, flash-attn, flashinfer, repo commit, date |
| `tables/*.csv` | one CSV per reported table; per-problem, per-seed correctness files beside each |
| `figures/*.pdf` + `.csv` | F1–F3, the budget curve, per-frame selection maps, evidence-position buckets, each with the data behind it |
| `latex/*.tex` | each table as a LaTeX snippet in the paper's column order (Full · the seven baselines · new baselines · OVAL); leave the medals to the authors |
| `scripts/revision/` | every driver used, resumable, with a README listing the exact commands in the order they were run |

**Reporting rules.** Two decimals as in the paper; CIs as \[low, high\]; every table names its stack; no number without a run directory behind it; incomplete runs marked incomplete, never filled in. The final message to the authors is the Discrepancies list, the three-model Tab. 4(b), the transfer matrix, the budget curve, and the extended Tab. 5, in that order, each with one sentence saying what it shows.
