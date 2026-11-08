# Block C (vision) — report

Branch `revision-c` (cut from `vision` at `bb7ce66`), machine **mint06**. Brief:
`docs/agents/revision-block-c.md` on `main`. C4 is out of scope (authors), not run.

## Blockers

- None at the moment. (No HF token: LongVideoBench and MLVU are gated, so C2 runs on
  LVBench, as the brief decides.)

## Discrepancies

- (none yet)

## Progress

| step | status |
|---|---|
| Environment check | done (below) |
| Env rebuild | done (stack identical to Tab. 5; `STACK.md`) |
| Reproduce 20 Long videos of `row1_full_n64` and `row3_ovalfull_n64` | **PASS, bit-exact** (below) |
| C1 Qwen2.5-VL-7B on Video-MME | not started |
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

## Open questions

1. **"20 videos" for the reproduction.** I use the first 20 Long videos (601-620),
   which are all 60 of their questions, not 20 questions. 20 questions is what the
   earlier `diag_newmachine` check used. 60 is the stricter reading.
