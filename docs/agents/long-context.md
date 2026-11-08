# Brief: transfer OVAL's pieces to long context (LongBench v2 and LongGenBench)

Read [docs/agents/README.md](README.md) first: survey, then confirm the
configuration with the user, then run. In this brief the confirmation step
matters even more than usual, because part of the work is design.

## Status

_Owned by the agent running this brief. Update it at each step: survey
report, design proposal, confirmed configuration (with date), progress,
results, deviations._

**2026-11-08: survey done; design proposal sent to the user and awaiting confirmation. No GPU job has run.**

### Survey report

**OVAL in one paragraph.** Every 32-token page of the paged KV cache keeps, per KV head, a rank-r record: a mean, a basis and coefficients. The four pieces use that record in four ways:

| piece | what it does | harness flag(s) | filename tag | evidence so far |
|---|---|---|---|---|
| 1. retrieval | scores pages by the logsumexp of logits rebuilt in a rank-8 basis of `M_eta` (key PCA at eta 0, output-aware at eta 1); FreeKV's avgSM / top-k / spec_ret do the rest | `--method spec_ret --page_rep oval --oval_rank 8 --eta E` | `oval_r8_eta<E>_q0` | Tables 1-2 (paper), including long context |
| 2. key compression | selected pages' keys are rebuilt from the same rank-8 record instead of read from the cache | `--oval_recon` | `_recon1` | reasoning only |
| 3. value compression | selected pages' values are rebuilt from a per-page PCA record | `--oval_vrank R --vbasis pca` | `_vr<R>_vpca` | reasoning only (Qwen-7B: ranks 8/16 within noise of exact) |
| 4. per-head eta | rule 1, `eta* = clip((a-c)/(a+b-2c), 0, 1)` per (layer, KV head), from the output errors at eta 0 and 1 on held-out traces | `--eta_table <json>` | `_etab<hash>` | reasoning only |

Sink and recent tokens always stay exact. **Prefill is dense** in the harness. Pages are built when decoding starts, and selection and reconstruction act only on decode steps.

**Long-context harness.**
- `eval.LongBench2.pred`: 503 questions; middle-truncation to `model2maxlen` = 65536; greedy; up to 128 new tokens (answers are about 10). It resumes by `_id`, but drops empty outputs, which then re-run on every resume.
- `eval.LongGenBench.pred`: `Dataset_short.json`, 400 items (Week / Floor / Menu Week / Block x 100), prompts of about 430 words, `max_gen 16000`, a sampling stop string, and `append_to_json` per record.
- The paper protocol is in the brief. Note that "B = 2048" means 128+128+1792 for LongBench v2 and 512+512+1024 (the default `--budget`) for LongGenBench. The reasoning runs used 512+512+2048.
- All the `--oval_*` / `--eta_table` flags reach both pred scripts through `eval/util.py`, but they have never been run there.

**On disk (machine C).**
- The weights of all three models are cached, as are LongBench v2, LongBench v1 (`data.zip`) and the LongBench-Pro card.
- There are no long-context outputs, captures or eta tables for any of these models. The paper's raw generations are on another machine.
- The rule-1 code splits cleanly:
  - `stage_stats` / `window_stats` only need captures holding (K, V, Q at window positions).
  - `stage_capture` is reasoning-specific: it uses the dataset files, the `<think>` prompt, and `generate()` with T 0.6, top-p 0.95 for 6144 tokens.

**Inconsistencies and gaps found:**
1. **Values PCA rank 32 is lossless.** A centred 32-token page has rank <= 31. `build_value_summary` clamps `r = min(rank, n-1, d)`, so `--oval_vrank 32` rebuilds the values exactly up to bf16 rounding (docs/3 calls it "the lossless reference"). Its record (mean + 31 basis vectors + coefficients) is also larger than the raw page. The brief proposes rank 16; the agreed setting is rank 32. This needs a decision (see the proposal).
2. **LongGenBench sampling is unseeded.** `eval/LongGenBench/pred.py` never calls `torch.manual_seed`. `--seed` only adds `-seed<S>` to the filename, so "seeds" are neither reproducible nor distinct by construction. The reasoning pred seeds properly.
3. **LongGenBench has no resume.** `append_to_json` re-reads and re-writes the whole list on every record. A restart appends duplicates; the scorer docstring says "last 400 records" for this reason. `--data_from` / `--data_idx_to` shards land in `res-test/`.
4. **LongGenBench top-p.** The brief says top-p 0.95, but FreeKV's own command has no `--top_p` (default 1.0). Nothing in the repo records which one Table 2's OVAL cells used.
5. **No LongGenBench scoring in the repo.** `analysis/score_accuracy.py` says it scores LongGenBench but contains only the LongBench v2 loop, and `all_tables.py` has no LongGenBench accuracy table. CR x Acc needs the Qwen3-32B judge under vLLM (`eval/LongGenBench/eval.py`). vLLM is not installed in `freekv`, Qwen3-32B is not cached (~65 GB), and the disk has 198 GB free (97% used).
6. **64K calibration captures do not fit.** One capture holds K and V for every layer: 3.8 GB (Qwen-7B), 8.6 GB (Llama-8B) or 12.9 GB (Qwen-14B) per prompt. `stage_capture` also runs a full-logits forward (64K x 152K vocab, about 20 GB). Long-context calibration must stream capture into stats per prompt rather than save captures.
7. **LongBench v2 pred.** `--data_from` references an undefined `data` (NameError). `--out_root_dir` is ignored (the output root is hard-coded). `lb2_eta_sweep.sh` deletes partial files although pred already resumes by `_id`. The brief's "append pitfall" applies only when two different runs share a path.
8. **The rule-1 regime differs.** In LongBench v2 only about 10 decode steps per question see compressed pages, and they attend to about 60K tokens. Reasoning calibration took windows of generated tokens at 2-6K context. Calibration queries should therefore be decode-time queries at the full prompt length.

## Where things stand

All of OVAL's pieces beyond retrieval were **built and validated on long
reasoning only** (R1-distilled models, MATH50 / AIME24 / GPQA50c):

| piece | long reasoning | long context |
|---|---|---|
| 1. retrieval | done | **done**: the submitted paper's Table 2 (LongBench v2, LongGenBench) |
| 2. key compression (`--oval_recon`) | done | not run |
| 3. value compression (`--oval_vrank`) | done | not run |
| 4. per-head eta (rule 1, `--eta_table`) | done | not run; **no calibration path exists** |
| 5. full pipeline | to run (three reasoning agents) | **your task** |

The long-context numbers in `results/paper_tables.md` (README Table 2) are
retrieval only: exact keys and values, `eta*` = best of five shared etas per
cell. Your job is to bring pieces 2-4 to long context and measure the full
pipeline there, **on both benchmarks of Table 2 -- LongBench v2 (long-context
understanding) and LongGenBench (long generation) -- and all three of the
paper's long-context models** (confirmed by the user, 2026-11-03):

| harness name | HF model |
|---|---|
| `llama-3.1-chat-8b` | `NousResearch/Meta-Llama-3.1-8B-Instruct` |
| `qwen-2.5-chat-7b` | `Qwen/Qwen2.5-7B-Instruct` |
| `qwen-2.5-chat-14b` | `Qwen/Qwen2.5-14B-Instruct` |

## What already works

- **Harness flags.** `eval.LongBench2.pred` (and `LongGenBench`,
  `InfiniteBench`) parse the shared arguments in `eval/util.py`, so
  `--oval_recon`, `--oval_vrank R --vbasis pca` and `--eta_table <json>` reach
  the attention patch, and the output filename gets the same tags as in
  reasoning (`_recon1`, `_vr16_vpca`, `_etab<hash>`;
  [results/reasoning/README.md](../../results/reasoning/README.md)). **This
  has never been run on long context**: smoke-test it first.
- **The paper's LongBench v2 protocol** (`scripts/lb2_eta_sweep.sh`):
  `--sink 128 --recent 128 --budget 1792` (B = 2048 in total), page 32,
  avgSM, spec_ret 2 with correction 0.8, greedy, inputs truncated to 64K
  (middle truncation, `config/model2maxlen.json`), `KVC_MAX_TOKENS=65536`.
  LongGenBench: sink = recent = 512, correction 0.9, temperature 0.95,
  top-p 0.95, 16K generation.
- **A pitfall already found**: the LongBench v2 writer appends, so a rerun
  without deleting a partial output file silently mixes records
  (`lb2_eta_sweep.sh` handles it; copy that logic).
- **Scoring**: `analysis/score_accuracy.py` (LongBench v2 / LongGenBench
  from raw generations), `analysis/all_tables.py`.

## What you must design (propose, then confirm with the user)

1. **Rule-1 calibration for long context.** `analysis/eta_rules_calibrate.py`
   only knows the reasoning datasets: it generates 6144-token traces with
   sampling and takes query windows from the generated tokens. Long-context
   understanding is a long prefill (up to 64K) and a short greedy answer.
   Propose:
   - the calibration data, **held out** from what you evaluate: LongBench v2
     has 503 questions and the paper evaluates all of them, so either a
     different long-context source, or a split of LongBench v2 that changes
     comparability with the paper -- say which and why;
   - which query positions calibrate eta (decode steps after the prefill;
     the end of the prompt; both) and how many;
   - the page layout used in calibration (it must match the protocol: sink
     and recent 128, B = 2048).
   Put the new code in a new file (e.g. `analysis/eta_rules_calibrate_lc.py`)
   that imports what it needs from `eta_rules_calibrate.py`, so the
   reasoning agents' calibration cannot change; if you must edit the shared
   file, keep existing behaviour identical and run `analysis/test_eta_rules.py`.
2. **Whether the reasoning operating points hold at 64K context.** Keys rank
   8 and values rank 16 were chosen on 16K-token reasoning. Propose a cheap
   check before the full runs (e.g. the offline output error of the records
   on a few captured long prompts, as `analysis/value_compression_check.py`
   does, or a small harness subset), and what result would make you
   propose a different rank.
3. **Which arms.** At minimum the full pipeline (pieces 1-4) per model. Ask
   whether intermediate arms are wanted to attribute any drop (retrieval +
   keys; + values with a fixed eta), and whether the retrieval-only arm is
   re-run in the same harness as a control or taken from the paper (the
   14B LongBench v2 cell once needed within-harness controls:
   `scripts/lb2_controls.sh`).
4. **The two benchmarks differ in regime, so propose calibration for each.**
   LongBench v2 is a long prefill (up to 64K) and a short greedy answer;
   LongGenBench generates up to 16K tokens with sampling (temperature 0.95,
   top-p 0.95), much like the reasoning traces rule 1 was built on. A rule-1
   table per (model, benchmark) is the default. Also propose how many
   sampling seeds LongGenBench gets (the paper's protocol and the cost
   decide).
5. **A driver**, `scripts/long_context_pipeline.sh`, in the style of
   `scripts/full_pipeline.sh` (resumable, gates on free GPU memory, one
   step per stage), and a doc, `docs/6-long-context.md`, written as piece 6
   in the same format as docs 1-5.

## Proposed configuration (starting point for your proposal)

| setting | proposed |
|---|---|
| benchmarks | LongBench v2, all 503 questions, Overall + Short / Medium / Long; LongGenBench, CR and CR x Acc |
| models | the three above, on both benchmarks |
| retrieval, LongBench v2 | `--method spec_ret --page_rep oval --oval_rank 8`, avgSM, spec_ret 2 / 0.8, sink 128, recent 128, budget 1792, page 32, greedy, 64K truncation, bf16 records (the paper's protocol, `scripts/lb2_eta_sweep.sh`) |
| retrieval, LongGenBench | as above but sink 512, recent 512, spec_ret correction 0.9, temperature 0.95, top-p 0.95, 16K generation (the paper's protocol) |
| keys | `--oval_recon` (scoring rank = reconstruction rank = 8) |
| values | `--oval_vrank 16 --vbasis pca` (pending check 2) |
| eta | `--eta_table` from rule 1, one table per (model, benchmark), calibrated on held-out data with rank-16 values (pending designs 1 and 4) |
| reference | the paper's retrieval-only numbers (`results/paper_tables.md`), unless the user asks for a re-run |

## Machine and cost

Unknown until you measure: 64K prefills make each cell much heavier than a
reasoning cell, and the 14B model needs ~30 GB of weights plus the cache.
Time one smoke run per model (a handful of questions) and give the user a
full estimate in Step 2.

## Files you own

- new: `analysis/eta_rules_calibrate_lc.py` (or similar), `scripts/long_context_pipeline.sh`,
  `docs/6-long-context.md`
- `results/long_context/` (new): copy finished harness outputs here from
  `third_party/FreeKV/accuracy/eval/<benchmark>/results*/` (the harness
  writes inside the untracked FreeKV tree), plus per-model calibration
  stats, eta tables and a scored table
- the Status section of this file
- once piece 6 exists, one row for it in `docs/README.md` and the README
  "Start here" table (tell the user when you add it)
