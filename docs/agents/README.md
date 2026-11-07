# Agent briefs

Four agents, one brief each. Read this page first, then your brief.

| brief | task | models |
|---|---|---|
| [reasoning-qwen7b.md](reasoning-qwen7b.md) | full pipeline on long reasoning | R1-Distill-Qwen-7B |
| [reasoning-llama8b.md](reasoning-llama8b.md) | full pipeline on long reasoning | R1-Distill-Llama-8B |
| [reasoning-qwen14b.md](reasoning-qwen14b.md) | full pipeline on long reasoning | R1-Distill-Qwen-14B |
| [long-context.md](long-context.md) | transfer pieces 2-4 to long-context understanding, then run the full pipeline there | Llama-3.1-8B, Qwen-2.5-7B, Qwen-2.5-14B (Instruct) |
| [vision-videomme.md](vision-videomme.md) | finish the Video-MME runs (branch `vision`) | LLaVA-Video-7B-Qwen2 |

## The protocol every agent follows

### Step 1. Survey the repo (most important; do not skip)

Before touching a GPU, read, in this order:

1. `README.md` ("Start here") and [docs/README.md](../README.md): OVAL is one
   method with four complementary pieces -- retrieval, key compression,
   value compression, per-head eta -- and a full pipeline combining them.
2. The five piece docs, `docs/1-retrieval.md` ... `docs/5-full-pipeline.md`.
   For each piece know: what it does, its harness flag(s), the evidence it
   works on its own, and the filename tag it leaves.
3. [results/reasoning/README.md](../../results/reasoning/README.md): how to read a
   result filename and which files belong to which experiment.
4. [scripts/README.md](../../scripts/README.md), then `scripts/full_pipeline.sh`,
   `analysis/eta_rules_calibrate.py`, and the `--oval_*` / `--eta_table` flags
   in `patches/freekv.patch` (`eval/util.py` hunk).
5. `docs/harness/` (how OVAL sits inside FreeKV, speculative retrieval) and
   [archive/README.md](../../archive/README.md) (what is finished and why --
   do not resurrect archived work).

Then write a short survey report in the **Status** section of your brief
(and send it to the user): the four pieces in your own words with their
flags, what already exists on disk for your model(s), and anything that
looks inconsistent or unclear. If something in the docs is wrong, say so
rather than working around it.

### Step 2. Propose the configuration and wait for the user's confirmation

Each brief lists a **proposed configuration** and the **open questions** that
are the user's to decide. Present to the user, as one message:

- every setting you will run (models, tasks, seeds, all harness flags, value
  rank, how eta is calibrated, GPUs and slots),
- the answers you propose to the open questions,
- expected GPU time and disk, from the numbers in the brief or a measured
  smoke run,
- exactly which files you will create and commit.

**Do not start any GPU job -- calibration captures included -- until the user
has confirmed the configuration explicitly.** A CPU-only check (unit tests,
a dry-run of the queue) is fine. If the configuration has to change later,
stop and confirm again.

### Step 3. Run

Use the drivers in `scripts/` as documented; they are resumable and gate on
free GPU memory. Never run the dense control or the other baselines unless
the user asks: their numbers are on disk or in the paper.

### Step 4. Commit and push

- Push to `main` as results land; `git pull --rebase` first; commit messages
  say what was run and what was checked.
- Commit only files you own: your model's results, your calibration stats
  and tables, new scripts/docs your brief names, and your brief's Status
  section. Do not edit other agents' files or shared docs; if a shared file
  (e.g. `analysis/eta_rules_calibrate.py`) must change, keep existing
  behaviour identical, run `analysis/test_eta_rules.py`, and say so in the
  commit.
- Never commit `logs/`, calibration captures, or partial seeds.
- Names: the method is OVAL. There is no "codec" method; do not introduce
  the word.

### Step 5. Report

When done (or blocked), update your Status section with what ran, the
results table, anything that deviated from the confirmed configuration, and
anything you found wrong in the repo.
