# Full pipeline: what to run next

**Status (2026-11-03): instructions written, nothing run yet.** This is the
current task for every agent working on long reasoning. Earlier experiments
each tested one component in isolation; this run evaluates all of them
together, and nothing should be extrapolated from the isolated runs.

## The configuration

| component | setting | flag(s) | evidence it works alone |
|---|---|---|---|
| page retrieval | OVAL scoring with rank-8 key records, avgSM GQA combine, FreeKV spec_ret | `--method spec_ret --page_rep oval --oval_rank 8` | `results/codec/reasoning_k8_*.md` |
| keys | selected pages rebuilt from the rank-8 key records (records ARE the key store) | `--oval_recon` | `results/codec/recon_rank_qwen7b.md` |
| values | selected pages rebuilt from rank-16 plain-PCA value records | `--oval_vrank 16 --vbasis pca` | `results/codec/value_rank_qwen7b.md`, `docs/value-codec.md` |
| eta | one per (layer, KV head), rule 1, **calibrated with rank-16 values** | `--eta_table results/eta_rules/vr16/tables/<model>_<task>_r1.json` | `docs/eta-rules.md` (calibrated at value rank 8) |
| protocol | B = 2048, sink 512, recent 512, page 32, T = 0.6, top-p 0.95, max_gen 16384, spec_ret 2 / 0.9, bf16 records, seeds 42-49 | as in every codec cell | |

Why eta is recalibrated: rule 1 picks the eta that minimises the method's own
output error, and that error includes the value reconstruction. The existing
tables (`results/eta_rules/tables/`) were fitted with rank-8 values, so for the
rank-16 pipeline they are recomputed at rank 16 (`--vrank 16`). Only the two
endpoint errors (eta = 0 and 1) are needed, so this is cheap.

Grid: 3 tasks (MATH50, AIME24, GPQA50c) x 8 seeds per model = **24 seed-cells
per model**.

## Who runs what

Split by model. Each model writes only its own files (see "What to commit"),
so machines never touch the same file.

| model | machine | status |
|---|---|---|
| `ds-r1-qwen-7b` | to be assigned (a machine with free 48GB GPUs) | not started |
| `ds-r1-llama-8b` | same as qwen-7b | not started |
| `ds-r1-qwen-14b` | separate machine (needs ~40GB free per cell) | not started |

Update this table in your first commit when you start, and again when done.

## Steps

### 0. Setup (once per machine)

```bash
git clone https://github.com/Ashkan13776/oval-kv && cd oval-kv
./setup.sh                               # clones FreeKV at the pinned commit, applies patches/freekv.patch
export KV_ROOT=$PWD
export KV_PY=/path/to/env/bin/python     # torch 2.8 + cu128, transformers 4.45.2, flash-attn, flashinfer 0.2.4
export PATH=$(dirname $KV_PY):$PATH CUDA_HOME=$(dirname $(dirname $KV_PY))
export TORCH_CUDA_ARCH_LIST=8.6          # A6000; 12.0 on Blackwell
python analysis/test_eta_rules.py        # CPU unit tests, must print "all checks passed"
```

Needs: the model weights from the HF hub
(`deepseek-ai/DeepSeek-R1-Distill-{Qwen-7B,Llama-8B,Qwen-14B}`), and disk for
the calibration captures in `logs/eta_rules/capture/` (6 traces x 3 tasks per
model; about 1 GB per trace for 7B/8B, about 1.5 GB for 14B).

### 1. Run everything for your model(s)

```bash
# qwen-7b + llama-8b on one 4 x 48GB machine
MODELS="ds-r1-qwen-7b ds-r1-llama-8b" GPUS="0 1 2 3" SLOTS=2 MINFREE=20000 \
  nohup ./scripts/full_pipeline.sh > logs/full_pipeline.out 2>&1 &

# qwen-14b (one cell needs ~40GB free; SLOTS=2 only on 80GB+ cards)
MODELS=ds-r1-qwen-14b GPUS="0 1" SLOTS=1 MINFREE=40000 \
  nohup ./scripts/full_pipeline.sh > logs/full_pipeline.out 2>&1 &
```

`scripts/full_pipeline.sh` runs five steps. Every step skips work whose output
exists, so after a crash you re-run the same command. `STEPS="..."` runs a
subset.

| step | what | output | cost |
|---|---|---|---|
| `capture` | 6 seed-7 traces per (model, task), 6144 tokens, post-RoPE K/V + query windows | `logs/eta_rules/capture/` (not committed) | ~25 min per (model, task) |
| `stats` | rule 1's endpoint errors (eta = 0, 1) on traces 1-4, with rank-16 values | `results/eta_rules/vr16/<model>_<task>.json` | minutes per (model, task) |
| `rules` | eta* = clip((a-c)/(a+b-2c)) per (layer, KV head) | `results/eta_rules/vr16/tables/<model>_<task>_r1.json` | seconds |
| `sweep` | supervised harness cells until all 24 per model are complete | `results/codec/<model>-spec_ret/*_etab<hash>_q0_recon1_vr16_vpca-*.jsonl` | 7B/8B: 3-5 h (MATH50), 7-8 h (AIME24, GPQA50c) per cell |
| `score` | avg@8 / pass@8 and eta* statistics per (model, task) | `results/eta_rules/vr16/table.md` | seconds |

The sweep gates on free GPU memory (`MINFREE`), so it waits rather than
crashing if another user holds the GPUs; `logs/eta_rules/supervisor_vr16.log`
says so when it is waiting. It restarts stalled cells and aborts after three
launches that produce nothing.

### 2. Check before the sweep is far along

- `results/eta_rules/vr16/tables/` has one `_r1.json` per (model, task), and
  `logs/eta_rules/full_pipeline_vr16.log` shows `degenerate 0/...` for each.
- `python analysis/eta_rules_queue.py --models <model> --tables_dir results/eta_rules/vr16/tables --vrank 16`
  lists 24 seed-cells for your model.
- The first result file's name contains `_etab<8 hex>_q0_recon1_vr16_vpca`.
  If it says `_eta0.25_` or `_vr8_`, the table or rank did not reach the
  harness: stop and fix it.

### 3. What to commit

Push to `main` as you go (rebase first: `git pull --rebase`). Commit only:

- `results/eta_rules/vr16/<model>_*.json` and `results/eta_rules/vr16/tables/<model>_*` (the tables the cells used),
- `results/codec/<model>-spec_ret/*_vr16_vpca-*.jsonl` files whose tag is `etab<hash>` (complete seeds only),
- `results/eta_rules/vr16/table.md` once your model is complete (regenerate with
  `STEPS=score MODELS="<all finished models>" ./scripts/full_pipeline.sh`),
- your row of the status table above.

Never commit `logs/` or the captures, and do not edit another model's files.

## What to compare against

`analysis/eta_rules_table.py --vrank 16` adds a fixed-eta baseline row where
one exists at value rank 16: `ds-r1-qwen-7b` has fixed-eta cells at rank-16
values (MATH50 0.25, AIME24 1.0, GPQA50c 0.25). Other references already on
disk:

| reference | files |
|---|---|
| keys-only codec, fixed eta, exact values | llama-8b, qwen-14b: `results/codec/<model>-spec_ret/*_eta<E>_q0_recon1-*.jsonl` (5 etas); qwen-7b: summary only, `results/codec/reasoning_k8_qwen7b.md` (its cells are not in the repo) |
| rule 1 at value rank 8 | `results/eta_rules/table.md` (qwen-7b, llama-8b) |
| FullKV and baselines | `results/paper_tables.md` (paper numbers; do not re-run dense) |

## Do not

- Run the dense control or the other baselines: their numbers are on disk or
  in the paper.
- Use the rank-8 tables (`results/eta_rules/tables/`) with `--oval_vrank 16`.
- Change the protocol flags: every cell must differ from the references only
  in the components listed above.
