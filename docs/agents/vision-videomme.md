# Brief: finish the OVAL video experiment (Video-MME, LLaVA-Video-7B)

Branch **`vision`** of oval-kv (not `main`). Read [docs/agents/README.md](README.md)
first, then `STATUS.md` at the repo root: it holds the model choice, the
query-agnostic protocol, the gates, the calibration and every user decision.
The configuration below is already **confirmed by the user (2026-11-06)**; do
not change flags, ranks, budgets or the eta table. Still send the user your
survey report and plan (machine, GPUs, which jobs) and wait for a go-ahead
before the first GPU job.

## Status

_Owned by the agent running this brief. Update at each step._

- 2026-11-06: paused on the first machine (4x RTX A6000 shared with another
  user's training; repeated CUDA OOM). Long split done for 5 of 12 rows;
  78 jobs remain in [vision/jobs_remaining.txt](../../vision/jobs_remaining.txt).
  First machine stopped; all remaining jobs belong to the next agent.

### 2026-11-06/07, second machine (`mint04`) -- setup + survey

**Hardware.** 4 x NVIDIA RTX A6000, 49,140 MiB each, driver 535.309.01 /
CUDA 12.2. Free at survey: GPU0 48,666 / GPU1 45,231 / GPU2 48,666 / GPU3
48,666 MiB -- one other user (`thrashc`) holds 3.4 GB on GPU1 only. This is
**much freer than the first machine** (which had 27-29 GB taken per card and
OOMed); at ~20 GB peak per job, four concurrent jobs fit comfortably.
**GPU type is the same as the first machine (RTX A6000)**, so by the brief's
own rule the partial rows 4 and 5 do *not* need rerunning and no hardware is
mixed inside a row.

**Disk.** One filesystem (`/`, 3.5 TB) shared with other users, **28 GB free**
after I reclaimed my own regenerable files. This is the binding constraint, not
GPU. Measured, not estimated: frame cache **10.6 MiB per video -> 9.3 GiB for
900 videos** (chunk 01: 30 videos, 318 MB); model weights 15.0 GiB; the 20
video chunks are 94.1 GiB total (largest 4.98 GiB). Peak during the frame build
is ~19 GiB, and ~24 GiB is resident afterwards, leaving ~4 GiB. To stay inside
that I build the **frame cache before downloading the model** -- they never need
to be resident at the same time. No experiment setting is affected.

**Environment (deviation, documented).** There is no `ovalkv` env on this
machine; the only suitable env is `freekv`: **torch 2.5.1+cu121**,
transformers 4.45.2, flash-attn 2.7.4.post1, flashinfer 0.2.4. STATUS.md
documents the first machine as torch 2.8.0+cu128 / flash-attn 2.8.3. Building a
matching env would cost 8-12 GB of disk that is not available. `vision/env.sh`
is therefore pointed at `freekv` (the brief explicitly allows editing it);
**this edit is local and deliberately not committed**, since it would break the
first machine. The equivalence check in the brief is exactly the gate for this:
`frames_sha1` / `pixels_sha1` / `ids_sha1` must match, and answer agreement is
reported. No flags, ranks, budgets or the eta table were changed.

**Things in the docs that are wrong or need a line added:**

1. **`pip install decord timm==0.9.16` silently upgrades torch.** Run as written
   it pulled **torch 2.14.1+cu130**, which does not run on a CUDA 12.2 driver
   ("NVIDIA driver on your system is too old"); CUDA stopped initialising in the
   env. Repaired with
   `pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchvision==0.20.1+cu121`.
   The brief should pin torch or add `--no-deps` here. (My error for running it
   unpinned, but the instruction as written causes it.)
2. **`frames.py` peak disk is ~2x a chunk, not the cache size.** `do_chunk`
   extracts every mp4 of the zip into a tmpdir under `$VMME_ROOT` and only then
   deletes the zip, so zip + extracted mp4s (~10 GiB) coexist. The brief's
   "~101 GB in total ... the frame cache is ~9 GB" is true but reads as if 9 GB
   is the requirement. Worth stating for machines near the disk limit.
3. **"the 20 zips, ~101 GB"**: the repo actually holds 20 video chunks totalling
   **94.1 GiB**, plus a `subtitle.zip` (0.01 GiB) that this protocol does not
   use. Minor.
4. `docs/agents/README.md` is the reasoning-era page: its reading list (the five
   piece docs, `results/reasoning/README.md`, `scripts/full_pipeline.sh`) is for
   the long-reasoning agents, and its Step 2 tells the agent to propose a
   configuration -- already confirmed here. The vision brief + `STATUS.md` are
   the operative documents. Not wrong, just not aimed at this task.

**Setup done on this machine:** worktree `~/oval-vision` on branch `vision`
(main untouched, separate worktree); `setup.sh` applied `patches/freekv.patch`
cleanly and installed the branch's token-domain `oval/page_basis.py` as
`third_party/FreeKV/accuracy/kvc/patch/oval.py` (verified); LLaVA-NeXT at
`bce12e47` (`pip install --no-deps -e`) and lmms-eval at `d45b4ad5`; decord
0.6.0, timm 0.9.16; Video-MME annotations fetched. Frame cache building
(chunk 01 done, 02-20 running).

**Next:** finish the frame cache, download the model, run the 20-question
equivalence check, report hash match + answer agreement, then the queue.

## Where things stand (Long split = 900 questions)

| row | run dir (`vision/runs/`) | Long | Short+Medium |
|---|---|---|---|
| 1 FullKV-64 | `row1_full_n64` | 900/900 (2 shards) | 0 |
| 2b FullKV-44 | `row2b_full_n44` | 900/900 | 0 |
| 2 FullKV-32 | `row2_full_n32` | 900/900 | 0 |
| 3 OVAL full | `row3_ovalfull_n64` | 900/900 | 0 |
| 7 OVAL keys-as-basis | `row7_ovalkeys_n64` | 900/900 | 0 |
| 4 FreeKV | `row4_freekv_n64` | 878/900 | 0 |
| 5 Quest | `row5_quest_n64` | 157/900 | 0 |
| 6 4-bit keys | `row6_q4key_n64` | 0 | 0 |
| 8 SnapKV | `row8_snapkv_n64` | 0 (smoke test passed) | 0 |
| 9 Sliding window | `row9_sliding_n64` | 0 (smoke test passed) | 0 |
| 10 ArkVale | `row10_arkvale_n64` | 0 (smoke test passed) | 0 |
| 11 RaaS | `row11_raas_n64` | 0 (smoke test passed) | 0 |
| question-in-prompt FullKV-64 | `fullkv64_question_in_prompt` | all 2,700, `--protocol official` | |

Runs are **resumable**: `run_videomme.py` reads the question ids already in its
`shard{s}of{n}{tag}.jsonl` and skips them. The jsonl files are committed, so a
fresh clone continues exactly where the first machine stopped. Never edit or
delete them; never run the same (row, split, shard) on two machines at once.

## Setting up a new machine

1. Clone, `git checkout vision`. Build the `ovalkv` env and FreeKV checkout as in
   `docs/harness/freekv-port.md` (FreeKV pin + `patches/freekv.patch`), then
   install `oval/page_basis.py` as `third_party/FreeKV/accuracy/kvc/patch/oval.py`.
2. Into `third_party/`: LLaVA-NeXT at `bce12e479bc4dfee2b9c50c88137b01ff51bd483`
   (`pip install --no-deps -e third_party/LLaVA-NeXT`) and lmms-eval at
   `d45b4ad57c6305a16e9c1133eb394af832e1b349` (used for its Video-MME prompt only;
   no install needed). Also `pip install decord timm==0.9.16`.
3. Edit `vision/env.sh` if the env is not at `~/miniconda3/envs/ovalkv`;
   `TORCH_CUDA_ARCH_LIST` must match the GPU (8.6 = A6000, 8.0 = A100, 9.0 = H100).
4. Data (root `$VMME_ROOT`, default `~/vmme`):
   `python vision/download.py model`, `python vision/download.py ann`, then
   `python vision/frames.py --chunks 1 2 ... 20 --n 64` (streams the 20 zips,
   ~101 GB in total, one at a time and deletes each; the frame cache is ~9 GB).
   The eta calibration is already done (`results/vision/eta/tables/llava-video-7b_VCAL_r1.json`,
   hash 42bf0455); no calibration data is needed.
5. **Equivalence check before any queued job** (one GPU, ~10 min):
   ```
   source vision/env.sh
   QIDS=$(python - <<'EOF'
   import json; print(" ".join(json.loads(l)["question_id"] for l in list(open("vision/runs/row1_full_n64/shard0of2.jsonl"))[:20]))
   EOF
   )
   CUDA_VISIBLE_DEVICES=0 python vision/run_videomme.py --protocol stream --budget 1792 --sink 128 \
       --recent 128 --page_size 32 --GQA_policy avgSM --skip_layer 1 --sparsity 1 --last_layer_budget 0 \
       --corr_group avg --batch_size 1 --method full --frames 64 --durations long \
       --qids $QIDS --out vision/runs/diag_newmachine
   python vision/compare.py vision/runs/row1_full_n64 vision/runs/diag_newmachine
   ```
   `frames_sha1`,
   `pixels_sha1` and `ids_sha1` must match on all 20; if any differs, stop and
   report. Answers may differ on a few low-margin questions across GPU types
   (bf16; 3/100 flips with margins < 0.25 between protocols on the first
   machine). Report the agreement to the user. The new machine has the same
   hardware (RTX A6000; user, 2026-11-06), so the partial rows 4 and 5 are
   continued, not rerun. Every answer line records `host` and `gpu` from now on.

## Running the remaining jobs

`vision/jobs_remaining.txt` lists the 78 jobs in the order the user chose (Long
split for every row first, then Short+Medium with `--file_tag _sm`, then the
question-in-prompt run). One job per GPU, ~20 GB peak each (SnapKV 19.5 GB):
```
mkdir -p vision/queue vision/logs && cp vision/jobs_remaining.txt vision/queue/jobs.txt
for g in 0 1 2 3; do nohup bash vision/worker.sh $g >/dev/null 2>&1 & done
```
`vision/queue/done.txt` logs START/END rc per job. To pause: kill the
`worker.sh` processes first, then the `run_videomme.py` children, then put
their command lines back at the top of `vision/queue/jobs.txt`. Throughput on
the shared A6000s was ~12-19 s per question, ~3-4 h per row for the Long split
on 4 GPUs. No latency numbers go in the paper from shared GPUs.

Commit the jsonl files after each finished row (small, ~0.5 MB per row), on
branch `vision`, author "Ashkan Shahbazi <ashkan.shahbazi@vanderbilt.edu>".
`vision/logs/` and `vision/queue/` are git-ignored.

## What to report

When the Long split is in for all 12 rows: `python vision/summarize.py` and
`python vision/memory.py vision/runs/row1_full_n64 vision/runs/row*_n64 ...`.
The user asked for the Long-split table with **paired cluster-bootstrap CIs
(over videos) for every row vs FullKV-64 and vs FullKV-32**, and the exact
memory % per row (FullKV-44 included). `summarize.py` still needs: RUNS
entries for rows 8-11, PAIRS for every row vs both references, a memory
column, and a dagger on SnapKV (run without eviction, an upper bound).
Then the same for all 2,700 questions, and the question-in-prompt FullKV-64
overall vs the published 63.3, reported separately from the protocol rows.

No budget-1024 runs (user, 2026-11-06). For context, the first-machine
preliminary finding was FullKV-32 vs FullKV-64 on Long = -0.7 [-2.9, +1.4],
i.e. frames barely matter at this length.

The first machine is stopped and runs nothing (user, 2026-11-06); this agent
owns all remaining jobs.
