# Running the long-reasoning codec sweep on a second machine

This splits the sweep by MODEL. Each model writes to its own directory under
`results/codec/`, so two machines never touch the same file and the results
merge by copying directories.

## Why this clones FreeKV

FreeKV here is the **harness, not a baseline**. OVAL is implemented as a patch
to it: `setup.sh` installs `oval/page_basis.py` into
`FreeKV/accuracy/kvc/patch/oval.py`, the patch modifies 19 of their files, and
the reasoning datasets (`math50`, `aime24`, `gpqa50c`) ship in their tree.
There is no standalone OVAL to run.

`--method spec_ret` is FreeKV's speculative-retrieval pipeline -- their query
cache, their cosine-similarity correction, their page budget. We keep all of
it and swap exactly one component: `--page_rep oval` replaces Quest's min/max
page digest with our rank-r basis. (That is why eta=0 reduces to plain
page-local key PCA.)

The arms you will run are:

| flag | what it is | run here? |
|---|---|---|
| `--method full` | dense FullKV | yes, the control |
| `--method spec_ret --page_rep quest` | FreeKV's own method | **no** |
| `--method spec_ret --page_rep oval --oval_recon` | ours | yes |

FreeKV's own numbers are taken from their paper, not re-run, so you never need
`--page_rep quest`.

## Who runs what

Work is split **by model**. Each model writes only to its own
`results/codec/<model>-*` directories, so machines never touch the same file
and merging is a directory copy.

| machine | model | status |
|---|---|---|
| A | `ds-r1-qwen-7b` | rank grid at seed 42; k=8 at eta=0 and at the published eta*; full eta grid at k=8 **in progress** |
| B | `ds-r1-llama-8b` | rank grid at seed 42 only -- **to run** |
| C | `ds-r1-qwen-14b` | rank grid at seed 42 only -- **to run** |

Cost of the full eta grid at k=8 (3 datasets x 5 etas x 8 seeds = **120
cells**), from cell times measured on machine A with 4 x 48GB GPUs:

| model | per-cell (MATH50 / AIME24 / GPQA50c) | 120 cells |
|---|---|---|
| `ds-r1-llama-8b` | 2.9h / 5.3h / 5.4h | ~5-6 days |
| `ds-r1-qwen-14b` | 4.2h / 7.6h / 7.7h | **~8 days** |

**Do not run the dense control.** `DENSE=1` exists in the script but is not
wanted: the FullKV numbers come from the paper's Full column, and machine A
already confirmed the harness reproduces them (AIME24 avg@k 55.00 measured vs
56.66 published, pass@k 83.33 vs 83.33). Re-measuring dense on every model
would cost 24 cells each and tell us nothing new. Run only our method.

If ten days is too long, narrow an axis -- see "Running less" below. The eta
axis is the one worth keeping, since re-deriving eta* is the point.

## Machine B -- ds-r1-llama-8b

Full eta grid at k=8, our method only:

```bash
git clone <this repo> oval-kv && cd oval-kv
export KV_ROOT=$PWD
export KV_PY=$(which python)          # a torch/CUDA env; see requirements.txt
./setup.sh                            # clones FreeKV at the pin, applies the patch,
                                      # installs the OVAL modules
./scripts/build_freekv.sh             # builds the CUDA extension

MODEL=ds-r1-llama-8b ./scripts/codec_reasoning_sweep.sh
```

## Machine C -- ds-r1-qwen-14b

Same command, different model:

```bash
MODEL=ds-r1-qwen-14b ./scripts/codec_reasoning_sweep.sh
```

**The 14B needs >=40GB per GPU.** Measured footprint on machine A was ~37GB
per cell: ~28GB weights in bfloat16, ~6GB KV at the realised sequence length,
~3GB summary slabs. It will not fit a 24GB card. If you only have smaller
GPUs, run fewer concurrent cells (`GPUS="0"`) -- it is the per-cell footprint
that binds, not the total.

MATH50 finishes roughly twice as fast as the other two datasets, so if you
want a complete dataset early, run it first:

```bash
MODEL=ds-r1-qwen-14b DATASETS=MATH50 ./scripts/codec_reasoning_sweep.sh
MODEL=ds-r1-qwen-14b ./scripts/codec_reasoning_sweep.sh           # the rest
```

The second call skips everything the first finished.

## Running less

Narrow any axis; the script skips whatever is already complete:

```bash
# only the PUBLISHED eta* per cell. Cheapest useful run (24 cells),
# but note it inherits the eta* we are trying to re-derive -- see the last
# section. llama-8b: 0.25 / 0.5 / 0.75.  qwen-14b: 0.5 / 0.25 / 0.5.
MODEL=ds-r1-llama-8b         DATASETS=MATH50  ETAS=0.25 ./scripts/codec_reasoning_sweep.sh
MODEL=ds-r1-llama-8b         DATASETS=AIME24  ETAS=0.5  ./scripts/codec_reasoning_sweep.sh
MODEL=ds-r1-llama-8b         DATASETS=GPQA50c ETAS=0.75 ./scripts/codec_reasoning_sweep.sh

# fewer seeds (k is however many you run)
MODEL=<your model> SEEDS="42 43 44 45" ./scripts/codec_reasoning_sweep.sh

# fewer GPUs
MODEL=<your model> GPUS="0 1" ./scripts/codec_reasoning_sweep.sh
```

## Things that will bite

**The script is restartable and concurrency-safe.** It queues only
(dataset, eta) pairs that do not already have all SEEDS on disk, and
`eval.reasoning.pred` itself resumes per problem. Kill it and re-run; nothing
is recomputed and no two processes ever write the same `.jsonl`.

**Budget flags are not what they look like.** The accuracy harness *adds*
sink and recent to `--budget`, so `--budget 2048 --sink 512 --recent 512` is
3072 tokens resident, not 2048. Do not "fix" this -- it matches the reference
reasoning protocol. (The performance harness carves them out instead.)

**`OVAL_QUANT=0` matters.** The default is 1, which both switches to packed
int4/int8 records and tags output paths `_q1`, so the resume check would never
match and every cell would re-run forever. The script exports it.

**`KVC_MAX_TOKENS=32768`.** At the 128K default the rank-31 summary slab alone
is ~15GB on the 14B model. Reasoning sequences never exceed ~16.5K here.

**Results are per model.** Nothing in `results/codec/ds-r1-llama-8b-*`
collides with machine A's qwen directories.

## Reporting back

Copy your model's results directory, e.g.

```
results/codec/ds-r1-llama-8b-spec_ret/   results/codec/ds-r1-qwen-14b-spec_ret/
```

They are small (`.jsonl` predictions). Then score everything with:

```bash
python analysis/score_avg_pass_at_k.py --models <your model>
```

It prints avg@k, pass@k, **k and n per cell**, and the per-seed sequence. Read
`k` before reading anything else: a cell with k < 8 is still running and its
mean is not comparable to a finished one.

## Why the eta grid is being re-measured

The published eta* values do not reproduce in this tree. On qwen-7b at k=8 the
published-eta* gaps over eta=0 came back +0.00 (paper +3.75), +2.50 (+6.67)
and -0.32 (+9.75) on MATH50/AIME24/GPQA50c. So eta* is being measured here
rather than taken from the paper's ablation table, and the same needs doing
for llama-8b and qwen-14b before any comparison is trusted.

For reference, the published eta* per cell is:

| model | MATH500 | AIME24 | GPQA | paper's gap over eta=0 |
|---|---|---|---|---|
| `ds-r1-qwen-7b` | 0.25 | 0.5 | 0.25 | +3.75 / +6.67 / +9.75 |
| `ds-r1-llama-8b` | 0.25 | 0.5 | 0.75 | +1.75 / +11.25 / +5.00 |
| `ds-r1-qwen-14b` | 0.5 | 0.25 | 0.5 | +0.25 / +7.50 / +4.25 |

Those are the values to BEAT or refute, not to assume.

## Protocol, for reference

Identical in every cell; only eta, seed and the codec flag vary.

```
--GQA_policy avgSM --spec_ret_steps 2 --spec_ret_corr 0.9
--temperature 0.6 --top_p 0.95 --max_gen 16384
--budget 2048 --sink 512 --recent 512 --page_size 32
--page_rep oval --oval_rank 8 [--oval_recon]
```

`--oval_recon` is the codec: selected pages are rebuilt from the records
instead of read from the key cache. Without it the records only drive
selection and the full KV stays resident.
