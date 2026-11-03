# FreeKV (ICLR 2026) — port to Blackwell, and the output-aware page basis

Paper: *FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference*,
arXiv 2505.13109. Code: https://github.com/sjtu-zhao-lab/FreeKV

Goal: run **our** output-aware page basis inside **their** harness, so the
comparison to their published Table 3 is apples-to-apples. We do not re-run
their baselines; their columns stand as the reference.

## Table 3 — what it is

"Accuracy results of long reasoning tasks":

| | |
| --- | --- |
| datasets | AIME24 (30 problems), MATH500→50, GPQA→50 |
| metrics | pass@k and avg@k, **k=8 seeds** (42–49) |
| models | DeepSeek-R1-Distill Llama-8B / Qwen-7B / Qwen-14B |
| budget | B=2048 for all retrieval methods |
| sampling | temperature 0.6, top_p 0.95, max_gen 16384 |
| method | `--method spec_ret --page_rep quest --GQA_policy avgSM --spec_ret_steps 2 --spec_ret_corr 0.9` |

**Dataset naming.** The paper's columns say "MATH500" and "GPQA", but the text
says *"we select 50 problems each from MATH500 and GPQA"*. The artifact ships
`math50.jsonl` and `gpqa50c.jsonl` (50 records each) and **no** `math500.jsonl`
— `--dataset MATH500` points at a missing file and crashes. Confirmed three
ways: their text; every published MATH500/GPQA pass@k is an exact multiple of
2.0 = 1/50 (full MATH500 would give 0.2 granularity); and `math50.jsonl` carries
MATH500's own schema (`problem`, `solution`, `answer`, `subject`, `level`,
`unique_id`). Correct flags are `--dataset MATH50` and `--dataset GPQA50c`.

## Building on sm_120 (RTX PRO 6000 Blackwell)

Their pins all predate Blackwell: `torch==2.5.1`, `flash-attn==2.6.3`,
`flashinfer-python==0.2.4`, and vendored submodules cutlass (2024-04),
flashinfer (2024-06), raft (2024-04). Despite that it builds, because their own
12 `.cu` files contain **no** `wgmma`, TMA, `mbarrier`, `mma.sync` or `sm_90`
guards, and their CMake uses `CUDA_ARCHITECTURES=native` so Hopper-specific
templates are never instantiated.

What was needed:

| problem | fix |
| --- | --- |
| torch 2.5.1 has no sm_120 kernels | torch 2.11+cu128, later aligned to **2.8.0+cu128** (see flash-attn) |
| CUDA 13.0 too new for 2024-era CUTLASS | build with **CUDA 12.8** from the `pyg310` env — first toolkit with sm_120, still compiles CUTLASS 3.5 |
| conda's `targets/x86_64-linux/` layout invisible to CMake's FindCUDAToolkit | symlink shim at `vendor/cuda128/` with standard `bin`/`include`/`lib64`/`targets` |
| CMake picked up Python **3.14** headers from an unrelated env | pin all five `Python*_EXECUTABLE`/`_ROOT_DIR`/`_INCLUDE_DIR` vars |
| `raft::IOType<nv_bfloat16>` compile errors | their README's **step-3 patches** (flashinfer GQA group sizes 5/7, raft bf16 vectorized IO) — I skipped these on the first attempt; that was the cause |
| no prebuilt flash-attn for torch 2.11 | align torch to **2.8.0+cu128** and use the prebuilt `flash_attn 2.8.3.post1 cu12torch2.8 cxx11abiTRUE cp310` wheel |

Result: 14/14 objects compile with `-gencode arch=compute_120,code=sm_120`,
zero errors; `estimate_scores` launches and returns finite, varying scores.

Scripts: `env/build_freekv.sh`, `env/fk_configure.sh`, `env/fk_compile.sh`.

### The CUDA extension is NOT used by Table 3

`accuracy/` — the harness that produces Table 3 — imports only torch,
transformers and datasets. It never imports `freekv_cpp`. So the build above is
not on the critical path for accuracy; it only matters for their latency path
(`source/pred.py`).

**This matters because their speculative-retrieval CUDA path is broken on this
hardware.** Isolated by ablation on `source/pred.py`:

```
plain     (cuda_cpy alone)        CLEAN
corronly  (+ correction)          CLEAN
speconly  (+ speculative retr.)   TOTAL CORRUPTION (CJK chars, zero-runs)
both      (+ spec + correction)   drifts, semi-coherent
```

Same config run twice gives *different* corruption, and batch-1 vs batch-1 is
bit-reproducible — so it is a **race**, not fp noise. `--thread_pool 1` does not
fix it. Worth reporting upstream. It does not affect our results: `accuracy/`
implements `spec_ret` in pure PyTorch, so we run their published command
unmodified.

## Batching is unsafe for accuracy (theirs too)

`--batch_size > 1` changes results. Verified with **their own** `page_rep=quest`,
greedy, fixed seed:

```
bs=1 vs bs=1 rerun : 4/4 identical   <- harness is deterministic
bs=1 vs bs=4       : 0/4 identical   <- batching changes outputs
```

Not caused by our patch. Their Table 3 must be batch-1. We run batch 1.

Running two *processes* concurrently is safe by contrast — separate CUDA
contexts doing identical computation — but buys little: measured on generated
tokens (which normalises for problem difficulty), 41.9 → 47.1 tok/s = **1.12×**.
The bottleneck is the per-decode-step Python page selection in their pure-torch
dynamic attention, not GPU compute (SM utilisation was already 70% at one
worker).

## The patch: `--page_rep locks --eta E`

85 lines added, 7 removed, 4 files. Their `quest`/`arkv` paths are untouched
(added as an `elif`, not an edit).

| file | change |
| --- | --- |
| `kvc/patch/locks_rep.py` | **new**: `build_page_summary`, `locks_sel`, `outproj_metric_sqrt`, `gqa_combine` |
| `kvc/patch/dynamic_attention.py` | `page_rep == "locks"` at both page-build sites; `_page_score` dispatcher; buffer growth |
| `kvc/patch/llama.py` | allocate `mu_k`/`basis_k`/`coef_k`; compute `G^{1/2}` from the already-reordered `o_proj` |
| `eval/util.py` | `--page_rep locks`, `--eta`, `--locks_rank`; **eta in the output path** |

**What is replaced.** Their page summary is two vectors per page — elementwise
min/max (`quest`) or a mean-absolute-deviation band around the midpoint
(`arkv`) — scored by the Quest bound `sum_d max(q_d·min_d, q_d·max_d)`. Ours is
a rank-8 basis of the page's keys, scored by `logsumexp_t (R_t·(Bᵀq) + mu·q)`.
Basis is the leading eigenvectors of

    M_eta = (1-eta)·XᵀX/tr(XᵀX) + eta·XᵀYGYᵀX/tr(XᵀYGYᵀX)

with X = centred keys, Y = centred values, G = WᵀW from `o_proj` summed over
each KV head's GQA group. eta=0 is plain key PCA; eta=1 is output-only.

**What is kept.** Everything downstream of the per-head page score: their
`avgSM` group-consistent combine, top-k, speculative q-cache, and correction.

### Verification (`harness/test_locks_rep.py`)

```
1. eta=0 vs plain key PCA subspace   : 4.98e-06
2. |B^T B - I|                       : 1.33e-06
3. |R - X B|                         : 5.15e-06
4. eta=0.5 Gram route vs d x d M_eta : 2.15e-06     <- the important one
5. G_0 vs manual group sum           : 2.91e-07
6. rank=page-1 scorer vs exact lse   : 7.79e-05
7. rank clamped 32 -> 31             : OK
```

Check 4 builds `M_eta` independently as a full 128×128 matrix and confirms the
cheap 32×32 Gram route spans the same subspace.

Two guards against silent corruption: the rank clamp (centring costs one dof, so
`rank == page_size` divides by a ~zero eigenvalue and returns garbage), and
normalising `A` to unit mean diagonal (the omission that produced the bogus
−18.0 result in the LOCKS work).

## Harness bugs found and worked around

- **`eval.py` scores incomplete cells.** On a live sweep it averaged a
  5-problem seed with a 30-problem seed and reported `pass@k = 320.0` (it
  divides by `n_problem[0]`). `harness/score_freekv.py` counts complete cells
  only.
- **`corrects` is 1-based** (`corrects.append(pid+1)`) — comparing it to a
  0-based index inflated our loose column to a nonsense 96% on MATH50.
- **No resume, results are appended** — a restart duplicated records into the
  avg@k denominator. Added resume to `pred.py`; it advances `data_from` past the
  completed prefix rather than reindexing (qid = enumerate index + data_from, so
  `select()` would renumber and collide).
- **eta was absent from the output path** — all four η arms would have written
  to one file and overwritten each other.
- **`eval/o1/datasets` path is stale** (dir renamed to `eval/reasoning`);
  symlinked rather than editing their source.
- **`--method full` crashes** (`past_key_values[0][0]` on None) — unrelated to
  us, but it means we cannot produce a same-engine dense reference here.

## Task quality caveats

- **`gpqa50c` is degenerate: all 50 answers are `'A'`.** Options were never
  shuffled, so the metric partly measures whether the model emits "A".
- **GPQA has a large format gap** (strict 26 vs loose 32 at eta=0): ~19/50
  generations answer in prose ("**Answer: A**") and score zero despite being
  right, because the scorer requires `\boxed{}` after `**Final Answer**`.
- **On GPQA the paper's own FreeKV (39.50) and Quest (38.75) beat FullKV
  (35.75)** — when sparse beats dense, the task is not measuring selection
  quality.
- **AIME24 saturates the generation cap**: 6–10 of 30 generations hit
  max_gen=16384 and score wrong regardless of KV quality.

MATH50 is the cleanest instrument: 50 problems, no format loss, 0–2 cap hits.

## Results so far — ds-r1-qwen-7b, k=1 (seed 42), B=2048

```
task            eta=0.0     eta=0.25      eta=0.5     eta=0.75     sd  range   peak   paper
AIME24      53.33/53     53.33/53     60.00/60     50.00/50      3.63  10.00    0.5   52.92
MATH50      70.00/70     74.00/74     70.00/70     72.00/72      1.66   4.00   0.25   70.00
GPQA50c     26.00/32     40.00/50     34.00/42     38.00/42      5.36  14.00   0.25   39.50
                                                                          (strict/loose)
```

**Port validation.** eta=0 (rank-8 key PCA — not their representation) lands on
their published FreeKV avg@8: MATH50 70.00 vs 70.00 exactly, AIME24 53.33 vs
52.92.

**The eta effect is a null on the trustworthy tasks.** Averaged over eta>0 the
change is +1.11 (AIME24) and +2.00 (MATH50) — both smaller than the scatter
across eta on the same task (sd 3.63, 1.66) — and the two tasks peak at
*different* eta (0.5 vs 0.25), which a genuine optimum cannot do.

GPQA's +11.33 rests on its eta=0 cell being an outlier low (26.00 against 34–40
for every eta>0), on the degenerate task with the largest format gap.

Per-problem traces show churn rather than accumulation: on AIME24 problem 11 is
lost at 0.25, recovered at 0.5, lost again at 0.75; on MATH50 problems 26 and 43
are gained at 0.25 and lost again at 0.5. A basis that genuinely selects better
pages should mostly *add* solved problems as it improves.

**Retracted:** an earlier note reported a pooled +4.62 with sign-test p=0.105 as
"suggestive". That compared eta=0 against eta=0.5 only — and eta=0.5 is AIME24's
peak. Selecting the maximum of a scatter and then testing it manufactures
significance; the full curves supersede it.

This matches the LOCKS/InfiniteBench result: across eta ∈ [0,1], on two
independent codebases, two model families and two benchmark suites, the
output-aware term does not beat plain rank-8 key PCA.

## Upstream bug: tuple KV cache never written in bf16 (`--method full`)

`kvc/patch/tuple_kv_cache.py` updated `past_key_value` only inside the
`if self.kv8:` branch. With `kv8=False` -- the default, and what the FullKV
baseline runs -- the attention returned the cache it was handed, unchanged.

Consequence: prefill is correct, so the first generated token is right; from
step 2 the model sees a tuple of `None`s, treats `past_key_values_length` as 0,
and re-RoPEs each new token at position 0. Output collapses into a repeated
token (U+200D on Llama-3.1-8B).

This is silent. On InfiniteBench it reads as FullKV = 9.39, passing only the
tasks graded on their first emitted token (longbook_choice 46.0, math_find
26.0, code_debug 16.0 -- byte-identical to every sparse arm) and scoring ~0 on
everything generative, including passkey.

Repro (20 s): greedy-decode 30 tokens from a 2K passkey prompt with
`--method full`. Cached path returns `' <|start_header_id|>‍‍...'`;
re-forwarding the whole prefix each step returns `' 91746. The passkey is...'`.
Not RoPE -- `INPLACE_ROPE_OFF=1` reproduces byte-identically.

Fix: add the missing `else` writing `(key_states, value_states)`, which are
already the concatenated full sequence at that point.

Scope: `--method full` only. The sparse methods (spec_ret / arkv / raas) use
the dynamic-attention cache in `kvc/patch/llama.py` and are unaffected -- they
score 100.0 on passkey, which the broken FullKV path cannot.

### Retraction this forces

The kv_retrieval "dead task" call was based on FullKV scoring 0.0 there with
degenerate output. That degenerate output was THIS bug, not a property of the
task or the model. The exclusion of kv-retr from the 10-task mean is
unsupported until the fixed FullKV run lands.
