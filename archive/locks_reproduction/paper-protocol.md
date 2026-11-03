# Protocol as specified by the paper

Extracted from the arXiv LaTeX source (`vendor/paper/src/`), not the HTML
rendering. Citations are file:line so every claim is checkable.

## Benchmark configuration

| item | value | source |
| --- | --- | --- |
| Model | GLM-4-9B-Chat-1M (`zai-org/glm-4-9b-chat-1m`) | tab_infinitebench_main.tex:2 |
| Benchmark | InfiniteBench, 100K+ context | appC_extres.tex:11 |
| Tasks | 11 (all 12 minus `math_calc`) | appC_extres.tex:12 |
| Why math_calc excluded | "a prompting artifact plus 30K-token generation the baselines cannot serve fairly" | tab_appC_benchmarks.tex:408 |
| Records | seeded subset of 50 per task | tab_infinitebench_main.tex:3 |
| Metric | `score` | tab_infinitebench_main.tex:3 |
| CI | 95% bootstrap over the 50 records, resampled within task | tab_infinitebench_main.tex:10-11 |
| `max_num_seqs` | 4 (all arms except KVzip) | appC_extres.tex:52 |

## Budget semantics (this is the axis we must match exactly)

- b is tokens per **(layer, KV-head)**, page size **B=16**. appC_extres.tex:55
- Page budget k = ceil(b/B) → **b=512 → 32 pages**, **b=2048 → 128 pages**.
- The always-kept **sink** and **most-recent** pages fill **2 of the k slots,
  counted _within_ b, never added on top**. appC_extres.tex:64-66
- LOCKS selects **per KV head** (no GQA union, unlike Quest). appC_extres.tex:31
- Deployed LOCKS = **rank-8 int-4** summary, share-average ("nrm") group
  combine. appC_extres.tex:16, tab_appC_benchmarks.tex caption

## Software stack (appF_repro.tex:13-18)

- vLLM **0.24.0**
- PyTorch **2.11.0** (cu130), CUDA **13.0** runtime, cuBLAS 13.0.0, driver 13.2
- KV cache dtype **bfloat16**
- **Full CUDA graphs**
- **Prefix caching OFF**
- Container: **NGC PyTorch `25.08-py3`** (Singularity)

## Hardware (appF_repro.tex:20-24)

- 1x NVIDIA **H200 NVL** (143,771 MiB HBM3e), sm_90.
- Note: the paper scopes this line to "efficiency cells". It does not state
  separate hardware for accuracy cells, so we assume the same node.

## Validity gates the paper's own harness enforced

These are useful as our own smoke tests even though the harness is unreleased:

- `benchmarks/run.py` "refuses to record a sparse cell whose measured
  sparse-decode activation is not 1.00 on every decode row". appC_extres.tex:60-61
- Runs must carry the `[locks] ACTIVE` and `score=r8i4` banners. appC_extres.tex:148

## Internal-consistency check (done, passes)

Table 1's groups are derivable from the per-task grid, confirming the groupings:

LOCKS @ b=2048:
- Retr    = mean(100.0, 100.0, 28.0)                      = 76.0  ✓
- LongQA  = mean(38.0, 27.4, 82.0, 18.0, 17.8)            = 36.64 → 36.6 ✓
- Avg(11) = 479.2 / 11                                    = 43.56 → 43.6 ✓

LOCKS @ b=512:
- Retr    = mean(100.0, 100.0, 14.0)                      = 71.33 → 71.3 ✓
- LongQA  = mean(26.0, 23.3, 82.0, 19.0, 17.9)            = 33.64 → 33.6 ✓
- Avg(11) = 452.2 / 11                                    = 41.11 → 41.1 ✓

So "LongQA" = dialog-qa + book-sum + book-choice + book-qa-en + book-qa-zh
(5 tasks), and Avg is the unweighted mean over all 11 tasks — NOT a
record-weighted mean.
