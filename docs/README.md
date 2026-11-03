# OVAL documentation

OVAL is one method. It keeps a rank-r record of every KV page and uses it in
four complementary ways, each tested on its own before they are combined:

```
 1. retrieval          2. key compression      3. value compression     4. per-head eta
 score pages with   ->  rebuild the selected -> rebuild the selected -> choose eta per
 rank-8 key records     pages' keys from the    pages' values from      (layer, KV head)
                        same records            rank-16 PCA records     with rule 1
                                         \___________________________________/
                                                          |
                                      5. full pipeline: all four together
```

| # | piece | doc | harness flags | state |
|---|---|---|---|---|
| 1 | retrieval | [1-retrieval.md](1-retrieval.md) | `--method spec_ret --page_rep oval --oval_rank 8 --eta E` | done: eta sweeps on 3 models, LongBench v2 / LongGenBench / InfiniteBench, latency, Pareto |
| 2 | key compression | [2-key-compression.md](2-key-compression.md) | `--oval_recon` (`--oval_recon_rank R`) | done: rank 8 = exact keys (k = 8, qwen-7b) |
| 3 | value compression | [3-value-compression.md](3-value-compression.md) | `--oval_vrank R --vbasis pca` | done: rank 16 and 8 = lossless (k = 8, qwen-7b) |
| 4 | per-head eta | [4-per-head-eta.md](4-per-head-eta.md) | `--eta_table <json>` | done at value rank 8 (qwen-7b, llama-8b): rule 1 >= the hand-picked eta |
| 5 | **full pipeline** | [5-full-pipeline.md](5-full-pipeline.md) | all of the above, values rank 16 | **to run** on long reasoning (one agent per model) |
| 6 | long context | [agents/long-context.md](agents/long-context.md) | pieces 2-4 transferred to LongBench v2 | **to design and run**; so far retrieval only (the paper's Table 2) |

**Agents start at [agents/README.md](agents/README.md)**: survey the repo, confirm the
configuration with the user, then run. One brief per agent.

## Supporting docs

| doc | what |
|---|---|
| [harness/freekv-port.md](harness/freekv-port.md) | how OVAL is built into the FreeKV harness |
| [harness/speculative-retrieval.md](harness/speculative-retrieval.md) | FreeKV's spec_ret, used by every reasoning cell |
| [harness/parallel-runs.md](harness/parallel-runs.md) | splitting a sweep across machines by model |
| [pareto-baselines.md](pareto-baselines.md) | the 26-arm matched-budget comparison on MATH50 and the memory definition |

## Where things are

| | |
|---|---|
| `oval/` | the method (`page_basis.py`: records, scoring, value records, per-head eta tables; `eta_rules.py`: rule 1) |
| `patches/freekv.patch` | our changes to the FreeKV harness, applied by `setup.sh` |
| `scripts/` | experiment drivers, mapped to the pieces in [scripts/README.md](../scripts/README.md) |
| `analysis/` | scoring, tables, figures (same map) |
| `results/reasoning/` | every long-reasoning cell, with a guide to the filename tags ([README](../results/reasoning/README.md)) |
| `results/eta_rules/` | rule-1 calibration stats, eta tables, figures, latency |
| `archive/` | finished side studies kept for provenance: [eta rules 2-6](../archive/eta_rules_r2_r6/README.md), the [LOCKS reproduction notes](../archive/locks_reproduction/README.md) |

## Terminology

"OVAL" names the whole method; "OVAL key / value compression" or
"reconstruction" names pieces 2 and 3. There is no separate "codec" method
(an earlier name; any remaining occurrence is a mistake to fix).
