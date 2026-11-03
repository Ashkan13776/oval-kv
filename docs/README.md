# Docs index

**Terminology.** There is one method, OVAL: retrieval with the page records
and, optionally, rebuilding the selected keys and values from them. There is
no separate "codec" method. Some paths still say `codec` (`results/codec/`,
`scripts/codec_*.sh`, `analysis/*codec*.py`, `docs/value-codec.md`) only
because result files and running scripts depend on those names; write
"OVAL" (or "OVAL key / value compression") in anything new.

## Current

| doc | what |
|---|---|
| [full-pipeline.md](full-pipeline.md) | **the next run**: OVAL retrieval + key compression + value compression + per-head eta, step-by-step instructions and assignments |
| [eta-rules.md](eta-rules.md) | rule 1: per-(layer, KV head) eta in closed form, calibration, results at value rank 8, latency, figures |
| [value-codec.md](value-codec.md) | value compression by per-page PCA, the rank sweep |
| [speculative-retrieval.md](speculative-retrieval.md) | FreeKV's spec_ret mechanism that every reasoning cell uses |
| [freekv-port.md](freekv-port.md) | how OVAL is built into the FreeKV harness |
| [parallel-runs.md](parallel-runs.md) | splitting a sweep across machines by model (the per-model rules still apply; its status table is historical) |
| [pareto-baselines.md](pareto-baselines.md) | the 26-arm Pareto comparison on MATH50 and the memory definition |

Result summaries live next to the data: `results/codec/*.md`,
`results/eta_rules/table.md`, `results/paper_tables.md`.

## Archive

`archive/` holds the notes of the earlier LOCKS / GLM-4-9B InfiniteBench
reproduction (protocol decisions, deviations, validation, subset findings).
They are kept for provenance; nothing in the current pipeline depends on them.
