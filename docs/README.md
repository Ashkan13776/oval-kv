# Documentation

Notes that go deeper than the paper, by topic.

| doc | paper section | what |
|---|---|---|
| [retrieval.md](retrieval.md) | Sec. 3.1 | scoring pages with the rank-8 key basis (`--page_rep oval --oval_rank 8 --eta E`) |
| [key-reconstruction.md](key-reconstruction.md) | Sec. 3.2 | rebuilding the selected pages' keys from the same basis (`--oval_recon`) |
| [value-basis.md](value-basis.md) | Sec. 3.3 | rank-16 PCA value basis (`--oval_vrank 16 --vbasis pca`) |
| [eta-rule.md](eta-rule.md) | Sec. 3.4 | the closed-form per-head eta rule and its calibration (`--eta_table`) |
| [long-reasoning.md](long-reasoning.md) | Sec. 4.2 | the full pipeline on long reasoning, step by step |
| [long-context.md](long-context.md) | Sec. 4.3 | the full pipeline on LongBench v2 and LongGenBench |
| [accuracy-vs-memory.md](accuracy-vs-memory.md) | Fig. 2a, App. | the matched-budget comparison and the memory model |
| [value-codec.md](value-codec.md) | App. | bytes of the value basis |
| [harness/freekv-port.md](harness/freekv-port.md) | | how OVAL is built into the FreeKV harness |
| [harness/speculative-retrieval.md](harness/speculative-retrieval.md) | | FreeKV's speculative retrieval, used by every reasoning cell |
| [harness/parallel-runs.md](harness/parallel-runs.md) | | splitting a sweep across machines |

Video and robot experiments are documented in their drivers' docstrings
([`experiments/video/`](../experiments/video), [`experiments/robot/`](../experiments/robot))
and in [`results/video/REPORT_C.md`](../results/video/REPORT_C.md).
