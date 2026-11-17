# Results

Raw outputs behind the paper. Every scorer reads these files directly, so the
tables are always recomputed from the per-question predictions.

| folder | paper | what |
|---|---|---|
| [`reasoning/`](reasoning/README.md) | Tab. 1, Tab. 4, Fig. 2a | every long-reasoning cell, with a guide to the filename tags |
| `long_context/` | Tab. 2 | LongBench v2 / LongGenBench predictions and per-head eta tables |
| `eta_rules/` | Sec. 3.4, App. | rule-1 calibration statistics, per-head eta tables (`vr16/` is the paper setting), figures, latency |
| `video/` | Tab. 3, Fig. 1b, Fig. 3, App. | Video-MME / LVBench / LIBERO runs (`runs/`), Watch-and-Do (`vla/`), tables, figures, LaTeX; `llava_runs/` and `llava_analysis/` hold the LLaVA-Video Video-MME runs and selection analyses; [REPORT_C.md](video/REPORT_C.md) |
| `ablations/` | Tab. 4, App. | rule variants (A3), disjoint calibration (A1, A2), compression-only arm (B), equal-bytes study; [REPORT.md](ablations/REPORT.md) |
| `latency/` | Fig. 2b,c | unedited stdout of the decode-latency runs (`fig6/`, `lb2/`) |
| `figures/` | | figures produced by `analysis/` |
| `paper_tables.md`, `all_tables.txt` | | earlier retrieval-only tables (exact keys and values) |

The latency log bodies predate the method's current name, so their banners
read `[locks] page_rep=locks ...`, the old spelling of `--page_rep oval`. The
logs are kept byte-for-byte as emitted.
