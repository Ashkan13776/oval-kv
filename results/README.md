# Results

| folder / file | piece | what |
|---|---|---|
| [`reasoning/`](reasoning/README.md) | 1-5 | every long-reasoning cell (MATH50, AIME24, GPQA50c; R1-Qwen-7B, R1-Llama-8B, R1-Qwen-14B), with a guide to the filename tags and per-experiment summaries |
| [`eta_rules/`](../docs/4-per-head-eta.md) | 4 | rule 1: calibration stats, per-head eta tables (`tables/`, values rank 8; `vr16/` for the full pipeline), `table.md`, figures, latency |
| `fig6_latency/`, `lb2_latency/` | 1 | unedited stdout of the decode-latency runs |
| `figures/` | 1 | the Pareto figure (`pareto_math50_b2048.*`) |
| `paper_tables.md`, `all_tables.txt` | 1 | the numbers as reported in the paper; every completed run scored from disk (`analysis/all_tables.py`) |

Rules 2-6 for per-head eta are in `archive/eta_rules_r2_r6/`.

The latency log **bodies** were produced before the method was renamed to
OVAL, so their banners read `[locks] page_rep=locks ...`. That is the old
spelling of today's `--page_rep oval`; the logs are left byte-for-byte as
emitted rather than rewritten after the fact. File names use the current arm
names (`oval0`, `oval50`, `oval100` for eta = 0, 0.5, 1.0).
