# Raw measurement artifacts

`fig6_latency/` and `lb2_latency/` hold the unedited stdout of the latency
runs; `paper_tables.md` holds the numbers as reported in the paper.

The log **bodies** were produced before the method was renamed to OVAL, so
their banners read `[locks] page_rep=locks ...`. That is the old spelling of
today's `--page_rep oval`; the logs are left byte-for-byte as emitted rather
than rewritten after the fact. File names use the current arm names
(`oval0`, `oval50`, `oval100` for eta = 0, 0.5, 1.0).
