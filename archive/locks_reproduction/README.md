# LOCKS reproduction notes

From the project's first phase: reproducing the LOCKS paper's InfiniteBench
results on GLM-4-9B-Chat-1M before OVAL replaced its page summary. Kept for
provenance.

| file | |
|---|---|
| `paper-protocol.md` | the protocol as the paper specifies it |
| `open-questions.md`, `decisions.md` | parameters the paper leaves open, and how they were settled |
| `deviations.md` | forced deviations (hardware, lengths) |
| `protocol-per-task.md`, `protocol-D.md` | the per-task protocol calibration and the final protocol |
| `validation.md` | harness checks run before the GPU results |
| `length-stats.md` | token lengths of the task subsets |
| `figure2-protocol.md` | Figure 2 (retrieval and QA vs budget) protocol |
| `finding-kv-retr.md`, `finding-dialog-qa.md` | two discrepancies with the paper and their causes; the seeded-subset rule from `finding-kv-retr.md` is still used by `analysis/subset.py` |
| `verify_locks_math.py` | check that the shipped LOCKS summary implements the paper's construction |
