# Harness validation

Checks run before the GPU results landed, so a defect in our code could not be
mistaken for a failure to reproduce.

## 1. Plugin is live and configured as Table 1 specifies

    [locks] ACTIVE variant=fast ... score=r8i4
    budget_abs=128  b_fix=[128]  n_pages=1376  page=16  sink=1  window=1
    [locks] SPARSE ACTIVATION 280/280 decode rows = 100.00%  [OK]

128 of 1376 pages attended (9.3%), page size 16, sink+window inside b — matching
appC_extres.tex:55-66. The activation gate is the plugin's own implementation of
the check the paper's harness enforced (appC_extres.tex:60).

Verified via `harness/check_budget.py`. Needed because the plugin's banner
prints `cfg.budget`/`cfg.coverage` but never `budget_pages`, so an unapplied
fixed-B budget would otherwise be silent.

## 2. All 11 official scorers run and discriminate

`harness/check_scorers.py` — every task scores 1.000 on echoed ground truth and
~0 on garbage. Covers the fragile paths: `longbook_sum_eng` (HF rouge metric),
`longbook_qa_chn` (character-level F1), `code_debug` / `longbook_choice_eng`
(multiple-choice parsing out of free text).

## 3. Bootstrap reproduces the paper's own CIs

Our `bootstrap_group` implements the paper's stated procedure: resample records
within each task, recompute the task-averaged capability score
(tab_infinitebench_main.tex:10-11).

| case | ours | reference |
| --- | --- | --- |
| 50 x correct | 0.0 | paper `passkey` ±0.0 |
| 25/25 split | 14.0 | analytic 1.96*sqrt(.25/50)*100 = 13.9 |
| 14/50 correct | 13.0 | paper `kv-retr` b=2048 ±12.6 |
| Retr group = (100.0, 100.0, 28.0) | **4.0** | paper `Retr` b=2048 **±4.1** |

The last row is the meaningful one: feeding the paper's exact per-task
composition through our bootstrap returns their published group CI. Our CI
widths are therefore comparable to theirs, which is what every "within CI"
verdict depends on.

## 4. End-to-end pipeline

`passkey` at b=2048: 50/50 correct, 0 truncated, scored **100.0** against the
paper's **100.0 ±0.0**. Confirms prompts, middle-truncation, greedy generation,
sparse selection, dump format and scoring all work together.

Caveat: `passkey` is the easiest cell in the table (100.0 for every method at
every budget, including Quest), so it validates plumbing, not the method.
