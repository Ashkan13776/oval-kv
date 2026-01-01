# Protocol D — the configuration Phase 2 runs under

    --prompt-model gpt4 --chat-template --gen-extra 24
    seeded-random 50 records, seed 42
    middle-truncation at max_model_len 786432
    max_num_seqs 4, prefix caching off, bf16 KV, enforce_eager

## How it was chosen

Against the paper's **FullKV** row, never against a LOCKS cell. FullKV is dense
attention: budget-independent, no LOCKS machinery, and prompt-sensitive in the
same way. Identifying the protocol on the control leaves the LOCKS row an
independent test.

## What the calibration showed

| protocol | passkey | dialog-qa | code-debug | math-find |
| --- | --- | --- | --- | --- |
| A `chatglm3` (completion) | 100 ✓ | **+26 ✗** | — | — |
| B `gpt4` | — | +8 ✓ (1 artifact) | — | — |
| C `gpt4`+chat | **0 ✗** | +4 ✓ | −8 ✓ | — |
| **D** C + 24 gen tokens | 100 ✓ | +4 ✓ | **+16 ✗** | **+20 ✗** |

Full protocol-D FullKV deltas vs paper: passkey +0.0, dialog-qa +4.0,
book-choice −8.0, code-run −6.0, code-debug +16.0, math-find +20.0.
Four of six inside CI; mean delta +4.3.

## Why we stopped tuning

Three protocol dimensions (template family, chat template, generation budget)
were varied against six control cells whose CIs are ±11–13. Each dimension
moves individual cells by 16–26 points. Continuing (e.g. `gen_extra=8`, which
would plausibly thread passkey and code-debug at once) would be **overfitting
the control row**: with enough free parameters the control can always be
matched, and because LOCKS and FullKV share the protocol, that would erode the
independence that made calibrating on FullKV legitimate in the first place.

## The finding this establishes

**Table 1's absolute cells are not reproducible from the paper's description.**
The unpublished protocol — prompt family, chat template, per-task generation
budget, record seed — moves individual cells by more than the intervals being
tested against. This is a property of the paper's reporting, not of LOCKS.

## What remains testable

1. **The b=512 → b=2048 delta.** Both budgets share subset, harness, prompts,
   truncation and scorer; only `budget_pages` differs. The delta is therefore
   immune to every protocol uncertainty above. Paper: Avg +2.5, Retr +4.7,
   kv-retr +14.0.
2. **Absolute cells, protocol-caveated.** Reported against Table 1 with the
   difference-CI test and this note attached.

Not testable in this scope (user scoped to the LOCKS row): the paper's central
claim that LOCKS matches FullKV at ~2% density. That needs a full FullKV row
under protocol D. Six FullKV cells exist in `results/calib/` and are kept.

## Known protocol-D artifacts to disclose with any number

* `math_find` +20.0 and `code_debug` +16.0 vs the paper's FullKV — both are
  "more room to reason" tasks; generations are clean, so this is protocol, not
  a scoring artifact.
* `code_run` is still budget-clipped at 29 tokens (the model starts reasoning
  and is cut off). It sits at floor (0–6) for every method in the paper, so it
  carries little signal either way.
* Scoring artifacts from the earlier protocol are GONE: false positives are 0
  on every calibration cell, and official == strict scoring throughout.
