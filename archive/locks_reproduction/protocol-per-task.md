# Per-task protocol calibration — scope, method, and what it costs

User directive (2026-08-14): get as close to the paper's numbers as possible,
calibrating whatever is needed. This note records exactly what was calibrated,
how each choice was made, and what that does to the interpretation.

## Two dimensions, two very different criteria

**1. Generation budget — per task, PAPER-INDEPENDENT.**
Rule: the smallest tested budget at which the model's answer COMPLETES, read
off the generations alone. Never chosen by proximity to the paper.

This rule is falsifiable and it cut against us: `code_debug` at the reference
budget emitted 3 distinct strings across 50 records, only 1 containing an
option letter, yet scored 24.0 (−8.0, comfortably inside CI) on 12 spurious
credits. The rule rejects that arm in favour of gen+24, which scores 48.0
(+16.0, outside CI). We took the worse-agreeing arm because it is the only one
that measures anything.

Justification for doing this per task: InfiniteBench already defines
`max_new_tokens` per task. Adjusting it for a chat model's preamble corrects an
existing per-task parameter rather than inventing one.

**2. Prompt family — per task, NOT paper-independent.**
There is no mechanical property that makes gpt4 templates "correct" for one
task and yarn_mistral "correct" for another. The only available criterion is
proximity to the paper's FullKV cell. This is fitting to the control row.

Justification is weaker on two counts: `MODEL_TO_PROMPT_TEMPLATE` maps a MODEL
to ONE family (per-task selection is a construct neither the benchmark nor the
paper has), and with 2 options x 11 tasks against ±11–13 CIs, some apparent
agreement will be selection noise.

## What this does to the claim

Calibrating on FullKV keeps the LOCKS row from being tuned *directly* — no
LOCKS cell is ever consulted. But LOCKS and FullKV share the protocol, so
fitting the control does propagate some flattery to LOCKS. The honest framing
of the final table is therefore:

> "Closest agreement achievable by calibrating the unpublished protocol
>  per task against the paper's dense control row"

NOT "LOCKS reproduces". The concern was raised and the user reaffirmed the
directive; proceeding as instructed, with the interpretation recorded here.

## What stays clean regardless

1. **b=512 → b=2048 delta.** Both budgets run under the identical per-task map,
   so every protocol choice cancels. Paper: Avg +2.5, Retr +4.7, kv-retr +14.0.
   This remains a genuine test of LOCKS' budget→quality curve.
2. **LOCKS vs our own FullKV**, compared to LOCKS vs their FullKV. The paper's
   actual claim (LOCKS ≈ FullKV at ~2% density) with the protocol offset
   cancelling on both sides.

## Reporting requirement

For every task, report: which arm was chosen, both arms' FullKV values, the
paper's FullKV value, and the margin by which the arm was selected. A cell
chosen by a 2-point margin between two arms with ±13 CIs is a coin flip and
must be visible as such.

## Runs

* D row (gpt4 + chat, per-task gen budget): 10/11 done, `qa_chn` in flight.
* A row (chatglm3/yarn_mistral, reference budgets): `dialog-qa` done (60.0 vs
  paper 34.0); other 10 queued, ~13h.
* Then: per-task map, then LOCKS at both budgets (~28h).
