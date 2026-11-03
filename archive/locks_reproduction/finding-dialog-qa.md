# dialog-qa: our 54.0 vs paper 38.0 is a scoring artifact, not selection quality

## The number

    dialog-qa  b=2048   ours 54.0   paper 38.0 ±13.5   delta +16.0

I initially passed this as "consistent" on the difference-CI (±19.4). That test
is technically satisfied but badly underpowered — a proper two-proportion test
gives z=1.605, **p=0.108**. At n=100/arm the same 16-point gap would give
p=0.023. n=50 simply cannot resolve it, so "consistent" was not evidence of
agreement, and I over-relied on it.

## Root cause

The official scorer (compute_scores.py:244) is a bare substring test:

    def get_score_one_longdialogue_qa_eng(pred, label, model_name):
        pred = pred.strip().upper()
        for item in label:
            if item.upper() in pred:
                return 1
        return 0

Any generation that *mentions* the right name anywhere in its 40 tokens scores
1, regardless of what it actually answered.

Our prompt makes that likely. `chatglm3` maps to `yarn_mistral_templates`
(eval_utils.py:63), whose longdialogue template (prompt.py:29) is
**completion-style**, ending:

    "The name that has been replaced with $$MASK$$ is likely"

The model continues in prose and name-drops other characters. The gpt4 and kimi
templates (prompt.py:14, :44) instead end with "Just say the name used by the
scriptwriter ... and nothing else."

## Measurement

Rescoring against the model's actual answer span (leading quoted name, else
text before the first period/newline) instead of the whole generation:

| scoring | value |
| --- | --- |
| official — substring anywhere | **54.0** (what we report) |
| strict — name in answer span | **38.0** |
| paper | **38.0 ±13.5** |

**8 of our 27 "correct" answers are false positives.** Examples:

| ground truth | model actually answered |
| --- | --- |
| HAMMOND | Alex |
| MICKEY | Gail |
| CLAUDIA | Jim Kurring |
| HILDY | Prissy Bensinger |
| SHERMAN | Maria Ruskin |
| BLOOM | Max Bialystock |

Strict scoring reproduces the paper's cell exactly. So LOCKS' *page selection*
on this task matches the paper; only the answer formatting differs.

## Consequence for Avg — must be disclosed

dialog-qa is 1 of 11 tasks in Avg, so a +16.0 inflation on this cell lifts our
Avg by **+1.45**. The paper's Avg CI half-width is ±2.6. A single scoring
artifact therefore accounts for over half the interval we are testing against.
Reporting Avg without this caveat would overstate agreement.

## Not a bug in our harness

The reference `eval_chatglm.py::get_pred` returns raw generation text with no
post-processing, and we followed the reference impl's own model→template
mapping. This is a protocol-sensitivity issue inherent to InfiniteBench's
lenient scorer, not something we implemented wrongly. The paper presumably used
a terser prompt (or a chat template) and never hit it.

## What to do

Report the **official** score as the headline number (it is what the paper's
`score` metric means, so it is the comparable figure), with the strict score
alongside and this note attached. Do NOT silently substitute strict scoring —
that would be changing the metric after seeing the result.

## Watch list

Other tasks whose scorers could reward verbosity, to check as they land:

* `longbook_choice_eng`, `code_debug` — multiple-choice parsing; a rambling
  answer mentioning several options may be scored on the first letter found.
* `kv-retr` — word-membership, but the target is a UUID, so incidental mention
  is implausible (and it already agrees: 30.0 vs 28.0).
* `passkey`, `num-str`, `math-find` — `first_int_match`, which takes the FIRST
  integer, so verbosity does not help.
* `book-qa-en`, `book-qa-zh` — F1, where rambling *lowers* precision, so the
  bias runs the other way.
