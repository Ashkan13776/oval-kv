# Speculative retrieval: what FreeKV's `spec_ret` actually does

This is a **latency** mechanism, not a scoring rule. It is part of the harness
we build on, not part of OVAL, and it is the reason there is more than one
OVAL arm in the results. Anyone reading our numbers needs to know which arms
carry it.

## The problem it solves

A retrieval step is a dependency chain. Nothing downstream can start until
each link finishes:

```
compute q_t  ->  score every page against q_t  ->  fetch the winning pages' KV  ->  attend
```

The scoring and the fetch sit on the critical path of every decoded token. On
a long context the fetch in particular is a large, latency-bound read.

## The trick

Score the pages with the **previous** step's query, `q_{t-1}`, which already
exists. Selection and the KV fetch for step *t* can then be issued *before*
`q_t` has been computed, overlapping with the step *t-1* compute instead of
queueing behind it.

The bet: consecutive decode queries point in nearly the same direction, so the
pages that mattered for `q_{t-1}` are still the pages that matter for `q_t`.

## The correction, and when the bet is called off

The bet is checked per KV head rather than assumed
(`accuracy/kvc/patch/dynamic_attention.py`, the `probe_q` block):

```python
past_q = self.q_cache[:, self.q_ptr, ...]        # last step's query
sim    = F.cosine_similarity(q, past_q, dim=-1)  # how far did the query move?
kv_head_corr = kv_head_sim < self.correct_sim    # threshold, 0.9 in our runs
probe_q = torch.where(corr, q, past_q)           # re-score ONLY the moved heads
```

- `cos(q_t, q_{t-1}) >= 0.9` — the query barely moved, keep the speculative
  selection, the latency stays hidden.
- `cos(q_t, q_{t-1}) < 0.9` — that head's query rotated away, so it is
  **corrected**: re-scored against the real `q_t`, paying back the latency the
  speculation was trying to avoid.

Correction is per KV head, so a step can be partly speculative. `corr_group`
(`max` / `avg`) sets how the per-query-head similarities are pooled to a
decision for the KV head they share.

Flags used throughout our runs: `--spec_ret_steps 2 --spec_ret_corr 0.9`.
`spec_ret_steps` is the depth of the rotating query cache; `spec_ret_corr` is
the cosine threshold above.

## Why it leaks into accuracy, and what that means for our tables

It is sold as a speed optimisation, and on a *correct* speculation it changes
nothing. But when the bet is taken and the query has moved somewhat — just not
past 0.9 — the pages selected are the ones `q_{t-1}` wanted, not the ones
`q_t` wanted. The attended set differs, so the output can differ. Speculative
retrieval is therefore **not accuracy-neutral**, and arms that carry it are
not directly comparable with arms that do not.

Our arms split on exactly this:

| arm | digest | selection query | in the figure |
| --- | --- | --- | --- |
| OVAL Retv.+Comp. | rank-8 basis | speculative (`q_{t-1}`, corrected) | yes |
| OVAL Retv. | rank-8 basis | speculative (`q_{t-1}`, corrected) | yes |
| OVAL Retv. no-spec. | rank-8 basis | current `q_t`, always | no -- see below |

FreeKV and ArkVale also carry it, because speculative retrieval *is* what
FreeKV is; Quest and every scorer we added (Loki, SparQ, InfLLM, Centroid,
ShadowKV, RocketKV, 4-bit keys, H2O, TOVA, SnapKV, Expected Attention,
K-norm, Random, the oracles) score with the current query, as their papers do.

So there are two matched comparisons, and **they disagree**:

| comparison | matched? | result |
| --- | --- | --- |
| OVAL Retv. 71.75 vs FreeKV 71.25 | both speculative | **+0.50** |
| OVAL Retv. no-spec. 69.75 vs Quest 71.25 | both current-query | **-1.50** |

Same digest, same budget, same eight seeds; the sign of the difference flips
with machinery that is not part of our contribution. Given that the whole
cluster sits inside the tie band -- and that both oracles sit in it too -- the
reading we are willing to defend is that **both differences are noise**, not
that speculation interacts favourably with a low-rank basis.

The practical rule for the paper: do not put OVAL Retv. (speculative) in the
same column as Loki or SparQ (non-speculative) and call it a win. The matched
arm for that table is OVAL Retv. no-spec., and in that table we are behind.

`OVAL Retv. no-spec.` is dropped from the Pareto figure for the same reason
the fixed-window and PyramidKV arms are: it differs from another plotted arm
only in harness machinery, not in the mechanism the figure is about. Its
results stay on disk and in `analysis/front.py`.

## Unmeasured

The harness counts `num_correct` and `num_correct_kv_heads`, so the fraction
of steps and heads that actually get corrected is recoverable but has not been
reported. It matters: if correction almost never fires, then nearly every
selection in the speculative arms was made from a stale query, which makes the
+0.50 over FreeKV harder still to attribute to the basis. Worth measuring
before any speculative number goes in the paper.
