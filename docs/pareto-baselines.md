# Pareto front on MATH50: 26 arms at matched budget

All arms run at the config the existing numbers use — `B=2048`, `sink 512`,
`recent 512`, `page_size 32`, 8 seeds, `ds-r1-qwen-7b`. No budget sweep: every
arm attends the same number of tokens, and the front is over the two axes that
are genuinely the method's own — the per-page digest it must store to decide
what to attend to, and what that decision costs at decode.

## The arms

Grouped by mechanism, because a front that mixes mechanisms without saying so
is the easiest way to publish an unfair comparison.

| family | arms | per-page digest |
| --- | --- | --- |
| bound digests | `quest`, `arkvale`, `freekv` | 512 B (min/max or centre±MAD) |
| low rank | `oval` (r=8), `loki` (shared basis), `centroid` (rank 0) | 2816 B / 512 B / 256 B |
| quantised keys | `q4key` (4 bit per channel) | 2560 B |
| representative tokens | `infllm` (top-4 by norm), `shadowkv` (centroid + outliers) | 1024 B / 256 B |
| query sparsity | `sparq` (top d/8 query dims) | 0 (reads the keys) |
| two stage | `rocketkv` (centroid coarse → exact fine) | 256 B |
| attention history | `h2o` (cumulative), `tova` (last step), `snapkv` (frozen) | 4 B / 4 B / 0 |
| query agnostic | `expatt` (prompt query distribution), `knorm` (key magnitude) | 0 / 2 B |
| budget allocation | `pyramid` (layer-wise schedule) | 512 B |
| no retrieval | `stream` (sink+recent), `sliding` (budget spent on a wider window) | 0 |
| controls | `random` (per step) | 0 |
| ceilings | `oracle` (true attention), `oracle_prev` (q_{t-1}) | n/a |
| dense | `full` | n/a |

`q4key` is the sharpest contrast on the memory axis: it stores keys at 4 bits
per channel, 2560 B/page against OVAL's 2816 B record. Both replace the keys
with something smaller in order to decide what to attend to, at essentially the
same footprint, so it is the direct test of whether a low-rank basis beats
scalar quantisation at equal size.

`sliding` surrenders its budget and spends it on a wider recent window, so it
attends exactly as many tokens as every other arm. It answers "is retrieval
better than just a bigger window?", which `random` does not.

## Families and the query they score with

- **A, `--method quest`** — scored with the CURRENT query, no speculation.
  This is how Loki, SparQ, InfLLM, H2O and SnapKV actually work, and it
  isolates the scoring rule from the retrieval machinery.
- **B, `--method spec_ret`** — FreeKV and ArkVale *are* speculative retrieval;
  OVAL's headline arm is run the same way so the number stays comparable with
  what is already reported.
- **C** — non-retrieval references.

`oracle` under A is the true ceiling; under B it is the q_{t-1} ceiling, which
is the correct bound for FreeKV's claim that the previous query is a good
enough proxy.

## What this benchmark does to the eviction methods

MATH50 prompts are tiny: median ~36 tokens, max ~182. With `sink 512`, **the
entire prompt sits inside the sink**, so every page is generated reasoning
tokens and there are no prompt pages at all. Three consequences, all of which
were bugs before they were findings:

- **SnapKV's observation window scores nothing.** Its selection is supposed to
  come from the prompt tail. Here there is no prompt page to score, so the
  frozen selection is anchored at compression *onset* instead — the first step
  the cache exceeds `sink+recent+budget` — using the query available there.
- **H2O's prompt accumulation initialises nothing.** An all-zero history makes
  `topk` break ties toward the lowest indices, so H2O locks onto the earliest
  pages permanently. Also anchored at onset.
- **SnapKV cannot hold a fixed budget and stay faithful.** Published SnapKV
  never evicts generated tokens; auto-retaining them here grows the attended
  set into dense attention, because most of a reasoning trace arrives after the
  selection is frozen. We keep the budget fixed and let post-anchor pages be
  covered only while they sit in the recent window. This is a DEVIATION, and it
  is also the finding: SnapKV's premise — that a static observation window
  predicts future attention — fails when most of the context is generated after
  the window closes.

Compression onset also explains why the budget matters so much here. With
`sink+recent = 1024` always exact and a mean sequence of 3869 tokens,
compression does not engage until token `1024+B`:

| B | onset | share of trace compressed |
| --- | --- | --- |
| 128 | 1152 | ~70% |
| 512 | 1536 | ~60% |
| 2048 | 3072 | ~20% |

At `B=2048` the methods are barely exercised, which is why the published arms
land within 1.5 points of each other with overlapping CIs. A supplementary
budget sweep is kept in `scripts/pareto_arms.sh` (set `B=`) for the curve.

## Testing the gap

Unpaired CIs over problems are dominated by how much problems differ in
difficulty, which is irrelevant: every arm sees the same 50 problems under the
same 8 seeds. `analysis/paired.py` pairs on the problem and bootstraps the
difference, which narrows the interval from about ±25 points to ±2.

At `B=128`, 8 seeds, paired against FreeKV:

| arm | Δ | 95% CI | |
| --- | --- | --- | --- |
| OVAL | +0.25 | [−1.50, +2.50] | tie |
| quest | −1.25 | [−2.50, 0.00] | borderline |
| arkvale | −1.50 | [−3.50, +0.25] | tie |
| raas | −11.25 | [−18.00, −5.25] | real |

So the defensible claim is **not** that OVAL is more accurate. It is that OVAL
matches FreeKV's accuracy at 35% less key memory (1182 vs 1815 KiB per head per
layer). RaaS is genuinely worse; the rest are a statistical tie.

## Implementation faults found while building this

Recorded because each produced plausible-looking numbers from a broken
mechanism, and output text alone would not have caught any of them.

- **TOVA was the oracle.** It stored the attention history and then returned
  the *current* query's exact attention.
- **H2O leaked the current query too**, and accumulated over pages it never
  attended — something an eviction method cannot observe. Both now score from
  history through step t−1 only, accumulated after attention over the attended
  set, so a dropped page stops accruing.
- **The oracle's softmax excluded sink and recent from the denominator.**
  Harmless within a head, but scores are then averaged over the GQA group, so a
  head putting 95% of its mass on the sink had its page preferences inflated
  ~20×. Now normalised over the full context.
- **Recent pages accumulated no history**, so a page graduating out of the
  recent window arrived with score 0 and could never be selected.
- **Unknown `page_rep` fell through to Quest**, which would have written
  Quest's numbers to disk under five different baseline names. Now raises.
- **PyramidKV collided with FreeKV on disk** — same `page_rep`, so the same
  filename. Tagged.
- **The prefill hook sat inside `if self.kv8:`** and never ran unquantised, so
  SnapKV, H2O, TOVA and Expected Attention got no prompt state at all.

## Validating that an arm is real

Smoke tests that only check the exit code prove nothing here: if
`sink+recent+budget` exceeds the sequence length the selection path never runs,
every arm falls through to dense attention, and all of them return byte
identical output while exiting 0.

Two checks that do work:

1. **Output fingerprint** — distinct hashes mean distinct selections. Requires
   a config where compression actually engages for most problems.
2. **Selection dump** — `KVC_DEBUG_SEL=1` with `KVC_DEBUG_LAYER=<n>` prints the
   chosen page indices. Point it at a layer past `skip_layer` (layer 0 never
   compresses), and sample deep into generation: at the first compressing step
   the page count equals the budget by construction, so every arm trivially
   picks the same pages.

Verified distinct at `sink 32 / recent 32 / budget 128`, steps 100–102:

```
quest     [1,4,5,6] [1,3,5,6] [1,4,5,6]    query dependent
oval      [0,1,3,6] [1,3,4,6] [0,1,3,6]    differs from quest
oracle    [1,3,5,6] [3,4,5,6] [1,4,5,6]    varies most
loki      [0,1,2,4] [1,2,3,4] [0,1,2,3]    prefers low pages
sparq     [0,4,5,6] [1,4,5,6] [1,4,5,6]
infllm    [1,4,5,6] [1,4,5,6] [3,4,5,6]    differs from sparq
random    [1,2,4,5] [0,1,3,4] [1,4,5,6]
h2o       [0,3,4,5] x3                     cumulative, slow moving
knorm     [0,1,4,5] x3                     query free, constant by design
expatt    [0,4,5,6] x3                     query agnostic, constant by design
```

The constant ones are correct: K-norm and Expected Attention do not depend on
the query, so an unchanging selection is the expected signature rather than a
bug. `q4key`, `shadowkv` and `rocketkv` are not yet in this table.


## Result: the front is flat at B=2048, and dense proves it

First read with 24 arms reporting (7 at the full 8 seeds, the rest at 3-4):

| arm | avg@k | KiB/hd/lyr |
| --- | --- | --- |
| pyramid | 72.67 | 1815 |
| loki | 72.50 | 1815 |
| sparq | 72.00 | 1760 |
| rocketkv | 72.00 | 1788 |
| **oval** | **71.75** | **1182** |
| infllm | 71.50 | 1870 |
| shadowkv | 71.33 | 1788 |
| quest / freekv | 71.25 | 1815 |
| **full (dense)** | **70.75** | 1760 |
| arkvale | 70.25 | 1815 |
| stream | 59.33 | 512 |
| raas | 50.50 | 1536 |

**Dense ranks 11th of 24.** Ten compressed arms "beat" full attention, which
cannot be a real effect: attending to a subset of the keys cannot be better
than attending to all of them. The spread across the top twelve is therefore
sampling noise, not method quality, and no ordering within it should be
reported as a result. Paired against FreeKV, only `stream` (-11.33) and `raas`
(-20.75) have intervals excluding zero; every other arm, OVAL included
(+0.50, CI [-1.25, +2.50]), is a statistical tie.

The cause is the budget arithmetic above: at B=2048 compression does not engage
until token 3072 against a mean sequence of 3869, so roughly 20% of the trace
is compressed and the scoring rule barely participates. The config is the right
one for consistency with the published table -- it simply cannot discriminate
between methods. B=128 is therefore run as a second operating point, where
~70% of the trace is compressed; the same RaaS comparison moves from -11.25 to
-20.75 between the two, which is the budget actually stressing the method.

### RocketKV is degenerate at this budget, by construction

RocketKV's coarse stage admits `2 x page_budget` pages before the exact pass
ranks them. At B=2048 the budget is 64 pages out of ~89 paged pages on a mean
sequence -- the budget already covers 72% of the context, so there is nothing
for a pruning stage to prune, and the arm reduces to plain exact scoring. Its
runtime advantage, which is the whole point of the method, cannot appear. This
is not worked around: forcing the coarse stage to prune would make it a
different method. It is reported as degenerate at B=2048 and measured properly
at B=128, where the budget is 4 pages of ~89 and the coarse stage does real
work. The same caveat applies to any two-stage or pruning baseline here.
