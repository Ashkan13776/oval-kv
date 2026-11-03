# Underdetermined parameters — NOT silently assumed

The paper does not pin these down, and each one moves the numbers. Listed with
the option I'd pick and what it costs us if wrong. Nothing here is decided.

## Q1. The 50-record subset (BIGGEST GAP)

"seeded subset of 50 records per task" — the seed and the sampler are not
published, and the released artifact has no `benchmarks/` directory. Different
50-record draws give different scores.

How much this matters: the per-task CIs are huge because n=50. `kv-retr` is
28.0 ±12.6, `dialog-qa` 38.0 ±13.5, `code-debug` 32.0 ±13.0. A different draw
can move a task by 10+ points and still be "consistent". The 11-task Avg CI is
±2.6, so the Avg is the only cell with real discriminating power.

Options: (a) first 50 records per task, (b) `random.Random(0).sample`, (c) run
all records per task and report the full-split number instead. (c) removes the
subset ambiguity entirely but costs ~4-8x the GPU time on the big tasks.

## Q2. Prompt templates

Not stated. InfiniteBench ships official per-task prompts; GLM-4-9B-Chat-1M
also has its own chat template. Whether the paper applied the chat template on
top of the InfiniteBench prompt is unknown, and it materially changes scores on
the QA/choice tasks.

## Q3. Generation length per task

Not stated. InfiniteBench's reference implementation sets per-task
`max_new_tokens` (e.g. book-sum needs ~1200, passkey ~6). Our choice changes
book-sum (a 2.2-point-CI task, so it is actually discriminating) the most.

## Q4. Sampling

sec5_eval.tex says "greedy" only in the Llama-3.1-8B retrieval figure caption
(fig q-retr). For InfiniteBench, sec5_eval.tex:13 defers to "per-arm sampling
settings ... stated with the per-task tables", but the InfiniteBench table
notes do not state them. Greedy is the obvious inference — but it is an
inference, so flagging it.

## Q5. Truncation policy for over-long records

Not stated. Some InfiniteBench records exceed 200K tokens. InfiniteBench's
reference implementation truncates from the middle. This interacts with a hard
hardware limit on our side — see deviations.md D2.

## Q6. `score` metric definition

Paper says `\texttt{score}`. Assumed to be InfiniteBench's official per-task
scorers, since the numbers are on InfiniteBench's natural scales (100.0 for
passkey/num-str, ~27 for book-sum ROUGE). Low risk but unconfirmed.
