# Figure 2 protocol — what the paper fixes, and what we had to decide

Figure 2 of the LOCKS paper (`q_retr`, §5.1) is **"Retrieval and QA vs. budget"**:
three panels, all on Llama-3.1-8B, sweeping the per-(layer, KV-head) budget
`b ∈ {64, 128, 256, 512, 1024, 2048}` against a same-engine FullKV.

| panel | benchmark | paper LOCKS (b=64→2048) | paper FullKV |
| --- | --- | --- | --- |
| (a) | LongBench-v1, 14-subset mean, full split | 45.7 46.7 46.7 47.1 47.2 47.2 | 47.0 |
| (b) | RULER-16K, 13-task mean | 78.1 87.4 90.6 91.4 93.0 93.7 | 94.3 |
| (c) | RULER-32K, 13-task mean | 73.1 83.4 85.5 86.1 87.2 88.0 | 88.8 |

Per-task targets for all three panels are in App. C (`tab_appC_benchmarks`,
Tables `appC-lbv1` / `appC-ruler16` / `appC-ruler32`), so every cell can be
validated individually, not just the panel mean.

The paper states only: *Llama-3.1-8B, greedy, same-engine FullKV, identical
records across methods.* Everything below is either inferred with stated
evidence or taken from the benchmark's own reference implementation.

## Decisions and their evidence

### Model: `Llama-3.1-8B-Instruct`
The paper never gives an HF model ID, only "Llama-3.1-8B" with a citation.
Two independent pointers say Instruct:
- the `locks-kv` README's own example is `vllm serve meta-llama/Llama-3.1-8B-Instruct`
- RULER's own `config_models.sh` maps `llama3.1-8b-chat → llama3.1-8b-Instruct`
  with `MODEL_TEMPLATE_TYPE="meta-llama3"`, the template family the paper's
  RULER numbers would have to come from

**Weights provenance.** `meta-llama/Llama-3.1-8B-Instruct` is gated and this
account is not on the authorized list. We pull from
`NousResearch/Meta-Llama-3.1-8B-Instruct` instead. This is *not* a substitution:
all four safetensors shards were verified **sha256-identical** to the canonical
gated repo's blobs before use. Same bits, so no deviation is recorded.

### RULER: generated with NVIDIA's official `scripts/data/prepare.py`
RULER is synthetic — it is *built* per tokenizer, not downloaded — so the
generator, seed and tokenizer all matter. Chosen over the pre-generated HF
copies (`simonjegou/ruler`, `MaxJeblick/Ruler`), which are third-party
generations with different needles and haystacks.

- 13 tasks, exactly the paper's list (NIAH-S1/2/3, MK1/2/3, MV, MQ, VT, CWE,
  FWE, QA-1, QA-2), taken from the repo's `synthetic.yaml` unmodified
- tokenizer: the Llama-3.1-8B-Instruct tokenizer, `--tokenizer_type hf`
- template: `--model_template_type meta-llama3`
- seed: **42**, RULER's own default (`--random_seed`), not a choice of ours
- haystack sources: the repo's own downloaders — 218 Paul Graham essays,
  SQuAD dev-v2.0, HotpotQA dev-distractor, the bundled English wordlist

### RULER: 50 records per task
**Not published.** Inferred, and the inference is checked against every
published cell. RULER's scorer divides by the number of references per record
(`string_match_all`), so each task family implies a distinct score granularity
at n=50 — and every value in the paper's tables is an exact multiple of it:

| task family | refs/record | slots at n=50 | granularity | paper values |
| --- | --- | --- | --- | --- |
| NIAH-S/MK, QA-1/2 | 1 | 50 | 2.0 | 98.0, 44.0, 84.0 ✓ |
| NIAH-MV, NIAH-MQ | 4 | 200 | 0.5 | 98.5, 79.5, 87.5 ✓ |
| VT | 5 | 250 | 0.4 | 82.8, 78.4, 20.8 ✓ |
| CWE | 10 | 500 | 0.2 | 87.4, 6.6, 0.2 ✓ |
| FWE | 3 | 150 | 0.667 | 96.7 (=145/150) ✓ |

RULER's own default is 500, which would contradict all five granularities.
Confirmed with the user before running.

### LongBench-v1: official reference protocol, full split
2,750 records across the 14 subsets the paper names (200 each, 150 for
`multifieldqa_en`) — matches "full split".

From the reference harness (`THUDM/LongBench`, `LongBench/pred.py`):
- prompt template per subset from `config/dataset2prompt.json`
- generation budget per subset from `config/dataset2maxlen.json`
- middle-truncation, head half + tail half (`pred.py:60`)
- **chat-template carve-out**: the chat template is applied to every subset
  *except* `trec`, `triviaqa`, `samsum` (`pred.py:63` — "chat models are better
  off without build prompts on these tasks"), which are fed as raw few-shot
  completions
- scoring: `eval.py`'s `dataset2metric` map, max over reference answers, with
  the first-line truncation `eval.py` applies to trec/triviaqa/samsum

**Truncation length.** Not stated by the paper. We do not truncate: the model's
native window is 128K and the figure's own caption says the LongBench average
context is ~11K tokens, which is the *untruncated* average. Truncating at the
LongBench default (3.5K/7.5K/31.5K) would contradict that caption.

## Arms

LOCKS at each of the 6 budgets plus a same-engine FullKV reference, per the
standing decision to reproduce the LOCKS row only. The paper additionally plots
an exact-LSE Oracle and four baselines (Quest, KVzip, ShadowKV, RocketKV);
those are out of scope here.

## Carried over from the InfiniteBench reproduction

- `LOCKS_DECODE_TRITON=1` — sm_120 has no hand-CUDA decode lane (deviations D1)
- page size 16, `block_size=16`, so `budget_pages = b / 16`
- sink=1 and window=1 counted *inside* b
- `enforce_eager`, prefix caching off, bf16 KV, `max_num_seqs=4`
- greedy (`temperature=0`), matching the figure caption
