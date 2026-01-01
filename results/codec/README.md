# OVAL records as a key codec — rank sweep

Normally the records only **select** pages, and pages surviving top-k are
attended with original keys. With `--oval_recon` the selected pages are instead
rebuilt from the records (`mu + coef @ basis^T`), so the records become the
only copy of the keys. Values, sink and recent tokens stay exact, as in every
baseline. Reconstructing only the *selected* pages is equivalent to storing
every page compressed — attention never touches an unselected page.

Reproduce: `scripts/codec_grid.sh`. Config is the paper's reasoning protocol,
identical in every cell (B=2048, sink/recent 512, page 32, avgSM,
spec_ret_steps 2, corr 0.9, T=0.6, top-p 0.95, max_gen 16384, seed 42). Rank
and the codec on/off flag are the only things that vary.

## Accuracy — ds-r1-qwen-7b / MATH50, single seed

| rank | MATH50 | attn-out err | key store | K compression | ms/token |
|---|---|---|---|---|---|
| dense (FullKV) | 72% | 0.0% | 422.6 MiB | 1.00× | — |
| selection-only r=8 | 72% | 0.0% | 422.6 MiB | 1.00× | 54.1 |
| codec r=1 | 66% | 51.7% | 29.7 MiB | 14.22× | 58.1 |
| codec r=4 | 66% | 34.4% | 79.2 MiB | 5.33× | 59.8 |
| codec r=8 | 70% | 20.7% | 145.3 MiB | 2.91× | 59.4 |
| codec r=16 | 74% | 6.6% | 277.3 MiB | 1.52× | 59.3 |
| codec r=32→31 | 76% | 0.0% | 525.0 MiB | 0.81× | 59.6 |

**Read the trend, not the cells.** At n=50 one problem is 2 points, and
pairwise McNemar over these arms gives p ≥ 0.5 everywhere — including
`selection-only r8 (72%)` vs `codec r8 (70%)`, which is 3 problems flipping one
way and 2 the other. r=16 and r=32 scoring above dense is noise, not the codec
beating exact keys. The *trend* is resolvable: Pearson r = 0.922 against
log2(rank), permutation p = 0.012, and it tracks the measured reconstruction
error. `analysis/score_codec.py` reports it this way by construction.

## Latency is flat in rank

58–60 ms/token across a 31× range of rank (16K context, 497 resident pages).
The reconstruction GEMM is not the cost — materialising the selected pages is,
and that happens either way. The codec costs ~10% over selection-only
(54.1 → 59.4 ms), which is the price of not reading exact keys.

## Memory: break-even is rank ~25

Above rank 25 the bfloat16 record is *larger* than the raw page it replaces
(r=32 is 0.81×, an expansion). Because a key basis never compresses values,
whole-cache compression is far below the key-only ratio at every rank — 1.44×
at r=8 against 2.91× on keys alone. `analysis/codec_memory.py` is the closed
form; it matches the slab bytes a real run allocates, to the byte.

## The comparison that matters: vs just quantizing the keys

Measured on the same captured decode keys, same metric, all 28 layers
(`analysis/codec_vs_quant.py`):

| representation | B/pg/kvh | vs raw | median attn err |
|---|---|---|---|
| raw bfloat16 | 8192 | 1.00× | 0.0% |
| rank-16 bf16 | 5376 | 1.52× | 6.5% |
| **int4 raw keys (per-channel)** | 2304 | **3.56×** | **6.8%** |
| int8 raw keys (per-channel) | 4352 | 1.88× | 0.3% |
| rank-16 + int4/int8 packing | 1762 | 4.65× | 8.2% |
| rank-8 + int4/int8 packing | 978 | 8.38× | 20.5% |

At matched error, plain int4 quantization gives 2.3× more compression than
rank-16; int8 gives more compression *and* 20× lower error. **As a standalone
codec, low-rank is dominated by scalar quantization.** It only becomes
competitive stacked with the record packing (4.65× at 8.2%), and even then it
draws with int4 rather than beating it.

The defensible claim is therefore narrow: the basis is an excellent *index* and
a mediocre *codec*. At rank 1 it still ranks 98.1% of true top-32 attention
mass while reconstructing keys at 51.7% error — selection needs almost no rank.
Using the records as the cache removes the objection that they are extra state
(at r=8 they *are* the keys, at 2.91× / 8.38× packed), but the compression
ratio is not a selling point against a quantization baseline that compresses
values too.
