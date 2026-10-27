# Value codec: compressing values as well as keys

The key codec (`--oval_recon`) rebuilds the selected pages' keys from their
rank-8 records. This study also replaces their **values** with a low-rank
record, so both halves of the paged cache are compressed. Sink and recent
tokens stay exact for both, as in every baseline.

## What is swept

`ds-r1-qwen-7b`, MATH50 / AIME24 / GPQA50c, k = 8 seeds (42-49), avg@k and
pass@k. One cell per (dataset, value rank).

| | |
|---|---|
| keys | rank 8 for scoring and reconstruction; eta fixed per dataset: MATH50 0.25, AIME24 1.0, GPQA50c 0.25 |
| values | plain PCA of the centred page values (`--vbasis pca`), ranks 1, 4, 8, 16, 32 |
| protocol | identical to the keys-only codec runs: B=2048, sink/recent 512, page 32, avgSM, spec_ret 2 / 0.9, T=0.6, top-p 0.95, max_gen 16384, bf16 records |

**Rank 64 is not run.** A centred 32-token page has rank at most 31, so a
request for 32 or 64 clamps to 31 and reconstructs exactly (0.3% residual in
the harness is bf16 rounding of the record). Ranks 32 and 64 would produce
byte-identical runs; rank 32 is kept as the lossless reference. Rank 64 needs
pages of 64+ tokens, which changes the key protocol too.

Run (supervised, restartable, one rank after another):

```bash
KV_PY=$(which python) setsid nohup scripts/value_rank_sweep.sh > logs/value_rank_sweep.out 2>&1 < /dev/null &
```

Score: `python analysis/value_rank_table.py` (all ranks, paired CIs), or one rank
with `python analysis/score_avg_pass_at_k.py --models ds-r1-qwen-7b --vrank 16 --vbasis pca`.

## Result

Full table and caveats: [`results/codec/value_rank_qwen7b.md`](../results/codec/value_rank_qwen7b.md).
Against the lossless rank-32 reference from the same sweep, **values compress to
rank 8 at no measurable cost on any task**; rank 4 costs about 8 points on
AIME24 (significant) and nothing measurable on MATH50 or GPQA50c; rank 1 fails
on all three. Value compression does not explain the AIME24 gap to dense --
that gap is there with lossless values, so it comes from the key codec.

## The value record

`oval/page_basis.py::build_value_summary` stores `(nu, D, T)` per page and KV
head, the same layout as the key record, and reconstructs `V_hat = 1 nu^T + T D^T`.
With `Gh = Ghi = None` (the swept configuration) it is plain PCA: `nu` is the
page mean, `D` the top-r right singular vectors of the centred values, `T`
their coefficients. Verified equal to the Eckart-Young optimum at every rank
(`analysis/test_page_basis.py`, check 8).

### Output-aware variant (implemented, not swept)

`--vbasis oval --veta E` builds the basis in the keys' `G = W_O^T W_O` metric.
With `Z = Y G^{1/2}` and `X`, `Y` the centred keys and values:

    M_rec = Z^T Z,   M_out = Z^T X X^T Z = A A^T,  A = G^{1/2} Y^T X
    M_eta = (1-eta) M_rec / tr(M_rec) + eta M_out / tr(M_out)
    U = top-r eigenvectors of M_eta;  T = Z U,  D = G^{-1/2} U

`M_out` comes from the first-order expansion of the output error
`du = sum_i W_i sum_t p_it e_t`: the page mean is exact, so the uniform part of
attention cancels and what remains is `tr(E^T X X^T E G)`. Both terms carry
`G` on the channel side, so the hybrid objective is `||M_t^{1/2} E G^{1/2}||^2`
-- separable -- and this record is its exact rank-r optimum over all
reconstructions (check 11). With Euclidean reconstruction for the first term
instead, the mix is no longer separable and has no closed form.

On real Qwen-7B decode state (`analysis/value_codec_check.py`) the output term
does not help: at rank 16 the residual-stream error is 1.50% for eta=0 (PCA in
the G metric), 1.55% for plain PCA and 1.77% at eta=1. Attention in reasoning
decode is too peaked for the first-order term to matter.

## Fixed along the way

- **Sweep restart skipped partial cells.** `codec_reasoning_sweep.sh` counted
  seed *files*, so once all 8 seeds had started a (dataset, eta) was skipped
  for good and every partial cell stayed unfinished after any restart. It now
  queues each seed whose file holds fewer records than the dataset has problems.
- **Partial trailing record.** A cell killed mid-`json.dump` left half a
  record; the next resume appended onto that line and corrupted two records.
  `pred.py` now truncates to the last newline before resuming.
- **Blind stall detector.** `grep -c . || echo 0` printed `0\n0` on an empty
  results directory, which broke every later comparison in
  `supervise_sweep.sh`. It also now *restarts* the cells after 45 minutes with
  no record from any of them (the longest normal machine-wide gap is ~8 min),
  rather than only logging.
- **Glob collisions.** Every `..._recon1*` glob (sweep, supervisor, scorer)
  would have mixed value-codec cells into keys-only cells. All now match the
  tag exactly (`_recon1-pQ2` vs `_recon1_vr16_vpca-pQ2`).

A resumed cell re-seeds at restart, so problems after a resume are drawn from a
different RNG stream than an uninterrupted run would use. That is statistically
equivalent and was already the case for every earlier sweep.

## Environment (this machine)

4 x RTX A6000 (sm_86), shared with another user's jobs. Conda env `ovalkv`:
torch 2.8.0+cu128, flash-attn 2.8.3, transformers 4.45.2 (FreeKV's pin),
flashinfer 0.2.4 (JIT, needs the env's CUDA 12.8 nvcc and `CUDA_HOME`),
`TORCH_CUDA_ARCH_LIST=8.6`. ~18 GB per cell, so `SLOTS=2, MINFREE=20000`.
