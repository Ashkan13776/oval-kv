# Forced deviations from the paper

Things we cannot match, with the reason and the expected blast radius.

## D1. GPU architecture: H200 (sm_90) → RTX PRO 6000 Blackwell (sm_120)

The paper measured on H200 NVL. We have an RTX PRO 6000 Blackwell Max-Q,
capability (12, 0). This is not a free substitution — the shipped code
**explicitly de-supports it**:

    backend/_runtime.py:733
    assert not _arch.is_blackwell_sm120(), (
        "locks fast decode: the sm_120 dispatch lane was removed 2026-07-21 "
        "(no-fallback rule: no silent wrong-arch route).")

Escape hatch: `_decode_backend()` checks `LOCKS_DECODE_TRITON=1` *before* the
assert, so forcing the Triton decode path bypasses it.

Caveat: the r8i4 **score** kernel is CUDA-only ("no Triton reference exists",
config.py:64). So the plan is Triton decode + CUDA r8i4 score. arch.py claims
every PTX instruction used was verified to assemble and execute on sm_120a, and
that only the shared-memory budget differs (99 KB vs Hopper's 227 KB, handled
by a tile shrink). Unverified by us — this is smoke-test #1.

Accuracy impact: should be nil if the kernels are numerically equivalent.
The decode path swap changes floating-point reduction order, so expect
small non-bitwise differences, not systematic shifts.

## D2. Context ceiling — RESOLVED, not a real limit

Initially read as a hard 262,144-token cap on sm_120 (`select.py:128`: "sm_120
99 KB -> 16384 pages -> 262144 tokens"). That is only the cap on the *Triton*
top-b kernel, which sorts the whole selectable slice in shared memory.

`topb_select()` does **capacity dispatch**: when `S_PAD * 4 > smem_cap()` it
routes to `_topb_select_torch`, "EXACT top-b, PAGE-COUNT-INDEPENDENT
(torch.topk; no shared-memory sort) ... unbounded in page count", with
semantics documented as identical (same tau, same count-and-demote tie rule,
same ascending compaction). So the ceiling lifts automatically.

Two costs, both accuracy-neutral:

1. **Not CUDA-graph-safe.** The torch twin host-syncs per request (`.item()`)
   and asserts it is not called under capture. So we must run
   `enforce_eager=True`, deviating from the paper's "full CUDA graphs"
   (appF_repro.tex:15). CUDA graphs are a latency feature; since we are
   measuring accuracy only, this does not affect any number we report.
2. **Equivalence is claimed, not gated.** The code says the torch twin "is not
   bitwise-gated in-engine yet (no EXACT_TOPB_VERIFICATION artifact on disk)".
   We take the documented semantics at face value.

**The real ceiling is GPU memory.** GLM-4-9B is 80 KB/token of KV
(40 layers x 4 KV heads x 128 dim x 2 x bf16). With 18 GB of weights and ~10%
LOCKS summaries, 97 GB holds roughly 800K tokens; the paper's 143.7 GB H200
holds ~1.3M.

**Measured impact: 4 records out of 550** exceed ~800K, all in
`longbook_qa_chn` (see notes/length-stats.md). All 4 also exceed the model's
own 1M native window (`seq_length=1048576`), so they required truncation on the
paper's hardware as well. We truncate them slightly harder (~780K vs ~1M).
That is 4/50 records in 1 of 11 tasks — under 1% of the 11-task Avg, against a
CI half-width of ±2.6.

## D3. Software stack

Paper: vLLM 0.24.0 / torch 2.11.0+cu130 / CUDA 13.0 / NGC 25.08-py3.
vLLM 0.24.0 is on PyPI, so the version is matchable. Open risk is whether
0.24.0-era builds have sm_120 kernel coverage, or whether we need a source
build. Decided after the environment probe.

## D4. Exact-match is impossible regardless

Even with a perfect environment, the seed for the 50-record subset is
unpublished (Q1), so cell-for-cell equality is not a reachable goal. The
meaningful test is whether our numbers land inside the paper's 95% CIs, with
the 11-task Avg (±2.6 at b=2048) as the primary endpoint.

## D6. The `fast` variant has no VRAM reservation for its own summary state

`memplan.py` exists precisely to stop this: it patches
`Worker.determine_available_memory` so vLLM's profiling-based `num_gpu_blocks`
sizing accounts for LOCKS' own allocations. But `register.py:260` gates it:

    if cfg.is_mem:
        from .memplan import patch_available_memory
        patch_available_memory()

`variant="fast"` — the K+V-resident arm Table 1 uses — never gets the patch. So
vLLM sizes the KV pool to `gpu_memory_utilization`, and LOCKS *then* allocates
the r8i4 summary (~9.5% of the pool, measured `selector_MiB=6346` at 49,870
blocks) on top. At `util=0.93` this OOMs during the first chunked prefill.

Workaround: run at `gpu_memory_utilization=0.85`, leaving room for the summary
plus prefill activations. Verified: KV pool 797,920 tokens, no OOM.

This changes only how much cache is resident, not selection or attention math,
so it does not affect accuracy — it just caps the longest servable record.

## D5. No harness, no baselines

`locks.bench` is declared as a console script in `pyproject.toml:74` but the
module ships in neither the wheel (45 files, zero matching "bench") nor the
sdist. Both GitHub URLs in the paper 404 (`Js-Hwang1/locks` from appF_repro
and `Js-Hwang1/locks-kv` from PyPI metadata). So `benchmarks/run.py`,
`benchmarks/registry.py`, and `vllm/integration/runner_common.py` — all cited
in Appendix C — are unavailable. We write the InfiniteBench runner and scorers
ourselves. Out of scope for this reproduction: FullKV, Oracle, and the three
baseline rows.
