# OVAL on a video LLM (Video-MME) — status

Branch `vision` of oval-kv, working tree `~/oval-vision`. Everything for this
experiment lives in `vision/`; nothing in `oval/` or `patches/` is changed.
Environment: `source vision/env.sh` (the `ovalkv` conda env: torch 2.8.0+cu128,
transformers 4.45.2, flash-attn 2.8.3, flashinfer 0.2.4 JIT), plus LLaVA-NeXT
(`bce12e4`, `pip install --no-deps -e`), decord, timm 0.9.16. Hardware: 4x RTX
A6000, **shared** — another user's training holds ~27-29 GB on every GPU, so each
run has ~20 GB and shares compute.

## 1. Repo map (no code changes)

1. **Long-context accuracy path.** `third_party/FreeKV/accuracy/eval/LongBench2/pred.py`,
   model loading and method set-up in `eval/util.py: load_model_and_tokenizer`
   (`AutoModelForCausalLM`, bf16, eager, `device_map=auto`). Generation is a
   **custom greedy decode loop** (`model(input_ids, past_key_values=...)` one token
   at a time), not HF `generate`; the cache is FreeKV's tuple cache
   (`kvc/patch/tuple_kv_cache.py`), not `DynamicCache`.
2. **Attention patch.** `kvc/patch/llama.py: enable_llama_dyn_attention` sets
   `module.forward = llama_dyn_attention_forward` for every `model.model.layers[i].self_attn`
   — it **replaces the whole forward** (projections, RoPE via flashinfer
   `apply_rope_inplace` on 1-D `position_ids`, cache, attention), and
   `enable_tuple_kv_cache` also replaces the model / decoder-layer / causal-LM
   forwards with Llama-style ones that accept `inputs_embeds` and keep only the
   last position's logits. Used for Llama and Qwen2 alike (Qwen2's q/k/v biases
   live in the Linear layers). → case **(b)**.
3. **Rule switches** (harness flags, `eval/util.py parse_common_args`): FullKV
   `--method full`; FreeKV `--method spec_ret --page_rep quest`; Quest
   `--method quest --page_rep quest`; 4-bit keys `--method quest --page_rep q4key`
   (attends with the dequantized keys); OVAL keys-as-basis `--method spec_ret
   --page_rep oval --oval_rank 8 --oval_recon`; OVAL full: + `--oval_vrank 16
   --vbasis pca --eta_table T.json`. Common: `--GQA_policy avgSM --spec_ret_steps 2
   --spec_ret_corr 0.8 --corr_group avg --skip_layer 1 --sparsity 1
   --last_layer_budget 0 --page_size 32 --batch_size 1`, `OVAL_QUANT=0`.
4. **eta_h calibration.** `analysis/eta_rules_calibrate.py` (reasoning only:
   seed-7 sampled traces, three windows of 96 *decode* queries). **There is no
   long-context (prefill-window) calibration path**: `docs/agents/long-context.md`
   lists it as not built ("no calibration path exists"; status "not started" on
   origin/main). A vision calibration path has to be written (see section 5).
5. **Memory accounting.** Closed forms in `analysis/oval_memory.py`,
   `analysis/baseline_memory.py`, `analysis/pareto.py` (Fig. 2: bytes per KV head
   per layer on a reference sequence; sink+recent exact; OVAL key basis
   (dr+mr+d)*2 = 2816 B/page at r=8, value basis 5376 B/page at r_v=16; min-max
   digest 512 B/page).
6. **Paired CIs.** `analysis/paired.py` (paired bootstrap over problems).
7. **Disk / versions.** 25-57 GB free on `/` (cleared pip/conda caches and
   unrelated HF models); Video-MME streamed chunk by chunk. Harness needs
   transformers 4.45.x (the Llama-style forwards are written for it).

## 2. Model

Case (b) → **`lmms-lab/LLaVA-Video-7B-Qwen2`** (stock Qwen2-7B LM: 28 layers, 4 KV
heads, plain 1-D RoPE). Qwen2.5-VL would need M-RoPE (3-D positions) inside the
replaced model forward and transformers >= 4.49. LLaVA-NeXT is installed with
`--no-deps` against transformers 4.45.2 (no separate env needed; LLaVA's own
`generate` is not used).

N = 64 frames (lmms-eval `force_sample`: `linspace(0, T-1, 64)`), 210 tokens per
frame → **13,440 visual tokens** per sample. N/2 = frames[::2] of the same cache
(6,720 visual tokens).

## Deviations from the brief (decided before any accuracy number)

- **Question streamed through the decode path (`--protocol stream`).** In this
  harness the prefill is dense for every rule, and Video-MME's answer is one
  letter, produced by the prefill's last position. With the question inside the
  prompt every row would answer from the full exact cache. So the prefill holds
  system prompt + video + time instruction ("cache the video, ask later"); the
  question/options/post-prompt (~80-120 tokens) and the answer go one token at a
  time through the decode path, i.e. through page selection and the compressed
  pages. Same for every row. Under FullKV this is mathematically the same
  computation as the standard protocol (checked in Gate B).
- **lmms-eval used for its task definition, not as the runner.** Prompt
  (`videomme_doc_to_text`, llava_vid post-prompt "The best answer is:"), sampling
  (`force_sample`, time instruction, conv template `qwen_1_5`), greedy,
  `max_new_tokens 16`, and the scorer (`extract_mcq_answer`) are copied/imported
  from lmms-eval `d45b4ad`; our own loop runs the model (the brief's fallback).
  Gate A runs the unpatched HF Qwen2 forward (sdpa) with a greedy loop that keeps
  only the last logits — LLaVA's `generate()` materialises 13K x 152K logits (8 GB)
  and does not fit next to the other user's jobs.
- SigLIP encodes frames 16 at a time (memory; frames are independent, features identical).

## Frame cache

`vision/frames.py`: decode once, 64 frames, resize to 384x384 with PIL bicubic
(what SiglipImageProcessor does), PNG. The processor's own resize is then the
identity. Every per-sample log carries `frames_sha1` (PNG bytes), `pixels_sha1`
(processor output) and `ids_sha1` (token ids).

## Gates

All on the first 100 Long questions (`--durations long --limit 100`), 64 frames.

- **Gate A — PASS.** Unpatched HF Qwen2 (sdpa), question in the prompt:
  **54.0%** on these 100 Long questions. Published LLaVA-Video-7B Video-MME
  (no subtitles, 64 frames): 63.3 overall, with Long the hardest split (low 50s);
  the overall number is checked when row 1 completes. Logs `vision/runs/gateA_official_n64`.
- **Gate B — PASS.** Patched harness, `--method full`, question in the prompt:
  99/100 same answer (and same text) as Gate A; input hashes identical.
- **Gate B, stream protocol — PASS (bf16 ties).** Patched dense, question streamed:
  97/100 same answer as Gate A. The three flips (601-1, 610-3, 621-1) have a
  top-2 logit margin of 0.0-0.25 in both runs, i.e. exact bf16 ties broken
  differently by the prefill kernel vs the one-token decode kernel. Accuracy 52.0
  vs 54.0 is those two ties.
- **OVAL first-step cost (found and fixed without touching `oval.py`).** The first
  decode step builds the value bases of all ~430 prefill pages per layer; the
  original `build_value_summary` eigendecomposes the 128x128 channel Gram, and
  batched 128x128 `eigh` has no fast CUDA path (6.7-11 s per layer on cusolver,
  magma or CPU) -> ~300 s per sample. `vision/fast_vpca.py` computes the same
  plain-PCA projection from the 32x32 token Gram (dual form; only used when
  `Gh is None`, i.e. `--vbasis pca`), swapped in at load time. Check on real
  video pages (`python vision/fast_vpca.py --check`, 5 layers): rebuilt values
  differ by <= 4.1e-4 relative (below bf16 rounding), value error identical to
  4 digits, 7 ms instead of 6.7 s per layer. Decode steps after the first take
  ~0.1 s, as for FreeKV. Reasoning runs never hit this (prompts fit in the sink).
- Observation: rank-16 value PCA leaves **25-35%** relative error on video pages
  (layers 1-27, calibration clips) — see the page-spectra figure.
- **Gate C** — pending (needs the calibrated table).
- **Gate D** — running (FullKV, 32 frames, stream, same 100 questions).

## Commands

```
source vision/env.sh
python vision/frames.py --chunks 1 ... 20            # frame cache
CUDA_VISIBLE_DEVICES=g python vision/run_videomme.py --protocol {official,prompt,stream} \
    --frames 64 --durations long --limit 100 --out vision/runs/<name> <harness flags>
```
