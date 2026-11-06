"""Video-MME (no subtitles) with LLaVA-Video-7B-Qwen2 inside the FreeKV/OVAL harness.

The harness's own argument parser and method set-up (eval/util.py
load_model_and_tokenizer) are used unchanged; only the model it loads is
swapped for LLaVA-Video, whose language model is a stock Qwen2-7B.

Protocols (--protocol), identical inputs in all three:
  official  unpatched LLaVA generate(), question in the prompt (Gate A).
  prompt    patched harness, question in the prefill (Gate B: must match official).
  stream    patched harness; the prefill holds system prompt + video only, so the
            cache is built (and, for the compressing rules, compressed) before the
            question is known; the question tokens and the answer then go through
            the decode path, i.e. through page selection and the compressed pages.
            With the question in the prefill, the one-letter answer would come
            from the dense prefill for every rule.

Prompt, sampling and scoring copy lmms-eval (tasks/videomme, models/simple/llava_vid):
force_sample, time instruction, conv template qwen_1_5, post prompt
"The best answer is:", greedy, max_new_tokens 16, extract_mcq_answer.

    python vision/run_videomme.py --protocol stream --frames 64 --out DIR \
        --method spec_ret --page_rep oval ... (any harness flag)
"""
import argparse, hashlib, importlib.util, json, os, sys, time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FK = os.path.join(HERE, "..", "third_party", "FreeKV", "accuracy")
LMMS = os.path.join(HERE, "..", "third_party", "lmms-eval")
sys.path.insert(0, FK)
sys.path.insert(0, HERE)

MODEL = "lmms-lab/LLaVA-Video-7B-Qwen2"
ANN = "/home/shahbaa/vmme/videomme/test-00000-of-00001.parquet"
CACHE = "/home/shahbaa/vmme/frames"
N_CACHE = 64

_spec = importlib.util.spec_from_file_location(
    "mcq_extract", os.path.join(LMMS, "lmms_eval", "tasks", "_task_utils", "mcq_extract.py"))
_mcq = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mcq)


def extract_letter(s):
    return _mcq.extract_mcq_answer(s, choices=["A", "B", "C", "D"])


def doc_to_text(doc):
    # lmms-eval videomme_doc_to_text with the llava_vid kwargs (post_prompt "The best answer is:")
    option_prompt = ("Select the best answer to the following multiple-choice question based on the video "
                     "and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.")
    question = doc["question"] + "\n" + "\n".join(doc["options"])
    return option_prompt + "\n" + question + "\n" + "The best answer is:"


def load_frames(vid, frames):
    d = os.path.join(CACHE, f"{vid}_n{N_CACHE}")
    meta = json.load(open(os.path.join(d, "meta.json")))
    if N_CACHE % frames == 0:            # N/2 = every other cached frame
        sel = list(range(0, N_CACHE, N_CACHE // frames))
    else:                                # e.g. 44: as uniform as the 64-frame grid allows
        sel = sorted(set(np.round(np.linspace(0, N_CACHE - 1, frames)).astype(int).tolist()))
        assert len(sel) == frames
    raw = [open(os.path.join(d, f"{i:03d}.png"), "rb").read() for i in sel]
    fh = hashlib.sha1(b"".join(raw)).hexdigest()[:16]
    arr = np.stack([np.array(Image.open(os.path.join(d, f"{i:03d}.png")).convert("RGB")) for i in sel])
    ftime = ",".join(f"{meta['frame_time'][i]:.2f}s" for i in sel)
    return arr, ftime, meta["video_time"], fh


def build_parser():
    from eval.util import parse_common_args
    ap = argparse.ArgumentParser()
    parse_common_args(ap)
    for a in ap._actions:                 # harness output dir is unused here (--out)
        if a.dest == "out_root_dir":
            a.required = False
    ap.add_argument("--protocol", choices=["official", "prompt", "stream"], required=True)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--durations", nargs="+", default=["short", "medium", "long"])
    ap.add_argument("--limit", type=int, default=None, help="first N questions after filtering")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--file_tag", default="", help="suffix of the shard file (keeps split phases apart)")
    ap.add_argument("--qids", nargs="+", default=None, help="diagnostics: only these question ids")
    ap.add_argument("--only_cached", action="store_true", help="smoke tests: skip videos not yet in the frame cache")
    return ap


def load_llava(args):
    from llava.model.builder import load_pretrained_model
    tok, model, proc, _ = load_pretrained_model(MODEL, None, "llava_qwen", torch_dtype="bfloat16",
                                                device_map="cuda:0", attn_implementation="sdpa")
    model.eval()
    # SigLIP attention is eager; encode frames 16 at a time (frames are independent, so
    # the features are identical) to stay inside the ~20 GB left on the shared GPUs.
    vt = model.get_vision_tower()
    vt_forward = vt.forward
    def chunked(images):
        if torch.is_tensor(images) and images.shape[0] > 16:
            return torch.cat([vt_forward(images[i:i + 16]) for i in range(0, images.shape[0], 16)])
        return vt_forward(images)
    vt.forward = chunked
    return tok, model, proc


def patch_harness(args, tok, model):
    """Run eval.util.load_model_and_tokenizer on the already-loaded LLaVA model."""
    from eval import util
    class _Gen:
        eos_token_id = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    orig = (util.AutoTokenizer.from_pretrained, util.AutoModelForCausalLM.from_pretrained,
            util.GenerationConfig.from_pretrained)
    util.AutoTokenizer.from_pretrained = staticmethod(lambda *a, **k: tok)
    util.AutoModelForCausalLM.from_pretrained = staticmethod(lambda *a, **k: model)
    util.GenerationConfig.from_pretrained = staticmethod(lambda *a, **k: _Gen())
    # The harness dispatches on config.model_type; LLaVA-Video's LM is a stock Qwen2.
    model.config.model_type = "qwen2"
    try:
        model, _, step_updater, eos_ids, config = util.load_model_and_tokenizer(MODEL, args)
    finally:
        model.config.model_type = "llava_qwen"
        (util.AutoTokenizer.from_pretrained, util.AutoModelForCausalLM.from_pretrained,
         util.GenerationConfig.from_pretrained) = orig
    return model, step_updater, eos_ids


def build_inputs(doc, args, tok, model, proc):
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    arr, ftime, vtime, fh = load_frames(doc["videoID"], args.frames)
    pix = proc.preprocess(list(arr), return_tensors="pt")["pixel_values"].to("cuda:0", torch.bfloat16)
    ph = hashlib.sha1(pix.float().cpu().numpy().tobytes()).hexdigest()[:16]
    ti = (f"The video lasts for {vtime:.2f} seconds, and {len(arr)} frames are uniformly sampled from it. "
          f"These frames are located at {ftime}.Please answer the following questions related to this video.")
    qs = DEFAULT_IMAGE_TOKEN + "\n" + f"{ti}\n{doc_to_text(doc)}"
    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    ids = tokenizer_image_token(prompt, tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda:0")
    # stream split: the prefill ends after the time instruction (it describes the video,
    # not the question); find its last token in the full tokenisation of the prompt.
    p = int((ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0])
    pre = tok("\n" + ti).input_ids
    full = ids[0, p + 1:].tolist()
    k = len(pre)
    while k > 0 and full[:k] != pre[:k]:
        k -= 1
    assert k >= len(pre) - 1, "time instruction does not tokenize as a prefix"
    build_inputs.n_ctx_text = k            # text tokens after the video kept in the prefill
    return ids, pix, dict(frames_sha1=fh, pixels_sha1=ph,
                          ids_sha1=hashlib.sha1(ids.cpu().numpy().tobytes()).hexdigest()[:16])


@torch.no_grad()
def answer(doc, args, tok, model, proc, step_updater, eos_ids):
    from llava.constants import IMAGE_TOKEN_INDEX
    ids, pix, hashes = build_inputs(doc, args, tok, model, proc)
    t0 = time.time()
    _, _, _, _, emb, _ = model.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, [pix],
                                                                     modalities=["video"])
    p = int((ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0])
    n_vis = emb.shape[1] - (ids.shape[1] - 1)
    kctx = build_inputs.n_ctx_text
    split = p + n_vis + kctx if args.protocol == "stream" else emb.shape[1]
    stream = ids[0, p + 1 + kctx:].tolist() if args.protocol == "stream" else []
    if args.protocol == "official":
        # Unpatched HF Qwen2 forward (sdpa), greedy; equivalent to LLaVA's generate() but
        # keeps only the last position's logits (generate() materialises all ~13K, 8 GB).
        from transformers import Qwen2ForCausalLM
        fwd = lambda **k: Qwen2ForCausalLM.forward(model, use_cache=True, num_logits_to_keep=1, **k)
        out = fwd(inputs_embeds=emb)
        pkv, gen = out.past_key_values, []
        top2 = out.logits[0, -1].float().topk(2)
        margin = (top2.values[0] - top2.values[1]).item()
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        for _ in range(args.max_new_tokens):
            gen.append(nxt.item())
            if gen[-1] in (im_end, tok.convert_tokens_to_ids("<|endoftext|>")):
                break
            out = fwd(input_ids=nxt, past_key_values=pkv)
            pkv = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        text = tok.decode(gen, skip_special_tokens=True).strip()
        return text, dict(hashes, n_visual=n_vis, prefill_len=emb.shape[1], n_gen=len(gen), margin=margin,
                          sec=time.time() - t0)
    if step_updater is not None:
        step_updater.reset(ids)
    out = model(inputs_embeds=emb[:, :split], past_key_values=None, use_cache=True)
    pkv = out.past_key_values
    t_prefill = time.time() - t0
    for t in stream:                      # question tokens, read through the decode path
        out = model(input_ids=torch.tensor([[t]], device="cuda:0"), past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
    gen = []
    top2 = out.logits[0, -1].float().topk(2)
    margin = (top2.values[0] - top2.values[1]).item()
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    for _ in range(args.max_new_tokens):
        gen.append(nxt.item())
        if gen[-1] in eos_ids:
            break
        out = model(input_ids=nxt, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    text = tok.decode(gen, skip_special_tokens=True).strip()
    ms = torch.cuda.memory_stats()
    return text, dict(hashes, n_visual=n_vis, prefill_len=split, n_stream=len(stream), n_gen=len(gen), margin=margin,
                      sec_prefill=t_prefill, sec=time.time() - t0,
                      peak_gb=torch.cuda.max_memory_allocated() / 2**30,
                      alloc_retries=ms.get("num_alloc_retries", 0))


def main():
    args = build_parser().parse_args()
    import pandas as pd
    df = pd.read_parquet(ANN)
    df = df[df.duration.isin(args.durations)].reset_index(drop=True)
    if args.only_cached:
        df = df[[os.path.exists(os.path.join(CACHE, f"{v}_n{N_CACHE}", "meta.json")) for v in df.videoID]]
    if args.limit:
        df = df.iloc[:args.limit]
    if args.qids:
        df = df[df.question_id.isin(args.qids)]
    df = df.iloc[args.shard::args.nshards]
    os.makedirs(args.out, exist_ok=True)
    fn = os.path.join(args.out, f"shard{args.shard}of{args.nshards}{args.file_tag}.jsonl")
    done = set()
    if os.path.exists(fn):
        for l in open(fn):
            if l.endswith("\n"):
                done.add(json.loads(l)["question_id"])
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
    tok, model, proc = load_llava(args)
    step_updater, eos_ids = None, None
    if args.protocol != "official":
        model, step_updater, eos_ids = patch_harness(args, tok, model)
    with open(fn, "a") as f:
        for _, r in df.iterrows():
            if r.question_id in done:
                continue
            doc = r.to_dict()
            doc["options"] = list(doc["options"])
            text, info = answer(doc, args, tok, model, proc, step_updater, eos_ids)
            pred = extract_letter(text)
            rec = dict(question_id=doc["question_id"], videoID=doc["videoID"], duration=doc["duration"],
                       task_type=doc["task_type"], answer=doc["answer"], output=text, pred=pred,
                       score=float(pred.lower() == doc["answer"].lower()), **info)
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(doc["question_id"], doc["duration"], repr(text), doc["answer"], f"{info['sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
