"""Qwen2.5-VL-7B-Instruct references in the official HF stack (env ~/qwenvl, transformers 4.51.3),
from the same stored frames as every harness arm:

  --mode prompt    question in the prompt: processor(full chat prompt, the 64 stored frames),
                   unpatched Qwen2_5_VLForConditionalGeneration, greedy (the lmms-eval setting).
  --mode hfstream  the stream protocol computed by HF itself: prefill = the stored embeddings
                   and 3-D positions from qwen_prep.py, then the question tokens and the answer
                   one at a time with explicit (t = h = w) text positions. This is what the
                   harness's --method full must reproduce; it checks the M-RoPE integration.

Writes the same per-question jsonl as run_qwen25vl.py (question_id, pred, score, margin, hashes).

    ~/qwenvl/bin/python scripts/revision/C/qwen_official.py --mode prompt --bench videomme --durations long --limit 100 --out DIR
"""
import argparse, hashlib, json, os, socket, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qwen_prep as QP


def lm_forward(model, **kw):
    out = model.model(**kw, use_cache=True)
    return model.lm_head(out.last_hidden_state[:, -1:]), out.past_key_values


@torch.no_grad()
def answer(doc, args, proc, model, tok, eos_ids):
    t0 = time.time()
    if args.mode == "prompt":
        fr, meta, fh = QP.extract(doc["videoID"], args.bench)
        prompt, inp = QP.processor_inputs(proc, fr, meta, text=doc["text"], add_generation_prompt=True)
        ph = hashlib.sha1(inp["pixel_values_videos"].float().numpy().tobytes()).hexdigest()[:16]
        inp = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in inp.items()}
        out = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False, temperature=None,
                             top_p=None, top_k=None, output_scores=True, return_dict_in_generate=True)
        gen = out.sequences[0, inp["input_ids"].shape[1]:].tolist()
        top2 = out.scores[0][0].float().topk(2)
        info = dict(frames_sha1=fh, pixels_sha1=ph, prefill_len=inp["input_ids"].shape[1])
    else:
        c = torch.load(QP.emb_path(doc["videoID"], args.bench), weights_only=True)
        dev = model.device
        logits, pkv = lm_forward(model, inputs_embeds=c["emb"][None].to(dev),
                                 position_ids=c["pos"][:, None].to(dev).long())
        pos = c["next_pos"]
        stream = tok(doc["text"] + "<|im_end|>\n<|im_start|>assistant\n").input_ids
        for t in stream:
            p = torch.full((3, 1, 1), pos, device=dev, dtype=torch.long)
            logits, pkv = lm_forward(model, input_ids=torch.tensor([[t]], device=dev), position_ids=p,
                                     past_key_values=pkv)
            pos += 1
        top2 = logits[0, -1].float().topk(2)
        gen, nxt = [], int(logits[0, -1].argmax())
        for _ in range(args.max_new_tokens):
            gen.append(nxt)
            if nxt in eos_ids:
                break
            p = torch.full((3, 1, 1), pos, device=dev, dtype=torch.long)
            logits, pkv = lm_forward(model, input_ids=torch.tensor([[nxt]], device=dev), position_ids=p,
                                     past_key_values=pkv)
            pos += 1
            nxt = int(logits[0, -1].argmax())
        info = dict(frames_sha1=c["frames_sha1"], pixels_sha1=c["pixels_sha1"], ids_sha1=c["ids_sha1"],
                    prefill_len=c["emb"].shape[0], n_stream=len(stream))
    text = tok.decode(gen, skip_special_tokens=True).strip()
    info.update(n_gen=len(gen), margin=(top2.values[0] - top2.values[1]).item(), sec=time.time() - t0,
                peak_gb=torch.cuda.max_memory_allocated() / 2**30)
    return text, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prompt", "hfstream"], required=True)
    ap.add_argument("--bench", default="videomme", choices=["videomme", "lvbench"])
    ap.add_argument("--durations", nargs="+", default=["short", "medium", "long"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--qids", nargs="+")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--file_tag", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "vision"))
    import importlib.util
    lmms = os.path.join(HERE, "..", "..", "..", "third_party", "lmms-eval")
    spec = importlib.util.spec_from_file_location("mcq", os.path.join(lmms, "lmms_eval", "tasks", "_task_utils", "mcq_extract.py"))
    mcq = importlib.util.module_from_spec(spec); spec.loader.exec_module(mcq)
    import qwen_docs
    df = qwen_docs.load_docs(args)
    if args.limit:
        df = df.iloc[:args.limit]
    if args.qids:
        df = df[df.question_id.isin(args.qids)]
    df = df.iloc[args.shard::args.nshards]
    os.makedirs(args.out, exist_ok=True)
    fn = os.path.join(args.out, f"shard{args.shard}of{args.nshards}{args.file_tag}.jsonl")
    done = set(json.loads(l)["question_id"] for l in open(fn) if l.endswith("\n")) if os.path.exists(fn) else set()
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
    proc, model = QP.load_model()
    tok = proc.tokenizer
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    with open(fn, "a") as f:
        for _, r in df.iterrows():
            if r.question_id in done:
                continue
            doc = r.to_dict()
            text, info = answer(doc, args, proc, model, tok, eos_ids)
            info["host"], info["gpu"] = socket.gethostname(), torch.cuda.get_device_name()
            pred = mcq.extract_mcq_answer(text, choices=["A", "B", "C", "D"])
            rec = dict(question_id=doc["question_id"], videoID=doc["videoID"], group=doc.get("group"),
                       task_type=doc.get("task_type"), answer=doc["answer"], output=text, pred=pred,
                       score=float(pred.lower() == str(doc["answer"]).lower()),
                       **{k: doc[k] for k in ("time_reference", "duration_minutes") if k in doc}, **info)
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(doc["question_id"], repr(text), doc["answer"], f"{info['sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
