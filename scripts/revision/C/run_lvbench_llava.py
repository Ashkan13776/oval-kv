"""LVBench (C2) with LLaVA-Video-7B-Qwen2: vision/run_videomme.py's model loading, harness patch,
stream protocol and answer() unchanged; only the question source (scripts/revision/C/lvbench.py),
the prompt text (lmms-eval lvbench_doc_to_text) and the frame cache ($LVB_ROOT/frames, built
with vision/frames.py's extract: 64 frames, 384x384) differ. The llava_vid time instruction is
kept, as in lmms-eval's llava_vid wrapper.

    python scripts/revision/C/run_lvbench_llava.py --protocol stream --frames 64 --out DIR <harness flags>
"""
import json, os, socket, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "vision")); sys.path.insert(0, HERE)
import torch
import run_videomme as RV
import lvbench as LVB
import sellog                                  # C5 selection maps (read-only proxy around torch.topk)
from llava.constants import IMAGE_TOKEN_INDEX
_CAP = {}
_build_inputs = RV.build_inputs


def _bi(*a, **k):                              # capture the prompt ids (position of the video span)
    r = _build_inputs(*a, **k); _CAP["ids"] = r[0]; return r


RV.build_inputs = _bi

RV.CACHE = os.path.join(LVB.LVB_ROOT, "frames")
_TEXT = {}
RV.doc_to_text = lambda doc: doc["text"]


def main():
    args = RV.build_parser().parse_args()
    df = LVB.load_docs()
    if args.limit:
        df = df.iloc[:args.limit]
    if args.qids:
        df = df[df.question_id.isin(args.qids)]
    df = df.iloc[args.shard::args.nshards]
    os.makedirs(args.out, exist_ok=True)
    fn = os.path.join(args.out, f"shard{args.shard}of{args.nshards}{args.file_tag}.jsonl")
    done = set(json.loads(l)["question_id"] for l in open(fn) if l.endswith("\n")) if os.path.exists(fn) else set()
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
    tok, model, proc = RV.load_llava(args)
    step_updater, eos_ids = None, None
    if args.protocol != "official":
        model, step_updater, eos_ids = RV.patch_harness(args, tok, model)
        sellog.install(args.budget // args.page_size)
    with open(fn, "a") as f:
        for _, r in df.iterrows():
            if r.question_id in done:
                continue
            doc = r.to_dict()
            while not os.path.exists(os.path.join(RV.CACHE, f"{doc['videoID']}_n{RV.N_CACHE}", "meta.json")):
                time.sleep(30)
            sellog.start()          # the dense prefill never reaches the page-selection call
            text, info = RV.answer(doc, args, tok, model, proc, step_updater, eos_ids)
            p = int((_CAP["ids"][0] == IMAGE_TOKEN_INDEX).nonzero()[0])
            sl = sellog.frames(args.sink, args.page_size, p, p + info["n_visual"], n_frames=args.frames)
            if sl is not None:
                info["sellog"] = dict(sl, frame_unit="frame")
            info["host"], info["gpu"] = socket.gethostname(), torch.cuda.get_device_name()
            pred = RV.extract_letter(text)
            rec = dict(question_id=doc["question_id"], videoID=doc["videoID"], group=doc["group"],
                       task_type=doc["task_type"], time_reference=doc["time_reference"],
                       duration_minutes=doc["duration_minutes"], answer=doc["answer"], output=text, pred=pred,
                       score=float(pred.lower() == doc["answer"].lower()), **info)
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(doc["question_id"], repr(text), doc["answer"], f"{info['sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
