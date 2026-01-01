"""InfiniteBench inside FreeKV's accuracy harness.

FreeKV's paper does not evaluate on InfiniteBench (their benchmarks are
LongBench v2, LongGenBench, MATH500, AIME24, GPQA), so this evaluator does not
exist upstream. It is added here so FreeKV's own page representation and our
OVAL basis can be compared on InfiniteBench INSIDE ONE HARNESS -- the control
that proved decisive on the LongBench v2 Qwen-14B cell, where our harness
reproduced FreeKV's own method 1.59 points low and a cross-harness comparison
would have been misleading.

Protocol is carried over verbatim from our LOCKS-side runs
(kv-compression/harness/run_infinitebench.py) so the two are comparable in
everything except the model:

  * 11 tasks, math_calc excluded (as in the LOCKS paper's Table 1)
  * seeded RANDOM 50 records per task, seed 42 -- NOT first-n. The .jsonl files
    are pre-sorted (kv_retrieval by needle position), so first-n is badly
    biased; see notes/finding-kv-retr.md.
  * InfiniteBench reference prompt templates and per-task max_new_tokens
  * middle-truncation, head half + tail half
  * official InfiniteBench scorers, run unmodified on the dumped predictions

MODEL CAVEAT: our LOCKS InfiniteBench runs used GLM-4-9B-Chat-1M, which this
harness CANNOT load -- it patches transformers' llama/mistral/qwen2 attention
and GLM-4 is ChatGLMModel via trust_remote_code. So this runs on a supported
long-context model instead, and BOTH arms (quest and oval) must be run here for
the comparison to be valid.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from eval.util import parse_common_args, build_chat, load_model_and_tokenizer, get_out_path

# our LOCKS-side protocol helpers
_KVC = Path("KV_ROOT")
sys.path.insert(0, str(_KVC / "harness"))
sys.path.insert(0, str(_KVC / "vendor/InfiniteBench/src"))
from subset import DEFAULT_SEED, load_subset  # noqa: E402

TASKS = [
    "passkey", "number_string", "kv_retrieval",
    "longdialogue_qa_eng", "longbook_sum_eng", "longbook_choice_eng",
    "longbook_qa_eng", "longbook_qa_chn",
    "math_find", "code_run", "code_debug",
]


def resolve_data_dir() -> Path:
    base = Path(os.environ.get(
        "IB_DATA_DIR",
        Path.home() / ".cache/huggingface/hub/datasets--xinrongzhang2022--InfiniteBench",
    ))
    if (base / "passkey.jsonl").exists():
        return base
    snaps = sorted((base / "snapshots").glob("*"))
    if not snaps:
        raise SystemExit(f"no InfiniteBench snapshot under {base}")
    return snaps[0]


def truncate_middle(tokens, max_length):
    """Reference policy (eval_utils.py:553) -- head half + tail half."""
    if len(tokens) <= max_length:
        return tokens, False
    half = max_length // 2
    return tokens[:half] + tokens[-half:], True


def main():
    p = argparse.ArgumentParser()
    parse_common_args(p)
    p.add_argument("--tasks", nargs="*", default=TASKS)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--ib_seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max_input", type=int, default=None,
                   help="truncate context to this many tokens "
                        "(default: model's max_position_embeddings minus headroom)")
    p.add_argument("--protocol_map", type=str,
                   default=str(_KVC / "harness/protocol_map.json"))
    args = p.parse_args()

    from eval_utils import DATA_NAME_TO_MAX_NEW_TOKENS, create_prompt, get_answer

    model_map = json.loads(
        open(Path(__file__).resolve().parents[3] / "config/model2path.json").read())
    path = model_map[args.model]
    model, tokenizer, step_updater, eos_token_ids, config = \
        load_model_and_tokenizer(path, args)

    max_input = args.max_input or (model.config.max_position_embeddings - 2048)
    pmap = json.loads(open(args.protocol_map).read())["tasks"]
    data_dir = resolve_data_dir()

    out_root = args.out_root_dir or "eval/InfiniteBench/results"
    os.makedirs(out_root, exist_ok=True)

    for task in args.tasks:
        out_path = get_out_path(args, config, out_root, True)
        out_path = out_path.replace(".jsonl", f"-{task}.jsonl") \
            if out_path.endswith(".jsonl") else f"{out_path}-{task}.jsonl"
        if os.path.exists(out_path):
            n = sum(1 for l in open(out_path) if l.strip())
            if n >= args.n:
                print(f"=== {task}: already {n} records, skip", flush=True)
                continue
            # their writers append; a partial file must be removed
            print(f"=== {task}: removing partial ({n})", flush=True)
            os.remove(out_path)

        cfg = pmap.get(task, {})
        pmodel = cfg.get("prompt_model", "gpt4")
        use_chat = cfg.get("chat_template", True)
        gen_len = DATA_NAME_TO_MAX_NEW_TOKENS[task] + cfg.get("gen_extra", 0)

        rows = [r for _i, r in load_subset(data_dir / f"{task}.jsonl",
                                           args.n, args.ib_seed)]
        print(f"=== {task}: {len(rows)} records, {pmodel}/chat={use_chat}, "
              f"gen={gen_len}, max_input={max_input}", flush=True)

        n_trunc = 0
        for r in tqdm(rows, desc=task):
            prompt = create_prompt(r, task, pmodel, str(data_dir))
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            ids, tr = truncate_middle(ids, max_input)
            n_trunc += bool(tr)
            text = tokenizer.decode(ids, skip_special_tokens=True)
            chat = build_chat(tokenizer, text, args.model) if use_chat else text
            # build_chat returns EITHER a str OR already-tokenized input,
            # depending on the model -- same branch as their LongBench2 pred.py
            if isinstance(chat, str):
                inp = tokenizer(chat, truncation=False,
                                return_tensors="pt").to("cuda")
            else:
                inp = chat.to("cuda") if hasattr(chat, "to") else chat
            # normalise to a plain input_ids tensor either way
            ids_t = inp.input_ids if hasattr(inp, "input_ids") else inp

            # Their patched forward does not accept `cache_position`, so
            # model.generate() fails; they use a manual greedy loop instead
            # (eval/LongBench2/pred.py:query_llm). Same loop here, verbatim.
            with torch.no_grad():
                if step_updater is not None:
                    step_updater.reset(ids_t)
                out0 = model(input_ids=ids_t, past_key_values=None, use_cache=True)
                pkv = out0.past_key_values
                tok = out0.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                gen = [tok.item()]
                if step_updater is not None:
                    step_updater.update(tok.item())
                for _ in range(gen_len - 1):
                    o = model(input_ids=tok, past_key_values=pkv, use_cache=True)
                    pkv = o.past_key_values
                    tok = o.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    t = tok.item()
                    gen.append(t)
                    if step_updater is not None:
                        step_updater.update(t)
                    if t in eos_token_ids:
                        break
            pred = tokenizer.decode(gen, skip_special_tokens=True)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "id": r.get("id", None),
                    "prediction": pred,
                    "ground_truth": get_answer(r, task),
                }, ensure_ascii=False) + "\n")
        print(f"=== {task}: done, truncated={n_trunc}/{len(rows)} -> {out_path}",
              flush=True)


if __name__ == "__main__":
    main()
