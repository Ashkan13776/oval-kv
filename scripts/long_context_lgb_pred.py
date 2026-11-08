"""LongGenBench prediction for the long-context pipeline (piece 6), as
eval/LongGenBench/pred.py does it, plus three things that script lacks:

  * real seeds: pred.py never seeds torch (its --seed only renames the file).
    Here item i is sampled under torch seed  SEED * 1000003 + i, so a run is
    reproducible and independent of sharding or restarts.
  * resume: pred.py appends to its output with no resume, so a restart
    duplicates records. Here a shard skips the items its part file holds, and a
    part file broken by a kill mid-write is restored from its last good copy.
  * shards: LGB_SHARD="k/n" runs items i with i % n == k (interleaved, so every
    shard sees all four types) into <out>.part<k>of<n>.json; --merge writes the
    canonical file in item order. n = 1 writes the canonical file directly.

Everything else -- prompts, chat template, sampling, stop string, output
record -- is pred.py's own code (imported). Run from third_party/FreeKV/accuracy
with pred.py's arguments:
  LGB_SHARD=0/2 python $KV_ROOT/scripts/long_context_lgb_pred.py --model ... [pred.py args]
  python $KV_ROOT/scripts/long_context_lgb_pred.py --merge <canonical .jsonl path> <n>
"""
import json, os, shutil, sys

sys.path.insert(0, os.getcwd())


def merge(out, n, total=400):
    """Interleave the n part files into the canonical file, item order."""
    parts = [json.load(open(out.replace(".jsonl", f".part{k}of{n}.json"))) for k in range(n)]
    merged = [parts[i % n][i // n] for i in range(total)]
    json.dump(merged, open(out, "w"), indent=4)
    print(f"merged {n} parts -> {out} ({len(merged)} records)")


if len(sys.argv) > 1 and sys.argv[1] == "--merge":
    merge(sys.argv[2], int(sys.argv[3]))
    sys.exit(0)

import torch
import eval.LongGenBench.pred as P
from eval.util import get_out_path, load_model_and_tokenizer


def canonical_path(args, config):
    out = get_out_path(args, config, f"eval/LongGenBench/results_{args.input_type}", False)
    head, tail = os.path.split(out)
    parent, last = os.path.split(head)
    return os.path.join(parent, last + tail)          # the same rename pred.py does


def read_list(path):
    for p in (path, path + ".bak"):
        if os.path.exists(p):
            try:
                d = json.load(open(p))
                if p != path:
                    print(f"restored {path} from its .bak ({len(d)} records)")
                    shutil.copyfile(p, path)
                return d
            except json.JSONDecodeError:
                continue
    return []


def main():
    args = P.parse_args()
    inputs = P.load_inputs(f"eval/LongGenBench/Dataset_{args.input_type}.json")
    k, n = map(int, os.environ.get("LGB_SHARD", "0/1").split("/"))
    idx = [i for i in range(len(inputs)) if i % n == k]
    model, tok, su, eos, config = load_model_and_tokenizer(P.model_map[args.model], args)
    stop = tok(P.stop_sign, truncation=False).input_ids
    out = canonical_path(args, config)
    part = out if n == 1 else out.replace(".jsonl", f".part{k}of{n}.json")
    os.makedirs(os.path.dirname(part), exist_ok=True)
    done = read_list(part)
    print(f"LGB shard {k}/{n}: {len(idx)} items, {len(done)} already done -> {part}", flush=True)
    for j, i in enumerate(idx):
        if j < len(done):
            continue
        s = args.seed * 1000003 + i
        torch.manual_seed(s); torch.cuda.manual_seed_all(s)
        P.get_pred([inputs[i]], [inputs[i]["prompt"]], args.model, model, tok, su, eos,
                   args.max_gen, args.temperature, args.top_p, stop, part)
        shutil.copyfile(part, part + ".bak")
        print(f"LGB item {i} done ({j + 1}/{len(idx)})", flush=True)
    print(f"LGB shard {k}/{n} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
