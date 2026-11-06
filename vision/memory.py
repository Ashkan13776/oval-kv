"""KV memory per row, the accounting of the paper's Fig. 2, at the actual Video-MME lengths.

Per KV head per layer, bf16, d = 128 (one token of keys or of values = 256 B), page 32:
  exact tokens      sink + recent (128 + 128) keep exact keys and values in every row;
  paged region      every complete page after the sink, P = floor((L - sink) / 32), carries
                    the row's per-page state (as in Fig. 2: digests / bases are written when
                    a page completes, also inside the recent window);
  FullKV            2 * L * 256
  FreeKV, Quest     2 * L * 256 + P * 512                 (exact cache + min/max digest)
  4-bit keys        exact V + exact sink/recent K + P * 2560 (4-bit page + per-channel min/max)
  OVAL keys         exact V + exact sink/recent K + P * 2816 (key basis (d r + m r + d) * 2, r = 8)
  OVAL full         exact sink/recent K,V + P * (2816 + 5376) (+ value basis (d r_v + m r_v + d) * 2, r_v = 16)
  ArkVale           as FreeKV (centre +- spread digest, 512 B)
  SnapKV, RaaS      eviction (Fig. 2): K,V of the min(L, 2048) retained tokens; RaaS also a
                    516 B digest per retained page. SnapKV is run without eviction (it keeps
                    the cache and stops attending), so its accuracy is an upper bound.
  Sliding window    K,V of the 128 sink + 1920 recent tokens
L is each sample's final length (prefill + question + answer); reported as the mean over
samples. "Model" adds layer 0, which the harness keeps dense (exact) for every rule.

    python vision/memory.py RUN_DIR [RUN_DIR ...]   (first dir = FullKV-N reference)
"""
import glob, json, sys

D, PAGE, SINK, RECENT, TOK = 128, 32, 128, 128, 256
KB, VB, MM, Q4 = 2816, 5376, 512, 2560
N_LAYERS = 28


def per_layer(rule, L):
    P = (L - SINK) // PAGE
    exact_kv = 2 * (SINK + RECENT) * TOK
    if rule == "full":
        return 2 * L * TOK
    if rule in ("snapkv", "raas", "sliding"):
        kept = min(L, 2048)
        return 2 * kept * TOK + (kept // PAGE * 516 if rule == "raas" else 0)
    if rule in ("freekv", "quest", "arkvale"):
        return 2 * L * TOK + P * MM
    if rule == "q4key":
        return L * TOK + (SINK + RECENT) * TOK + P * Q4
    if rule == "oval_keys":
        return L * TOK + (SINK + RECENT) * TOK + P * KB
    if rule == "oval_full":
        return exact_kv + P * (KB + VB)
    raise ValueError(rule)


def rule_of(args):
    if args["method"] == "full":
        return "full"
    if args["method"] == "raas":
        return "raas"
    if args["method"] == "arkv":
        return "arkvale"
    if args["budget"] == 0:
        return "sliding"
    rep = args["page_rep"]
    if rep == "oval":
        return "oval_full" if args.get("oval_vrank") else "oval_keys"
    if rep == "q4key":
        return "q4key"
    if rep == "snapkv":
        return "snapkv"
    return "freekv" if args["method"] == "spec_ret" else "quest"


def lengths(d):
    out = []
    for f in glob.glob(f"{d}/*.jsonl"):
        for l in open(f):
            if l.endswith("\n"):
                r = json.loads(l)
                out.append(r["prefill_len"] + r.get("n_stream", 0) + r["n_gen"])
    return out


def row(d):
    a = json.load(open(f"{d}/args.json"))
    rule = rule_of(a)
    Ls = lengths(d)
    layer = sum(per_layer(rule, L) for L in Ls) / len(Ls)
    model = sum(per_layer(rule, L) * (N_LAYERS - 1) + per_layer("full", L) for L in Ls) / len(Ls)
    return rule, a["frames"], sum(Ls) / len(Ls), layer, model


if __name__ == "__main__":
    ref = None
    for d in sys.argv[1:]:
        rule, fr, L, layer, model = row(d)
        if ref is None:
            ref = (layer, model)
        print(f"{d}: {rule:<10} frames {fr:>3}  mean L {L:8.0f}  {layer / 1024:8.1f} KiB/head/layer "
              f"({100 * layer / ref[0]:5.1f}%)  model {100 * model / ref[1]:5.1f}%")
