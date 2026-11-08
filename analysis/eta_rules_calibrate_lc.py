"""Rule-1 calibration for LONG CONTEXT (LongBench v2, LongGenBench), per (model,
benchmark), per (layer, KV head). Piece 6: docs/6-long-context.md.

The measurement core is the reasoning calibration's, imported unchanged from
analysis/eta_rules_calibrate.py (window_stats, Acc, rule1 via oval/eta_rules.py,
write_table). Only the data and the capture differ, so the reasoning agents'
calibration cannot change:

  * held-out data. LongBench v2 is evaluated in full (503 questions), so it is
    calibrated on LongBench-Pro (English; 3 documents in each of the 16k / 32k
    / 64k / 128k buckets, 2 calibrate + 1 test per bucket), framed like
    LongBench v2's 0-shot prompt and middle-truncated to 65536 tokens exactly
    as eval/LongBench2/pred.py does. LongGenBench is evaluated on
    Dataset_short.json (400 items); it is calibrated on Dataset_long.json from
    the LongGenBench repo (same four templates, different items; no prompt in
    common), 6 prompts, 4 calibrate + 2 test.
  * query positions. The harness prefill is dense; pages are selected and
    rebuilt only on decode steps. LongBench v2: one window at the end of the
    prompt, ending on a page boundary after 64 greedy answer tokens, so its 96
    queries are the decode steps plus the last prompt tokens (the question),
    attending to the full (up to 64K) context. LongGenBench: the reasoning
    recipe -- 6144 sampled tokens (T 0.95, top-p 1.0 as FreeKV's command,
    seed 7), 3 windows of 96 generated positions, stride 1024.
  * page layout = the benchmark's harness protocol: LongBench v2 sink 128,
    recent 128, budget 1792; LongGenBench sink 512, recent 512, budget 1024;
    page 32 for both.
  * streamed. A 64K capture of K/V for every layer is 3.8-12.9 GB per prompt,
    so nothing is saved: each prompt is captured on the GPU, measured, and its
    accumulators are checkpointed (resumable per prompt).

Also measured on the first prompts (--rankcheck_n): the relative output error
of key ranks {4, 8, 16} and value ranks {8, 16, 31}, and of retrieval alone
(exact keys and values), at eta 0.5 -- the check that the reasoning operating
points still hold at 64K (compare LongBench2 against the LongGenBench windows).

Usage (repo root, harness env):
  python analysis/eta_rules_calibrate_lc.py stats --model qwen-2.5-chat-7b --bench LongBench2
  python analysis/eta_rules_calibrate_lc.py stats --model qwen-2.5-chat-7b --bench LongGenBench
  python analysis/eta_rules_calibrate_lc.py rules --models qwen-2.5-chat-7b --benches LongBench2 LongGenBench
Outputs (the driver passes --out results/long_context/eta_rules/kr<K>_vr<V>):
  <out>/<model>_<bench>.json, <out>/tables/<model>_<bench>_r1.json
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eta_rules_calibrate as C           # the reasoning calibration, used read-only
from paths import ROOT, FREEKV

PB, R = C.PB, C.R
FK_ACC = FREEKV / "accuracy"
MODEL2PATH = json.load(open(FREEKV / "config/model2path.json"))
MAXLEN = json.load(open(FREEKV / "config/model2maxlen.json"))
MODELS = ["qwen-2.5-chat-7b", "llama-3.1-chat-8b", "qwen-2.5-chat-14b"]
BENCHES = ["LongBench2", "LongGenBench"]
LAYOUT = {"LongBench2": dict(sink=128, recent=128, budget=1792),
          "LongGenBench": dict(sink=512, recent=512, budget=1024)}
LGB_LONG = os.environ.get("LGB_LONG_FILE", str(ROOT / "logs/long_context/calib_data/LongGenBench_Dataset_long.json"))
LB_PRO_REPO = "caskcsg/LongBench-Pro"
LB_PRO_BUCKETS = ["16k", "32k", "64k", "128k"]


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stats", "rules"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--bench", default=None, choices=BENCHES)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--benches", nargs="+", default=BENCHES)
    ap.add_argument("--out", default=str(ROOT / "results/long_context/eta_rules"))
    ap.add_argument("--ckpt_dir", default=str(ROOT / "logs/long_context/calib_ckpt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model_path", default=None, help="HF path override (tests)")
    ap.add_argument("--nprob", type=int, default=None, help="default 12 (LongBench2) / 6 (LongGenBench)")
    ap.add_argument("--gen_seed", type=int, default=7)
    ap.add_argument("--ngen", type=int, default=None, help="default 64 (LongBench2) / 6144 (LongGenBench)")
    ap.add_argument("--nwin", type=int, default=None, help="default 1 (LongBench2) / 3 (LongGenBench)")
    ap.add_argument("--win_stride", type=int, default=1024)
    ap.add_argument("--nq", type=int, default=96)
    ap.add_argument("--page", type=int, default=32)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--vrank", type=int, default=32)
    ap.add_argument("--ngrid", type=int, default=2)
    ap.add_argument("--rankcheck_n", type=int, default=None, help="prompts with the rank check; default 4 / 2")
    ap.add_argument("--layers", default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke: only the first N prompts, tagged output")
    ap.add_argument("--pick", default=None, help="smoke: comma-separated prompt indices, tagged output")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.bench:
        a.__dict__.update({k: v for k, v in LAYOUT[a.bench].items()})
        lb2 = a.bench == "LongBench2"
        a.nprob = a.nprob or (12 if lb2 else 6)
        a.ncalib = 8 if lb2 else 4
        a.ngen = a.ngen or (64 if lb2 else 6144)
        a.nwin = a.nwin or (1 if lb2 else 3)
        a.rankcheck_n = 4 if a.rankcheck_n is None and lb2 else (2 if a.rankcheck_n is None else a.rankcheck_n)
        a.task = a.bench                  # names used by the imported helpers
    if a.limit or a.pick:
        a.tag = a.tag or f"_smoke{a.limit or a.pick.replace(',', '-')}"
    return a


log = C.log
now = C.now


# =========================================================================
# held-out prompts
# =========================================================================
LB2_FRAME = ("Please read the following text and answer the question below.\n\n<text>\n{doc}\n</text>\n\n"
             "{q}\n\nAnswer the question based on the text above.")


def lb_pro_items(n):
    """3 English documents per length bucket (16k..128k), spread over primary
    tasks, deterministic. Split per bucket: first 2 calibrate, third tests."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(LB_PRO_REPO, "longbench_pro.json", repo_type="dataset")
    d = [e for e in json.load(open(p)) if e["language"] == "English"]
    per = max(1, n // len(LB_PRO_BUCKETS))
    picked = {b: [] for b in LB_PRO_BUCKETS}
    for b in LB_PRO_BUCKETS:
        rows = sorted([e for e in d if e["token_length"] == b], key=lambda e: e["id"])
        seen = set()
        for e in rows:
            if e["primary_task"] in seen:
                continue
            seen.add(e["primary_task"]); picked[b].append(e)
            if len(picked[b]) == per:
                break
    # order: calib items of every bucket first, then the test items
    calib = [e for b in LB_PRO_BUCKETS for e in picked[b][:per - 1]]
    test = [e for b in LB_PRO_BUCKETS for e in picked[b][per - 1:]]
    out = calib + test
    return [(e["id"][:12], LB2_FRAME.format(doc=e["context"].strip(), q=e["question_nonthinking"].strip()),
             e["token_length"]) for e in out], len(calib)


def lgb_long_items(n):
    d = json.load(open(LGB_LONG))
    short = {x["prompt"] for x in json.load(open(FK_ACC / "eval/LongGenBench/Dataset_short.json"))}
    idx = [int(round(i * len(d) / n)) for i in range(n)]       # spread over the 4 types
    items = [(f"long{i}", d[i]["prompt"], d[i]["type"]) for i in idx]
    assert not any(p in short for _, p, _ in items), "calibration prompt overlaps the evaluated set"
    return items, None


def held_out(args):
    if args.bench == "LongBench2":
        items, nc = lb_pro_items(args.nprob)
        args.ncalib = nc
    else:
        items, _ = lgb_long_items(args.nprob)
    if args.limit:
        items = items[:args.limit]
    if args.pick:
        items = [items[int(i)] for i in args.pick.split(",")]
    return items


# =========================================================================
# capture (in memory) + stats, one prompt at a time
# =========================================================================
def harness_ids(tok, prompt, model, bench):
    """Token ids exactly as the harness builds them (pred.py truncation + util.build_chat)."""
    if bench == "LongBench2":
        ids = tok.encode(prompt)
        ml = MAXLEN[model]
        if len(ids) > ml:
            ids = ids[:ml // 2] + ids[-ml // 2:]
            prompt = tok.decode(ids, skip_special_tokens=True)
    if "llama" in model:
        return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, return_tensors="pt")
    msgs = [{"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba."},
            {"role": "user", "content": prompt}]
    return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
               truncation=False, return_tensors="pt").input_ids


def windows_for(T, prompt_len, args):
    if args.bench == "LongBench2":
        L = args.sink + ((T - args.sink) // args.page) * args.page
        return [L] if L - args.sink - args.recent >= args.page else []
    wins = []
    for k in range(args.nwin):
        L = args.sink + ((T - k * args.win_stride - args.sink) // args.page) * args.page
        if L - args.sink - args.recent >= args.page and L - args.nq >= prompt_len:
            wins.append(L)
    return sorted(set(wins))


@torch.no_grad()
def capture_one(model, tok, prompt, args):
    """Generate, then one forward of the whole sequence through the base model
    (no 64K x vocab logits), hooking q_proj at the window positions.
    Returns K, V [nl] each [nkv, T, hd] bf16 on GPU, Q {li: [W, nq, nh, hd]}, windows."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    dev = args.device
    ids = harness_ids(tok, prompt, args.model, args.bench).to(dev)
    P = ids.shape[-1]
    torch.manual_seed(args.gen_seed); torch.cuda.manual_seed_all(args.gen_seed)
    kw = (dict(do_sample=False) if args.bench == "LongBench2"
          else dict(do_sample=True, temperature=0.95, top_p=1.0))
    seq = model.generate(ids, max_new_tokens=args.ngen, min_new_tokens=args.ngen,
                         pad_token_id=tok.eos_token_id, **kw)
    T = seq.shape[-1]
    wins = windows_for(T, P, args)
    if not wins:
        return None
    qpos = torch.cat([torch.arange(L - args.nq, L) for L in wins]).to(dev)
    cfg = model.config
    nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = cfg.hidden_size // nh
    qcap = {}
    layers = model.model.layers
    hooks = [l.self_attn.q_proj.register_forward_hook(
                lambda m, i, o, li=li: qcap.__setitem__(li, o[0, qpos].detach()))
             for li, l in enumerate(layers)]
    out = model.model(seq, use_cache=True)
    for h in hooks: h.remove()
    pkv = out.past_key_values
    cos, sin = model.model.rotary_emb(qcap[0][None].float(), qpos[None])
    K, V, Q = [], [], {}
    for li in range(len(layers)):
        k_, v_ = (pkv[li][0][0], pkv[li][1][0]) if not hasattr(pkv, "key_cache") else (pkv.key_cache[li][0], pkv.value_cache[li][0])
        K.append(k_.to(torch.bfloat16)); V.append(v_.to(torch.bfloat16))
        q = qcap[li].float().view(1, -1, nh, hd).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, q, cos, sin)
        Q[li] = q[0].transpose(0, 1).reshape(len(wins), args.nq, nh, hd)
    del out, pkv, qcap
    return dict(K=K, V=V, Q=Q, windows=wins, T=T, prompt_len=P,
                n_dec_in_window=[max(0, L - P) for L in wins])


def rank_check_window(li, q, K, V, L, qpos, Wh, Gh, args, eta=0.5):
    """Relative output error at eta 0.5 for: retrieval only (exact K, V), key
    ranks {4, 8, 16} (exact V), and keys rank 8 with value ranks {8, 16, 31}."""
    H, g, nq, d = q.shape
    P, lo, hi = C.page_layout(L, args)
    bp = args.budget // args.page
    lg = torch.einsum('hgqd,htd->hgqt', q, K) / math.sqrt(d)
    lg = lg.masked_fill(torch.arange(L, device=q.device)[None, None, None, :] > qpos[None, None, :, None], float('-inf'))
    u = C.group_out(Wh, torch.einsum('hgqt,htd->hgqd', torch.softmax(lg, -1), V))
    u2 = float((u * u).sum())
    Kp = K[:, lo:hi].reshape(H, P, args.page, d).permute(1, 2, 0, 3)[None].contiguous()
    Vp = V[:, lo:hi].reshape(H, P, args.page, d).permute(1, 2, 0, 3)[None].contiguous()
    flat = lambda X: X.float()[0].permute(1, 0, 2, 3).reshape(H, P * args.page, d)
    Kex, Vex = K[:, lo:hi], V[:, lo:hi]
    vhat = {}
    for vr in (8, 16, 31):
        vmu, vD, vT = PB.build_value_summary(Kp, Vp, None, None, 0.0, vr)
        vhat[vr] = flat(vT.float() @ vD.float().transpose(-1, -2) + vmu.float().unsqueeze(-2))
    res = {}
    for kr in (4, 8, 16):
        mu, bs, cf = PB.build_summary(Kp, Vp, Gh, eta, kr)
        sel = C.select_pages(q, mu, bs, cf, H, g, bp)
        khat = flat(cf.float() @ bs.float().transpose(-1, -2) + mu.float().unsqueeze(-2))
        err = lambda Kh, Vh: float(((u - C.group_out(Wh, C.retrieval_loss_terms(q, K, V, L, qpos, Kh, Vh, sel, args))) ** 2).sum())
        if kr == 8:
            res["retrieval_only"] = err(Kex, Vex)
            for vr in (8, 16, 31):
                res[f"k8_v{vr}"] = err(khat, vhat[vr])
        res[f"k{kr}_vexact"] = err(khat, Vex)
    return u2, res


def ckpt_path(args):
    return os.path.join(args.ckpt_dir, f"{args.model}_{args.bench}_kr{args.rank}_vr{args.vrank}{args.tag}.pt")


def stage_stats(args):
    dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True); os.makedirs(args.ckpt_dir, exist_ok=True)
    out_json = os.path.join(args.out, f"{args.model}_{args.bench}{args.tag}.json")
    if os.path.exists(out_json) and not args.overwrite:
        log(f"stats: {out_json} exists; skipping"); return
    items = held_out(args)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = args.model_path or MODEL2PATH[args.model]
    log(f"stats: {args.model}/{args.bench}: {len(items)} prompts ({args.ncalib} calibrate), layout sink {args.sink} "
        f"recent {args.recent} budget {args.budget} page {args.page}, key rank {args.rank}, value rank {args.vrank}")
    try:
        tok = AutoTokenizer.from_pretrained(path, use_fast=False)
    except Exception:
        tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(dev).eval()
    cfg = model.config
    nl, nh, nkv = len(model.model.layers), cfg.num_attention_heads, cfg.num_key_value_heads
    hd = cfg.hidden_size // nh; D = cfg.hidden_size; g = nh // nkv
    Wh = [l.self_attn.o_proj.weight.detach().float().view(D, nkv, g, hd) for l in model.model.layers]
    Gh = [PB.publish_metric(l.self_attn.o_proj.weight.detach().float(), nkv, hd, g) for l in model.model.layers]
    layers = C.layer_subset(args.layers, nl)
    grid = [round(i / (args.ngrid - 1), 6) for i in range(args.ngrid)]
    ck = ckpt_path(args)
    if os.path.exists(ck) and not args.overwrite:
        st = torch.load(ck, weights_only=False)
        acc, meta, rc = st["acc"], st["meta"], st["rankcheck"]
        log(f"stats: resuming from {ck}: {len(meta)} prompts done")
    else:
        acc, meta, rc = C.Acc(nl, nkv, len(grid)), {}, []
    t0 = time.time()
    for k, (pid, prompt, kind) in enumerate(items):
        if pid in meta:
            continue
        split = "calib" if k < args.ncalib else "test"
        tc = time.time()
        cap = capture_one(model, tok, prompt, args)
        if cap is None:
            log(f"stats: {pid} too short for a window; skipped"); meta[pid] = {"skipped": True}; continue
        tcap = time.time() - tc
        do_rc = k < args.rankcheck_n
        rck = {"pid": pid, "kind": kind, "T": cap["T"], "u2": np.zeros(nl), "err": {}}
        for wi, L in enumerate(cap["windows"]):
            qpos = torch.arange(L - args.nq, L, device=dev)
            for li in layers:
                K = cap["K"][li][:, :L].float(); V = cap["V"][li][:, :L].float()
                q = cap["Q"][li][wi].float().permute(1, 0, 2).reshape(nkv, g, args.nq, hd)
                C.window_stats(acc, split, li, q, K, V, L, qpos, Wh[li], Gh[li], grid, args)
                if do_rc:
                    u2, e = rank_check_window(li, q, K, V, L, qpos, Wh[li], Gh[li], args)
                    rck["u2"][li] += u2
                    for kk, vv in e.items():
                        rck["err"].setdefault(kk, np.zeros(nl))[li] += vv
        if do_rc:
            rc.append(rck)
        meta[pid] = {"split": split, "kind": kind, "T": cap["T"], "prompt_len": cap["prompt_len"],
                     "windows": cap["windows"], "n_dec_in_window": cap["n_dec_in_window"],
                     "capture_s": round(tcap, 1), "total_s": round(time.time() - tc, 1)}
        del cap; torch.cuda.empty_cache()
        torch.save({"acc": acc, "meta": meta, "rankcheck": rc}, ck)
        log(f"stats: {args.model}/{args.bench} {pid} ({split}, {kind}) T={meta[pid]['T']} windows={meta[pid]['windows']} "
            f"capture {tcap:.0f}s, total {meta[pid]['total_s']:.0f}s, {time.time() - t0:.0f}s so far")
    L_grid = {sp: acc.d2[sp] / np.maximum(acc.u2[sp][..., None], 1e-300) for sp in ("calib", "test")}
    eta1, deg1 = R.rule1_endpoint(acc.a, acc.b, acc.c)
    rcs = {}
    for r in rc:
        for kk, v in r["err"].items():
            rel = v / np.maximum(r["u2"], 1e-300)
            rcs.setdefault(kk, []).append(float(rel[1:].mean()))
    res = {
        "model": args.model, "task": args.bench, "bench": args.bench, "created": now(),
        "config": {k: v for k, v in vars(args).items() if k != "stage"},
        "n_layers": nl, "n_kv_heads": nkv, "n_heads": nh, "head_dim": hd, "layers_done": layers,
        "grid": grid, "prompts": meta,
        "loss_grid": {sp: L_grid[sp].tolist() for sp in L_grid},
        "n_queries": {sp: acc.nq[sp].tolist() for sp in acc.nq},
        "best_grid_eta": {sp: np.array(grid)[L_grid[sp].argmin(-1)].tolist() for sp in L_grid},
        "rules": {"r1": {"eta": eta1.numpy().tolist(), "degenerate": deg1.numpy().tolist(),
                         "a": acc.a.tolist(), "b": acc.b.tolist(), "c": acc.c.tolist(), "default": 0.5}},
        "rank_check": {"eta": 0.5, "prompts": [r["pid"] for r in rc], "context_T": [r["T"] for r in rc],
                       "rel_err_layers1plus": rcs},
    }
    json.dump(res, open(out_json, "w"))
    log(f"stats: wrote {out_json} in {time.time() - t0:.0f}s")
    C._print_eta_summary(res)
    for kk, v in rcs.items():
        log(f"rank check {kk:<15} rel. output error {np.mean(v):.4f} (per prompt {['%.4f' % x for x in v]})")


def stage_rules(args):
    for m in args.models:
        for b in args.benches:
            f = os.path.join(args.out, f"{m}_{b}{args.tag}.json")
            if not os.path.exists(f):
                continue
            st = json.load(open(f))
            rr = st["rules"]["r1"]
            p, h = C.write_table(args, m, b, "r1", rr["eta"], {"degenerate_count": int(np.sum(rr["degenerate"]))})
            log(f"rules: {m}/{b} r1 -> {p} hash {h}")
            C._print_eta_summary(st)


if __name__ == "__main__":
    args = parse()
    if args.gen_seed in range(42, 50):
        raise SystemExit("gen_seed must be outside the evaluated seeds 42..49")
    if args.stage == "stats" and (args.model is None or args.bench is None):
        raise SystemExit("--model and --bench are required")
    torch.backends.cuda.matmul.allow_tf32 = False
    {"stats": stage_stats, "rules": stage_rules}[args.stage](args)
