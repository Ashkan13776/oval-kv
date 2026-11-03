"""Offline calibration of eta rule 1, per (model, task), per (layer, KV head).
See docs/4-per-head-eta.md for the definitions.

Stages (each resumable; outputs are never overwritten without --overwrite):

  capture  GPU. Generate NPROB held-out reasoning traces (sampling seed 7, the
           harness prompt format) and save post-RoPE K/V for every layer, the
           queries of NWIN windows of NQ decode positions each (re-rotated at
           their own positions), and the model's o_proj weights.
             -> CAPTURE_DIR/<model>_<task>_p<idx>.pt,  CAPTURE_DIR/<model>_oproj.pt
  stats    GPU. From the captures: the actual retrieval loss L(eta) on a fine
           grid per (layer, head) for the calibration and the test split, and
           rule 1's endpoint sums a, b, c and eta.   -> OUT/<model>_<task>.json
  rules    CPU. Write rule 1's deployable eta table from the stats file.
             -> OUT/tables/<model>_<task>_r1.json
  regret   GPU. Evaluate the table on the TEST split with the harness's own
           per-head build and report the offline regret vs the per-head grid
           optimum. -> OUT/<model>_<task>.json (adds "regret")

Usage (from the repo root; the harness env):
  python analysis/eta_rules_calibrate.py capture --model ds-r1-qwen-7b --task MATH50
  python analysis/eta_rules_calibrate.py stats   --model ds-r1-qwen-7b --task MATH50
  python analysis/eta_rules_calibrate.py rules
  python analysis/eta_rules_calibrate.py regret  --model ds-r1-qwen-7b --task MATH50
"""
import argparse, datetime, importlib.util, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT, FREEKV

def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
PB = _load("page_basis", "oval/page_basis.py")
R = _load("eta_rules", "oval/eta_rules.py")

MODEL2PATH = {
    "ds-r1-qwen-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "ds-r1-llama-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "ds-r1-qwen-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
}
TASK2FILE = {"MATH50": "math50.jsonl", "AIME24": "aime24.jsonl", "GPQA50c": "gpqa50c.jsonl"}
TASK2FIELD = {"MATH50": "problem", "AIME24": "problem", "GPQA50c": "question"}
RULES = ["r1"]
MODELS = ["ds-r1-qwen-7b", "ds-r1-llama-8b"]
TASKS = ["MATH50", "AIME24", "GPQA50c"]


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["capture", "stats", "rules", "regret"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--task", default=None)
    ap.add_argument("--models", nargs="+", default=MODELS, help="rules stage: models")
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--model_path", default=None, help="HF path override (tests)")
    ap.add_argument("--dataset_file", default=None, help="jsonl override (tests)")
    ap.add_argument("--out", default=str(ROOT / "results/eta_rules"))
    ap.add_argument("--capture_dir", default=str(ROOT / "logs/eta_rules/capture"))
    ap.add_argument("--device", default="cuda")
    # calibration data
    ap.add_argument("--nprob", type=int, default=6, help="traces per (model, task)")
    ap.add_argument("--ncalib", type=int, default=4, help="first NCALIB traces calibrate, the rest test")
    ap.add_argument("--gen_seed", type=int, default=7, help="sampling seed; MUST be outside 42..49")
    ap.add_argument("--ngen", type=int, default=6144)
    ap.add_argument("--nwin", type=int, default=3, help="query windows per trace")
    ap.add_argument("--win_stride", type=int, default=1024)
    ap.add_argument("--nq", type=int, default=96, help="queries per window")
    # protocol (the harness's reasoning protocol)
    ap.add_argument("--sink", type=int, default=512)
    ap.add_argument("--recent", type=int, default=512)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--page", type=int, default=32)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--vrank", type=int, default=8)
    ap.add_argument("--ngrid", type=int, default=21)
    ap.add_argument("--layers", default=None, help="subset like 0-2 or 0,5,9 (smoke tests)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for output names (smoke tests)")
    return ap.parse_args()


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def log(*a):
    print(f"[{now()}]", *a, flush=True)


def layer_subset(spec, nl):
    if spec is None:
        return list(range(nl))
    if "-" in spec:
        a, b = spec.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def refuse_overwrite(path, args):
    if os.path.exists(path) and not args.overwrite:
        raise SystemExit(f"{path} exists; pass --overwrite to replace it")


# =========================================================================
# capture
# =========================================================================
def problems(args):
    f = args.dataset_file or str(FREEKV / "accuracy/eval/reasoning/datasets" / TASK2FILE[args.task])
    rows = [json.loads(l) for l in open(f) if l.strip()]
    n = len(rows)
    # spread the calibration problems over the file (the eval runs every problem
    # under seeds 42..49; our traces differ by sampling seed, not by problem)
    idx = [int(round(i * n / args.nprob)) for i in range(args.nprob)]
    idx = sorted(set(min(i, n - 1) for i in idx))
    return [(i, rows[i]) for i in idx]


def harness_prompt(row, task):
    field = TASK2FIELD[task]
    # eval/reasoning/config/dataset2prompt.json + eval/util.py::build_chat for ds-r1 models
    return (row[field] + " Please reason step by step, and put your final answer within \\boxed{}.\n"
            + "<think>\n")


def stage_capture(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    os.makedirs(args.capture_dir, exist_ok=True)
    path = args.model_path or MODEL2PATH[args.model]
    todo = [(i, r) for i, r in problems(args)
            if not os.path.exists(capture_file(args, i))]
    oproj_file = os.path.join(args.capture_dir, f"{args.model}_oproj.pt")
    if not todo and os.path.exists(oproj_file):
        log(f"capture: all {args.nprob} traces for {args.model}/{args.task} exist; nothing to do")
        return
    log(f"capture: loading {path}")
    try:
        tok = AutoTokenizer.from_pretrained(path, use_fast=False)      # as eval/util.py does
    except Exception as e:                                             # test fixtures ship fast files only
        log(f"capture: slow tokenizer unavailable ({type(e).__name__}); using the fast one")
        tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16 if args.device != "cpu" else torch.float32,
        attn_implementation="sdpa").to(args.device).eval()
    cfg = model.config
    nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = cfg.hidden_size // nh
    nl = len(model.model.layers)
    if not os.path.exists(oproj_file):
        torch.save({"o_proj": [l.self_attn.o_proj.weight.detach().cpu() for l in model.model.layers],
                    "n_layers": nl, "n_heads": nh, "n_kv_heads": nkv, "head_dim": hd,
                    "hidden": cfg.hidden_size, "model": args.model, "path": path}, oproj_file)
        log(f"capture: saved o_proj weights -> {oproj_file}")
    for pidx, row in todo:
        text = harness_prompt(row, args.task)
        ids = tok(text, return_tensors="pt").input_ids.to(args.device)
        torch.manual_seed(args.gen_seed); torch.cuda.manual_seed_all(args.gen_seed)
        t0 = time.time()
        with torch.no_grad():
            seq = model.generate(ids, max_new_tokens=args.ngen, min_new_tokens=args.ngen,
                                 do_sample=True, temperature=0.6, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
        T = seq.shape[-1]
        log(f"capture: {args.model}/{args.task} problem {pidx}: prompt {ids.shape[-1]} + {T - ids.shape[-1]} "
            f"generated tokens in {time.time() - t0:.0f}s")
        # windows: ends L_k = largest sink + m*page <= T - k*stride, k = 0..nwin-1
        wins = []
        for k in range(args.nwin):
            L = args.sink + ((T - k * args.win_stride - args.sink) // args.page) * args.page
            if L - args.sink - args.recent >= args.page and L - args.nq >= 0:
                wins.append(L)
        wins = sorted(set(wins))
        qpos = torch.cat([torch.arange(L - args.nq, L) for L in wins]).to(args.device)   # [W*nq]
        qcap = {}
        hooks = [layer.self_attn.q_proj.register_forward_hook(
                    lambda m, i, o, li=li: qcap.__setitem__(li, o[0, qpos].detach()))
                 for li, layer in enumerate(model.model.layers)]
        with torch.no_grad():
            fw = model(seq, use_cache=True)
        for h in hooks: h.remove()
        pkv = fw.past_key_values
        cos, sin = model.model.rotary_emb(qcap[0][None].float(), qpos[None])
        K = torch.empty(nl, nkv, T, hd, dtype=torch.bfloat16)
        V = torch.empty(nl, nkv, T, hd, dtype=torch.bfloat16)
        Q = torch.empty(nl, len(wins), args.nq, nh, hd, dtype=torch.bfloat16)
        for li in range(nl):
            k_, v_ = pkv[li][0][0], pkv[li][1][0]                     # [nkv, T, hd], post-RoPE
            K[li], V[li] = k_.to(torch.bfloat16).cpu(), v_.to(torch.bfloat16).cpu()
            q = qcap[li].float().view(1, -1, nh, hd).transpose(1, 2)   # [1, nh, W*nq, hd]
            q, _ = apply_rotary_pos_emb(q, q, cos, sin)
            Q[li] = q[0].transpose(0, 1).reshape(len(wins), args.nq, nh, hd).to(torch.bfloat16).cpu()
        del fw, pkv, qcap
        torch.save({"K": K, "V": V, "Q": Q, "windows": wins, "nq": args.nq, "T": T,
                    "prompt_len": ids.shape[-1], "problem_idx": pidx, "gen_seed": args.gen_seed,
                    "model": args.model, "task": args.task, "text": tok.decode(seq[0]),
                    "created": now()}, capture_file(args, pidx))
        log(f"capture: saved {capture_file(args, pidx)} windows={wins}")
        torch.cuda.empty_cache() if args.device != "cpu" else None


def capture_file(args, pidx):
    return os.path.join(args.capture_dir, f"{args.model}_{args.task}_p{pidx}.pt")


# =========================================================================
# stats: the measurement core
# =========================================================================
class Acc:
    """Per-(layer, head) accumulators, float64 numpy."""
    def __init__(self, nl, H, ngrid):
        z = lambda *s: np.zeros((nl, H) + s)
        self.u2 = {"calib": z(), "test": z()}                   # sum ||u_dense||^2
        self.d2 = {"calib": z(ngrid), "test": z(ngrid)}         # sum ||Delta_eta||^2 on the grid
        self.nq = {"calib": z(), "test": z()}
        self.a, self.b, self.c = z(), z(), z()                  # rule 1


def page_layout(L, args):
    P = (L - args.sink - args.recent) // args.page
    lo, hi = args.sink, args.sink + P * args.page
    return P, lo, hi


def group_out(Wh, o):
    """Wh [D, H, g, d], o [H, g, nq, d] -> per-KV-head residual contribution [H, nq, D]."""
    return torch.einsum('mhgd,hgqd->hqm', Wh, o)


def retrieval_loss_terms(q, K, V, L, qpos, Khat, Vhat, sel_mask, args):
    """Sparse attention of the method for one layer and a set of per-head etas.

    q [H, g, nq, d]; K, V [H, L, d] exact; Khat, Vhat [H, P*page, d] the
    reconstructed pages; sel_mask [H, nq, P] selected pages. Attends to
    sink (exact) + selected pages (reconstructed) + recent (exact, causal).
    Returns o_hat [H, g, nq, d].
    """
    H, g, nq, d = q.shape
    P, lo, hi = page_layout(L, args)
    s = 1.0 / math.sqrt(d)
    lg_sink = torch.einsum('hgqd,htd->hgqt', q, K[:, :lo]) * s
    lg_pg = torch.einsum('hgqd,htd->hgqt', q, Khat) * s
    tokmask = sel_mask.repeat_interleave(args.page, -1)                         # [H, nq, P*page]
    lg_pg = lg_pg.masked_fill(~tokmask[:, None], float('-inf'))
    lg_rec = torch.einsum('hgqd,htd->hgqt', q, K[:, hi:L]) * s
    rpos = torch.arange(hi, L, device=q.device)
    lg_rec = lg_rec.masked_fill(rpos[None, None, None, :] > qpos[None, None, :, None], float('-inf'))
    lg = torch.cat([lg_sink, lg_pg, lg_rec], -1)
    p = torch.softmax(lg, -1)
    n0, n1 = lo, lo + P * args.page
    o = (torch.einsum('hgqt,htd->hgqd', p[..., :n0], V[:, :lo])
         + torch.einsum('hgqt,htd->hgqd', p[..., n0:n1], Vhat)
         + torch.einsum('hgqt,htd->hgqd', p[..., n1:], V[:, hi:L]))
    return o


def select_pages(qw, mu, basis, coef, H, g, budget_pages):
    """OVAL scores with the CURRENT query (spec_ret simplification) and avgSM
    combine, exactly oval_sel's arithmetic; top-k pages per KV head -> [H, nq, P] mask."""
    nq = qw.shape[2]
    qb = qw.permute(2, 0, 1, 3).reshape(nq, 1, H * g, -1)                       # [nq, 1, nh, d]
    sc = PB.oval_sel(qb, basis.expand(nq, -1, -1, -1, -1), coef.expand(nq, -1, -1, -1, -1),
                     mu.expand(nq, -1, -1, -1), "avgSM", H * g, H)            # [nq, H, P]
    P = sc.shape[-1]
    k = min(budget_pages, P)
    idx = sc.topk(k, -1).indices
    mask = torch.zeros_like(sc, dtype=torch.bool).scatter_(-1, idx, True)
    return mask.permute(1, 0, 2).contiguous()                                  # [H, nq, P]


def window_stats(acc, split, li, q, K, V, L, qpos, Wh, Gh, grid, args, rule_tables=None):
    """All measurements for one (trace, window, layer). q [H, g, nq, d] float32,
    K, V [H, L, d] float32. rule_tables: {name: eta [H]} evaluated on top of the
    grid (regret stage)."""
    dev = q.device
    H, g, nq, d = q.shape
    P, lo, hi = page_layout(L, args)
    budget_pages = args.budget // args.page
    s = 1.0 / math.sqrt(d)
    # ---- dense reference
    lg = torch.einsum('hgqd,htd->hgqt', q, K) * s
    lg = lg.masked_fill(torch.arange(L, device=dev)[None, None, None, :] > qpos[None, None, :, None], float('-inf'))
    p = torch.softmax(lg, -1)                                                    # [H, g, nq, L]
    o = torch.einsum('hgqt,htd->hgqd', p, V)
    u = group_out(Wh, o)                                                         # [H, nq, D]
    u2 = (u * u).sum((-1, -2)).double().cpu().numpy()
    acc.u2[split][li] += u2; acc.nq[split][li] += nq
    # ---- pages in the harness layout [1, P, page, H, d]
    Kp = K[:, lo:hi].reshape(H, P, args.page, d).permute(1, 2, 0, 3)[None].contiguous()
    Vp = V[:, lo:hi].reshape(H, P, args.page, d).permute(1, 2, 0, 3)[None].contiguous()
    vmu, vD, vT = PB.build_value_summary(Kp, Vp, None, None, 0.0, args.vrank)
    Vhat = (vT.float() @ vD.float().transpose(-1, -2) + vmu.float().unsqueeze(-2))[0]   # [P, H, page, d]
    Vhat = Vhat.permute(1, 0, 2, 3).reshape(H, P * args.page, d)

    def loss_at(eta):
        mu, bs, cf = PB.build_summary(Kp, Vp, Gh, eta, args.rank)
        Khat = (cf.float() @ bs.float().transpose(-1, -2) + mu.float().unsqueeze(-2))[0]
        Khat = Khat.permute(1, 0, 2, 3).reshape(H, P * args.page, d)
        sel = select_pages(q, mu, bs, cf, H, g, budget_pages)
        oh = retrieval_loss_terms(q, K, V, L, qpos, Khat, Vhat, sel, args)
        delta = u - group_out(Wh, oh)                                            # [H, nq, D]
        return delta

    deltas = {}
    for gi, eta in enumerate(grid):
        delta = loss_at(float(eta))
        acc.d2[split][li, :, gi] += (delta * delta).sum((-1, -2)).double().cpu().numpy()
        if eta in (0.0, 1.0):
            deltas[eta] = delta
    if rule_tables is not None:
        out = {}
        for name, eta_h in rule_tables.items():
            delta = loss_at(torch.as_tensor(eta_h, dtype=torch.float64))
            out[name] = (delta * delta).sum((-1, -2)).double().cpu().numpy()
        return out
    if split != "calib":
        return None
    # ---- rule 1 inputs from the endpoint errors
    D0, D1 = deltas[0.0], deltas[1.0]
    acc.a[li] += (D0 * D0).sum((-1, -2)).double().cpu().numpy()
    acc.b[li] += (D1 * D1).sum((-1, -2)).double().cpu().numpy()
    acc.c[li] += (D0 * D1).sum((-1, -2)).double().cpu().numpy()
    return None


def load_capture(args, pidx, dev):
    return torch.load(capture_file(args, pidx), map_location="cpu")


def model_weights(args, dev):
    w = torch.load(os.path.join(args.capture_dir, f"{args.model}_oproj.pt"), map_location="cpu")
    nl, nh, nkv, hd, D = w["n_layers"], w["n_heads"], w["n_kv_heads"], w["head_dim"], w["hidden"]
    g = nh // nkv
    Wh = [wt.float().to(dev).view(D, nkv, g, hd) for wt in w["o_proj"]]           # [D, H, g, d]
    Gh = [PB.publish_metric(wt.float().to(dev), nkv, hd, g) for wt in w["o_proj"]]
    return w, nl, nh, nkv, hd, D, g, Wh, Gh


def stats_file(args):
    return os.path.join(args.out, f"{args.model}_{args.task}{args.tag}.json")


def stage_stats(args):
    dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    out_json = stats_file(args)
    if os.path.exists(out_json) and not args.overwrite:
        log(f"stats: {out_json} exists; skipping (pass --overwrite to recompute)")
        return
    w, nl, nh, nkv, hd, D, g, Wh, Gh = model_weights(args, dev)
    layers = layer_subset(args.layers, nl)
    if args.ngrid < 2:
        raise SystemExit("--ngrid must be >= 2 so that 0 and 1 are grid points (rule 1)")
    grid = [round(i / (args.ngrid - 1), 6) for i in range(args.ngrid)]
    acc = Acc(nl, nkv, len(grid))
    probs = problems(args)
    splits = {pidx: ("calib" if k < args.ncalib else "test") for k, (pidx, _) in enumerate(probs)}
    windows = {}
    t0 = time.time()
    for pidx, _ in probs:
        cap = load_capture(args, pidx, dev)
        split = splits[pidx]
        windows[str(pidx)] = list(cap["windows"])
        for wi, L in enumerate(cap["windows"]):
            qpos = torch.arange(L - cap["nq"], L, device=dev)
            for li in layers:
                K = cap["K"][li, :, :L].float().to(dev); V = cap["V"][li, :, :L].float().to(dev)
                q = cap["Q"][li, wi].float().to(dev).permute(1, 0, 2).reshape(nkv, g, cap["nq"], hd)
                window_stats(acc, split, li, q, K, V, L, qpos, Wh[li], Gh[li], grid, args)
            log(f"stats: {args.model}/{args.task} p{pidx} ({split}) window L={L} done, {time.time() - t0:.0f}s")
        del cap
    # ---- rule 1 per (layer, head) from the accumulators
    L_grid = {sp: acc.d2[sp] / np.maximum(acc.u2[sp][..., None], 1e-300) for sp in ("calib", "test")}
    eta1, deg1 = R.rule1_endpoint(acc.a, acc.b, acc.c)
    best_calib = np.array(grid)[L_grid["calib"].argmin(-1)]
    best_test = np.array(grid)[L_grid["test"].argmin(-1)]
    res = {
        "model": args.model, "task": args.task, "created": now(),
        "config": {k: v for k, v in vars(args).items() if k not in ("stage",)},
        "n_layers": nl, "n_kv_heads": nkv, "n_heads": nh, "head_dim": hd, "layers_done": layers,
        "grid": grid, "windows": windows,
        "splits": {str(p): s for p, s in splits.items()},
        "loss_grid": {sp: L_grid[sp].tolist() for sp in L_grid},          # [layer][head][grid] relative error
        "n_queries": {sp: acc.nq[sp].tolist() for sp in acc.nq},
        "best_grid_eta": {"calib": best_calib.tolist(), "test": best_test.tolist()},
        "rules": {
            "r1": {"eta": eta1.numpy().tolist(), "degenerate": deg1.numpy().tolist(),
                   "a": acc.a.tolist(), "b": acc.b.tolist(), "c": acc.c.tolist(), "default": 0.5},
        },
    }
    json.dump(res, open(out_json, "w"))
    log(f"stats: wrote {out_json} in {time.time() - t0:.0f}s")
    _print_eta_summary(res)


def _print_eta_summary(res):
    for name, rr in res["rules"].items():
        e = np.array(rr["eta"])[1:]            # layer 0 is dense in the harness
        dg = np.array(rr["degenerate"])[1:]
        print(f"  {name}: eta mean {e.mean():.3f} std {e.std():.3f} min {e.min():.3f} med {np.median(e):.3f} "
              f"max {e.max():.3f}  at0 {np.mean(e <= 0):.2f} at1 {np.mean(e >= 1):.2f}  degenerate {int(dg.sum())}/{dg.size}")


# =========================================================================
# rules: the deployable table file
# =========================================================================
def table_dir(args):
    return os.path.join(args.out, "tables")


def write_table(args, model, task, rule, eta, info):
    d = table_dir(args); os.makedirs(d, exist_ok=True)
    eta = [[round(float(v), 4) for v in row] for row in eta]
    path = os.path.join(d, f"{model}_{task}_{rule}{args.tag}.json")
    h = PB.eta_table_hash(eta)
    if os.path.exists(path):
        old = json.load(open(path))
        if old.get("hash") == h:
            return path, h
        if not args.overwrite:
            raise SystemExit(f"{path} exists with a different table (hash {old.get('hash')} vs {h}); --overwrite to replace")
    json.dump({"model": model, "task": task, "rule": rule, "deploy": True, "hash": h,
               "n_layers": len(eta), "n_kv_heads": len(eta[0]), "eta": eta, "created": now(),
               "source": f"{model}_{task}{args.tag}.json",
               **info}, open(path, "w"), indent=1)
    return path, h


def stage_rules(args):
    stats = {}
    for m in args.models:
        for t in args.tasks:
            f = os.path.join(args.out, f"{m}_{t}{args.tag}.json")
            if os.path.exists(f):
                stats[(m, t)] = json.load(open(f))
    if not stats:
        raise SystemExit("rules: no stats files found")
    for (m, t), st in stats.items():
        for rule in RULES:
            rr = st["rules"][rule]
            p, h = write_table(args, m, t, rule, rr["eta"], {"degenerate_count": int(np.sum(rr["degenerate"]))})
            log(f"rules: {m}/{t} {rule} -> {os.path.basename(p)} hash {h}")
        _print_eta_summary(st)


# =========================================================================
# regret on the test split with the harness's per-head build
# =========================================================================
def stage_regret(args):
    dev = torch.device(args.device)
    f = stats_file(args); st = json.load(open(f))
    w, nl, nh, nkv, hd, D, g, Wh, Gh = model_weights(args, dev)
    layers = st["layers_done"]
    grid = st["grid"]
    tables = {}
    for rule in RULES:
        p = os.path.join(table_dir(args), f"{args.model}_{args.task}_{rule}{args.tag}.json")
        if os.path.exists(p):
            tables[rule] = np.array(json.load(open(p))["eta"])
    tables["calib_grid_opt"] = np.array(st["best_grid_eta"]["calib"])
    probs = problems(args)
    test = [p for k, (p, _) in enumerate(probs) if k >= args.ncalib]
    acc = Acc(nl, nkv, len(grid))
    d2 = {k: np.zeros((nl, nkv)) for k in tables}
    for pidx in test:
        cap = load_capture(args, pidx, dev)
        for wi, L in enumerate(cap["windows"]):
            qpos = torch.arange(L - cap["nq"], L, device=dev)
            for li in layers:
                K = cap["K"][li, :, :L].float().to(dev); V = cap["V"][li, :, :L].float().to(dev)
                q = cap["Q"][li, wi].float().to(dev).permute(1, 0, 2).reshape(nkv, g, cap["nq"], hd)
                out = window_stats(acc, "test", li, q, K, V, L, qpos, Wh[li], Gh[li], grid, args,
                                   rule_tables={k: v[li] for k, v in tables.items()})
                for k in out:
                    d2[k][li] += out[k]
            log(f"regret: {args.model}/{args.task} p{pidx} window L={L} done")
    u2 = np.maximum(acc.u2["test"], 1e-300)
    Lg = acc.d2["test"] / u2[..., None]
    Lmin = Lg.min(-1)
    res = {"grid_min_test": Lmin.tolist(), "best_test_eta": np.array(grid)[Lg.argmin(-1)].tolist(), "tables": {}}
    for k in tables:
        Lk = d2[k] / u2
        res["tables"][k] = {"loss": Lk.tolist(), "regret": (Lk - Lmin).tolist(),
                            "mean_regret_layers1plus": float((Lk - Lmin)[1:].mean()),
                            "mean_loss_layers1plus": float(Lk[1:].mean())}
        log(f"regret: {k:<14} mean loss {Lk[1:].mean():.4f}  mean regret {(Lk - Lmin)[1:].mean():.5f}")
    log(f"regret: grid optimum mean loss {Lmin[1:].mean():.4f}")
    st["regret"] = res; st["regret_created"] = now()
    json.dump(st, open(f, "w"))
    log(f"regret: updated {f}")


if __name__ == "__main__":
    args = parse()
    if args.gen_seed in range(42, 50):
        raise SystemExit("gen_seed must be outside the evaluated seeds 42..49")
    if args.stage != "rules" and (args.model is None or args.task is None):
        raise SystemExit("--model and --task are required")
    torch.backends.cuda.matmul.allow_tf32 = False
    {"capture": stage_capture, "stats": stage_stats, "rules": stage_rules, "regret": stage_regret}[args.stage](args)
