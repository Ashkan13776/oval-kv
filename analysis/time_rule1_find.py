"""Time the cost of FINDING rule 1's eta* table for one (model, task).

Rule 1 needs, per (layer, KV head), the sums a = sum ||D0||^2, b = sum ||D1||^2,
c = sum D0.D1 of the output errors at the two endpoints eta = 0 and eta = 1,
over the calibration windows (analysis/eta_rules_calibrate.py, stats stage).
The stats stage also evaluates the 21-point grid and rules 2-6, so its wall
time overstates rule 1; this script runs ONLY the rule-1 part on the same
captured calibration traces (logs/eta_rules/capture, the first ncalib traces),
checks that the resulting table equals the deployed one, and records

  gen_s      generating the calibration traces (dense HF decode, from the
             capture log; the trace generation is part of finding eta*)
  load_s     reading the captures from disk
  compute_s  endpoint errors + a, b, c + the closed form, on the GPU

to results/eta_rules/latency/<model>_<task>_r1_find.json.

  python analysis/time_rule1_find.py --model ds-r1-qwen-7b --task AIME24
"""
import json, math, os, re, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT

sys.argv = [sys.argv[0], "stats"] + sys.argv[1:]
import eta_rules_calibrate as C                       # noqa: E402

args = C.parse()
torch.backends.cuda.matmul.allow_tf32 = False
dev = torch.device(args.device)


def endpoint_terms(q, K, V, L, qpos, Wh, Gh):
    """a, b, c contributions of one (window, layer): window_stats's dense
    reference and loss_at(0), loss_at(1), nothing else."""
    H, g, nq, d = q.shape
    P, lo, hi = C.page_layout(L, args)
    s = 1.0 / math.sqrt(d)
    lg = torch.einsum('hgqd,htd->hgqt', q, K) * s
    lg = lg.masked_fill(torch.arange(L, device=dev)[None, None, None, :] > qpos[None, None, :, None], float('-inf'))
    o = torch.einsum('hgqt,htd->hgqd', torch.softmax(lg, -1), V)
    u = C.group_out(Wh, o)
    Kp = K[:, lo:hi].reshape(H, P, args.page, d).permute(1, 2, 0, 3)[None].contiguous()
    Vp = V[:, lo:hi].reshape(H, P, args.page, d).permute(1, 2, 0, 3)[None].contiguous()
    vmu, vD, vT = C.PB.build_value_summary(Kp, Vp, None, None, 0.0, args.vrank)
    Vhat = (vT.float() @ vD.float().transpose(-1, -2) + vmu.float().unsqueeze(-2))[0]
    Vhat = Vhat.permute(1, 0, 2, 3).reshape(H, P * args.page, d)
    D = []
    for eta in (0.0, 1.0):
        mu, bs, cf = C.PB.build_summary(Kp, Vp, Gh, eta, args.rank)
        Khat = (cf.float() @ bs.float().transpose(-1, -2) + mu.float().unsqueeze(-2))[0]
        Khat = Khat.permute(1, 0, 2, 3).reshape(H, P * args.page, d)
        sel = C.select_pages(q, mu, bs, cf, H, g, args.budget // args.page)
        D.append(u - C.group_out(Wh, C.retrieval_loss_terms(q, K, V, L, qpos, Khat, Vhat, sel, args)))
    D0, D1 = D
    return (D0 * D0).sum((-1, -2)).double(), (D1 * D1).sum((-1, -2)).double(), (D0 * D1).sum((-1, -2)).double()


w, nl, nh, nkv, hd, Dm, g, Wh, Gh, Gi = C.model_weights(args, dev)
calib = C.problems(args)[:args.ncalib]
a = torch.zeros(nl, nkv, dtype=torch.float64, device=dev); b = torch.zeros_like(a); c = torch.zeros_like(a)
load_s = compute_s = 0.0
for pidx, _ in calib:
    t = time.time()
    cap = C.load_capture(args, pidx, dev)
    load_s += time.time() - t
    for wi, L in enumerate(cap["windows"]):
        qpos = torch.arange(L - cap["nq"], L, device=dev)
        for li in range(nl):
            K = cap["K"][li, :, :L].float().to(dev); V = cap["V"][li, :, :L].float().to(dev)
            q = cap["Q"][li, wi].float().to(dev).permute(1, 0, 2).reshape(nkv, g, cap["nq"], hd)
            torch.cuda.synchronize(); t = time.time()
            da, db, dc = endpoint_terms(q, K, V, L, qpos, Wh[li], Gh[li])
            a[li] += da; b[li] += db; c[li] += dc
            torch.cuda.synchronize(); compute_s += time.time() - t
    del cap
t = time.time()
eta, _ = C.R.rule1_endpoint(a.cpu().numpy(), b.cpu().numpy(), c.cpu().numpy())
compute_s += time.time() - t

dep = json.load(open(ROOT / f"results/eta_rules/tables/{args.model}_{args.task}_r1.json"))
diff = float(np.abs(np.asarray(eta.numpy())[1:] - np.asarray(dep["eta"])[1:]).max())   # layer 0 is dense

gen_s = None
logf = ROOT / f"logs/eta_rules/calib_{args.model}.log"
if logf.exists():
    want = {p for p, _ in calib}
    pat = re.compile(rf"capture: {re.escape(args.model)}/{args.task} problem (\d+):.* in (\d+)s")
    gen_s = sum(int(m.group(2)) for m in map(pat.search, open(logf)) if m and int(m.group(1)) in want)

res = {"model": args.model, "task": args.task, "ncalib": args.ncalib, "ngen": args.ngen,
       "nwin": args.nwin, "gen_s": gen_s, "load_s": load_s, "compute_s": compute_s,
       "max_abs_diff_vs_deployed": diff, "gpu": torch.cuda.get_device_name(dev)}
out = ROOT / f"results/eta_rules/latency/{args.model}_{args.task}_r1_find.json"
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res))
