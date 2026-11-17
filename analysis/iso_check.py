"""Does the isotropic-query assumption cost anything in practice?

For every KV head (layers >= 1) we build each page's rank-r key basis twice:
  iso : top-r eigenvectors of M_eta computed with E[qq^T] = I   (deployed OVAL)
  wht : the query-whitened basis (Supp. corollary), using Sigma_q estimated on calibration traces 1-4
and measure, on real held-out queries (traces 5-6), the error of the attention output after W_O when
the page keys are replaced by their reconstruction (values exact, page-local softmax). The whitened
basis is the optimum of the hybrid risk under the true query second moment, so err_iso / err_wht >= 1
(up to estimation noise) and its size is the cost of assuming isotropy.

usage: python iso_check.py --capdir $KV_ROOT/logs/eta_rules/capture --model ds-r1-qwen-7b --task MATH50
"""
import argparse, glob, json, math, os, re
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--capdir", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--task", default="MATH50"); ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--page", type=int, default=32); ap.add_argument("--npages", type=int, default=24)
ap.add_argument("--etas", default="0,0.5,1"); ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()
dev, f64 = a.device, torch.float64
etas = [float(e) for e in a.etas.split(",")]

files = glob.glob(os.path.join(a.capdir, f"{a.model}_{a.task}_p*.pt"))
files = sorted(files, key=lambda f: int(re.search(r"_p(\d+)\.pt$", f).group(1)))
assert len(files) == 6, files
calib, held = files[:4], files[4:]
op = torch.load(os.path.join(a.capdir, f"{a.model}_oproj.pt"))
nl, nh, nkv, hd = op["n_layers"], op["n_heads"], op["n_kv_heads"], op["head_dim"]
g = nh // nkv; m, r = a.page, a.rank

def sqrtm_psd(S, inv=False):
    w, U = torch.linalg.eigh(S); w = w.clamp_min(1e-12)
    return (U * (w.rsqrt() if inv else w.sqrt())) @ U.T

# Sigma_q per (layer, KV head) from calibration traces (all windows, all query heads of the group)
Sig = torch.zeros(nl, nkv, hd, hd, dtype=f64); cnt = torch.zeros(nl, nkv, dtype=f64)
for f in calib:
    c = torch.load(f, map_location="cpu")
    Q = c["Q"].to(f64)                                            # [nl, W, nq, nh, hd]
    for h in range(nkv):
        q = Q[:, :, :, h * g:(h + 1) * g, :].reshape(nl, -1, hd)  # [nl, N, hd]
        Sig[:, h] += torch.einsum("lni,lnj->lij", q, q); cnt[:, h] += q.shape[1]
    del c, Q
Sig /= cnt[..., None, None]

def topr(M):
    w, U = torch.linalg.eigh(M)                                   # ascending
    return U[..., -r:]

res = {e: {"iso": torch.zeros(nl, nkv, dtype=f64), "wht": torch.zeros(nl, nkv, dtype=f64)} for e in etas}
ref = torch.zeros(nl, nkv, dtype=f64)
gen = torch.Generator().manual_seed(0)
for f in held:
    c = torch.load(f, map_location="cpu")
    K, V, Q, wins, nq = c["K"], c["V"], c["Q"], c["windows"], c["nq"]
    last = (min(wins) - nq) // m                                  # pages every held-out query can see
    pages = torch.arange(1, last)                                 # skip the sink page
    pages = pages[torch.randperm(len(pages), generator=gen)[:a.npages]].sort().values
    for l in range(1, nl):
        W = op["o_proj"][l].to(dev, f64)                          # [hidden, nh*hd]
        for h in range(nkv):
            Wo = [W[:, i * hd:(i + 1) * hd] for i in range(h * g, (h + 1) * g)]
            G = sum(w.T @ w for w in Wo)
            S = Sig[l, h].to(dev); S = S + 1e-4 * torch.trace(S) / hd * torch.eye(hd, device=dev, dtype=f64)
            Sh, Shi = sqrtm_psd(S), sqrtm_psd(S, inv=True)
            Kp = torch.stack([K[l, h, p * m:(p + 1) * m] for p in pages]).to(dev, f64)   # [P, m, d]
            Vp = torch.stack([V[l, h, p * m:(p + 1) * m] for p in pages]).to(dev, f64)
            mu = Kp.mean(1, keepdim=True); X = Kp - mu; Y = Vp - Vp.mean(1, keepdim=True)
            q = Q[l, :, :, h * g:(h + 1) * g, :].reshape(-1, g, hd).to(dev, f64)       # [N, g, d]
            def out_err(Xhat):
                Kh = mu + Xhat
                tot = 0.0
                for j in range(g):
                    qq = q[:, j, :]                                                     # [N, d]
                    a0 = torch.softmax(torch.einsum("pmd,nd->pnm", Kp, qq) / math.sqrt(hd), -1)
                    a1 = torch.softmax(torch.einsum("pmd,nd->pnm", Kh, qq) / math.sqrt(hd), -1)
                    do = torch.einsum("pnm,pmd->pnd", a0 - a1, Vp)                       # [P, N, d]
                    tot = tot + torch.einsum("pnd,hd->pnh", do, Wo[j])
                return (tot ** 2).sum().item()
            # reference scale: output deviation of the page from its mean value (for a relative error)
            tot = 0.0
            for j in range(g):
                a0 = torch.softmax(torch.einsum("pmd,nd->pnm", Kp, q[:, j]) / math.sqrt(hd), -1)
                tot = tot + torch.einsum("pnd,hd->pnh", torch.einsum("pnm,pmd->pnd", a0, Y), Wo[j])
            ref[l, h] += (tot ** 2).sum().item()
            Xt = X @ Sh                                                                 # whitened keys
            for e in etas:
                for name, XX in (("iso", X), ("wht", Xt)):
                    Mk = XX.transpose(1, 2) @ XX
                    Mo = XX.transpose(1, 2) @ Y @ G @ Y.transpose(1, 2) @ XX
                    tk = Mk.diagonal(dim1=1, dim2=2).sum(-1)[:, None, None] + 1e-12
                    to = Mo.diagonal(dim1=1, dim2=2).sum(-1)[:, None, None] + 1e-12
                    B = topr((1 - e) * Mk / tk + e * Mo / to)                           # [P, d, r]
                    Xh = XX @ B @ B.transpose(1, 2)
                    if name == "wht": Xh = Xh @ Shi                                     # back to query coords
                    res[e][name][l, h] += out_err(Xh)
    del c, K, V, Q

# anisotropy of the query second moment (layers >= 1): effective rank / d and top-8 energy share
ev = torch.linalg.eigvalsh(Sig[1:])                                                     # [nl-1, nkv, d]
effr = (ev.sum(-1) ** 2 / (ev ** 2).sum(-1)) / hd
top8 = ev[..., -8:].sum(-1) / ev.sum(-1)
out = {"model": a.model, "task": a.task, "rank": r, "pages_per_trace": a.npages, "heads": (nl - 1) * nkv,
       "query_eff_rank_over_d_median": effr.median().item(), "query_top8_energy_median": top8.median().item()}
for e in etas:
    iso, wht = res[e]["iso"][1:], res[e]["wht"][1:]
    ratio = (iso / wht.clamp_min(1e-30)).flatten()
    out[f"eta={e}"] = {"pooled_excess_pct": 100 * (iso.sum() / wht.sum() - 1).item(),
                       "per_head_excess_median_pct": 100 * (ratio.median() - 1).item(),
                       "per_head_excess_p90_pct": 100 * (ratio.quantile(0.9) - 1).item(),
                       "iso_rel_err_pct": 100 * (iso.sum() / ref[1:].sum()).item(),
                       "wht_rel_err_pct": 100 * (wht.sum() / ref[1:].sum()).item()}
json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1))
