"""Verify the OVAL page representation before spending GPU on the eta sweep.

Checks, in order of what would silently corrupt results:
  1. eta=0 basis == plain rank-r key PCA (no output term leaks in)
  2. basis is orthonormal
  3. coefficients satisfy R = X B exactly
  4. the eta-mixed basis spans the same subspace as an INDEPENDENT d x d
     construction of M_eta (catches an error in the page_size x page_size
     Gram route, which is the non-obvious part)
  5. G = W^T W is summed over the right GQA group
  6. logsumexp scorer reproduces the exact page mass when rank == page_size
"""
import sys, torch
# load oval/page_basis.py directly: the kvc.patch package __init__ imports llama.py,
# which needs flash_attn -- irrelevant to this pure-math check
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "page_basis",
    str(__import__("pathlib").Path(__file__).resolve().parent.parent
        / "oval/page_basis.py"))
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
build_summary, publish_metric, oval_sel = (
    _m.build_summary, _m.publish_metric, _m.oval_sel)

torch.manual_seed(0)
B, P, page, nkv, hd, r = 1, 6, 32, 4, 128, 8
g = 4
K = torch.randn(B, P, page, nkv, hd).double() * 1.6
V = torch.randn(B, P, page, nkv, hd).double() * 0.4
W = (torch.randn(nkv * g * hd, nkv * g * hd) / 64).double()
Gh = publish_metric(W, nkv, hd, g)

X = (K.permute(0,1,3,2,4) - K.permute(0,1,3,2,4).mean(3, keepdim=True))

# 1 + 2 + 3 -------------------------------------------------------------
mu, bs, cf = build_summary(K, V, None, 0.0, r)
cov = X.transpose(-1,-2) @ X
ev, evec = torch.linalg.eigh(cov)
ref = evec[..., -r:]
# compare SUBSPACES (sign/order of eigenvectors is arbitrary)
proj_a = bs.double() @ bs.double().transpose(-1,-2)
proj_b = ref @ ref.transpose(-1,-2)
print(f"1. eta=0 vs plain key PCA subspace   : {(proj_a-proj_b).abs().max():.2e}")
I = torch.eye(r).double()
print(f"2. |B^T B - I|                        : {((bs.double().transpose(-1,-2)@bs.double())-I).abs().max():.2e}")
print(f"3. |R - X B|                          : {(cf.double() - X@bs.double()).abs().max():.2e}")

# 4: independent d x d construction of M_eta ----------------------------
eta = 0.5
mu2, bs2, cf2 = build_summary(K, V, Gh, eta, r)
Y = (V.permute(0,1,3,2,4) - V.permute(0,1,3,2,4).mean(3, keepdim=True))
G = Gh.double() @ Gh.double()
Mkey = X.transpose(-1,-2) @ X
YG = Y @ G
Mout = X.transpose(-1,-2) @ (YG @ Y.transpose(-1,-2)) @ X
trk = torch.diagonal(Mkey,dim1=-2,dim2=-1).sum(-1)[...,None,None]
tro = torch.diagonal(Mout,dim1=-2,dim2=-1).sum(-1)[...,None,None]
M = (1-eta)*Mkey/trk + eta*Mout/tro
_, ev2 = torch.linalg.eigh(M)
ref2 = ev2[..., -r:]
pa = bs2.double() @ bs2.double().transpose(-1,-2)
pb = ref2 @ ref2.transpose(-1,-2)
print(f"4. eta=0.5 Gram route vs d x d M_eta  : {(pa-pb).abs().max():.2e}")

# 5: G group summation ---------------------------------------------------
Wh = W.reshape(W.shape[0], nkv*g, hd)
G0 = sum(Wh[:,j,:].T @ Wh[:,j,:] for j in range(g))     # KV head 0's group
print(f"5. G_0 vs manual group sum            : {(G[0]-G0).abs().max():.2e}")

# 6: full-rank scorer == exact page logsumexp ---------------------------
# centring costs one dof, so rank(X) = page-1; that is the rank at which the
# rank-r reconstruction becomes exact (asking for `page` divides by a ~0 eig,
# which build_summary now clamps against).
mu3, bs3, cf3 = build_summary(K, V, None, 0.0, page - 1)
q = torch.randn(B, 1, nkv*g, hd).double()
approx = oval_sel(q.float(), bs3.float(), cf3.float(), mu3.float(), "avgS", nkv*g, nkv)
Kp = K.permute(0,1,3,2,4)                                # [B,P,nkv,page,hd]
qh = q.reshape(B, nkv, g, hd)
exact = torch.logsumexp(torch.einsum('bpktd,bkgd->bpkgt', Kp, qh), dim=-1)
exact = exact.permute(0,2,3,1).reshape(B, nkv*g, P)
# oval_sel returns the GQA-reduced layout [B, n_kv, P]; avgS is a plain mean
# over the group, so apply the same mean to the exact per-head scores
exact = exact.reshape(B, nkv, g, P).mean(dim=2)
print(f"6. rank=page-1 scorer vs exact lse    : {(approx.double()-exact).abs().max():.2e}")
# 7: the clamp actually fires
_, bs4, _ = build_summary(K, V, None, 0.0, page)      # over-rank request
print(f"7. rank clamped {page} -> {bs4.shape[-1]:<21}: {'OK' if bs4.shape[-1]==page-1 else 'NOT CLAMPED'}")
