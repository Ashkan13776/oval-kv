"""Rank-r output-aware page representation, as a drop-in for FreeKV's page_rep.

FreeKV summarises a page with two vectors -- elementwise min/max of the keys
("quest"), or a mean-absolute-deviation band around their midpoint ("arkv") --
and scores a page with the Quest bound  sum_d max(q_d*min_d, q_d*max_d).

This module replaces that summary with a rank-r basis of the page's keys, and
scores the page by reconstructing per-token logits inside the page and taking
their logsumexp (the exact page mass, up to the rank-r truncation).

The basis is the leading eigenvectors of the trace-normalised mix

    M_eta = (1-eta) * X^T X / tr(X^T X)  +  eta * X^T Y G Y^T X / tr(X^T Y G Y^T X)

with X = HK the centred keys of the page, Y = HV its centred values, and
G = W^T W the value-space metric from the attention output projection, summed
over each KV head's GQA group.  eta=0 is the plain key PCA; eta=1 is the
output-only basis; in between the two terms are mixed at matched trace so the
weight is scale-free (a raw lambda is uninterpretable -- on real pages the two
traces differ by ~8 orders of magnitude).

Everything downstream of the per-head page score -- the GQA_policy combine, the
top-k, the speculative q-cache and the correction -- is FreeKV's, untouched.
"""
import os
import torch


def _eigh_chunked(M):
    """torch.linalg.eigh over a large batch, in chunks.

    cuSOLVER's batched syev fails with CUSOLVER_STATUS_INVALID_VALUE once the
    batch is large enough -- at 128K context that is ~4000 pages x 8 KV heads
    = 32k matrices in one call. The error text blames NaN, which is a red
    herring. Same fix as LOCKS_EIGH_CHUNK on the LOCKS side.
    """
    n = M.shape[-1]
    flat = M.reshape(-1, n, n)
    chunk = int(os.environ.get("OVAL_EIGH_CHUNK", "4096"))
    if flat.shape[0] <= chunk:
        w, V = torch.linalg.eigh(flat)
    else:
        ws, Vs = [], []
        for i in range(0, flat.shape[0], chunk):
            a, b = torch.linalg.eigh(flat[i:i + chunk])
            ws.append(a); Vs.append(b)
        w, V = torch.cat(ws, 0), torch.cat(Vs, 0)
    return w.reshape(*M.shape[:-1]), V.reshape(M.shape)


def outproj_metric_sqrt(o_proj_weight, num_kv_heads, head_dim, num_kv_groups):
    """G^{1/2} per KV head from the attention output projection.

    o_proj maps the concatenated head outputs to the model dimension, so its
    weight is [d_model, num_q_heads * head_dim].  The value-space metric seen by
    KV head k is the sum over the query heads in its group:

        G_k = sum_{g in group(k)}  W_g^T W_g          [head_dim, head_dim]

    We return G^{1/2} so the page build can fold it into Y once and then work
    with plain Gram matrices.  Symmetric PSD, so the root is taken by eigh.

    NOTE: this must be called AFTER llama.py's reorder_linear_weights, so the
    head order matches the dynamic-attention head order the pages are keyed by.
    """
    W = o_proj_weight.float()                       # [d_model, n_q*head_dim]
    n_q = W.shape[1] // head_dim
    Wh = W.reshape(W.shape[0], n_q, head_dim)       # [d_model, n_q, head_dim]
    G = torch.einsum("dgi,dgj->gij", Wh, Wh)        # W_g^T W_g per query head
    # sum over each KV head's GQA group
    G = G.reshape(num_kv_heads, num_kv_groups, head_dim, head_dim).sum(dim=1)
    ev, Q = torch.linalg.eigh(G.double())
    Gh = (Q * ev.clamp_min(0).sqrt().unsqueeze(-2)) @ Q.transpose(-1, -2)
    return Gh.to(o_proj_weight.dtype)               # [num_kv_heads, hd, hd]


def build_page_summary(paged_k, paged_v, Gh, eta, rank):
    """Rank-r summary of one batch of pages.

    paged_k, paged_v : [B, P, page_size, n_kv, head_dim]
    Gh               : [n_kv, head_dim, head_dim]  (G^{1/2}), or None when eta=0
    returns mu [B,P,n_kv,hd], basis [B,P,n_kv,hd,r], coef [B,P,n_kv,page,r]

    The eigenproblem is solved in the page_size x page_size Gram, not the
    head_dim x head_dim covariance: page_size (32) is far below head_dim (128),
    so the Gram route is both cheaper and exactly equivalent on the leading
    min(page_size, head_dim) directions.
    """
    dt = paged_k.dtype
    # [B,P,n_kv,page,hd] -- put the token axis where the Gram wants it
    X = paged_k.permute(0, 1, 3, 2, 4).float()
    mu = X.mean(dim=3)                                     # page centroid
    X = X - mu.unsqueeze(3)                                # centred keys

    if eta > 0.0:
        Y = paged_v.permute(0, 1, 3, 2, 4).float()
        Y = Y - Y.mean(dim=3, keepdim=True)                # centred values
        Z = Y @ Gh.to(Y.dtype)                             # Y G^{1/2}
        # Scale-normalise Z per page before the Gram. Long repetitive contexts
        # (passkey at 128K) can drive Z@Z^T past fp32 range; Mo then holds inf
        # and (eta/tr_out)*Mo evaluates to 0*inf = NaN, which reaches eigh as
        # CUSOLVER_STATUS_INVALID_VALUE. The scale cancels in the trace-
        # normalised mix below.
        Z = Z / Z.abs().amax(dim=-1, keepdim=True).amax(dim=-2, keepdim=True).clamp_min(1e-20)
        Mo = Z @ Z.transpose(-1, -2)                       # Y G Y^T  [.,page,page]
        Xg = X @ X.transpose(-1, -2)                       # X X^T
        # traces of the two d x d metrics, computed in the Gram domain:
        #   tr(X^T X)         = sum(X*X)
        #   tr(X^T Y G Y^T X) = sum( (Y G Y^T) * (X X^T) )
        tr_key = (X * X).sum(dim=(-1, -2)).clamp_min(1e-30)
        tr_out = (Mo * Xg).sum(dim=(-1, -2)).clamp_min(1e-30)
        eye = torch.eye(X.shape[-2], device=X.device, dtype=X.dtype)
        A = ((1.0 - eta) / tr_key)[..., None, None] * eye \
            + (eta / tr_out)[..., None, None] * Mo
        # normalise A to unit mean diagonal.  Without this the absolute scale of
        # A rides straight into the eigenvalues and can push near-degenerate
        # pages under the sqrt/divide guard below, which silently corrupts the
        # basis of exactly the flattest pages.
        A = A * (X.shape[-2] / A.diagonal(dim1=-2, dim2=-1).sum(-1)
                 .clamp_min(1e-30))[..., None, None]
        # A page whose metric is still non-finite falls back to the identity
        # (plain key PCA for that page) rather than poisoning the batched eigh.
        _bad = ~torch.isfinite(A).all(dim=-1).all(dim=-1)
        if bool(_bad.any()):
            A = torch.where(_bad[..., None, None], eye.to(A.dtype).expand_as(A), A)
        a, Q = _eigh_chunked(A)
        Ah = (Q * a.clamp_min(0).sqrt().unsqueeze(-2)) @ Q.transpose(-1, -2)
        Xw = Ah @ X                                        # A^{1/2} X
    else:
        Xw = X                                             # plain key PCA

    Gm = Xw @ Xw.transpose(-1, -2)                         # [.,page,page]
    S2, U = _eigh_chunked(Gm)
    # Centring costs one degree of freedom, so rank(X) <= page_size - 1: asking
    # for page_size directions divides by a ~zero eigenvalue and returns a
    # garbage basis rather than a more accurate one.  Clamp instead of trusting
    # the caller (verified in harness/test_locks_rep.py).
    r = min(rank, U.shape[-1] - 1, X.shape[-1])
    S = S2[..., -r:].clamp_min(1e-10).sqrt()
    U8 = U[..., -r:]
    basis = Xw.transpose(-1, -2) @ U8 / S.unsqueeze(-2)    # [.,hd,r], orthonormal
    coef = X @ basis                                       # R = X B  [.,page,r]
    return mu.to(dt), basis.to(dt), coef.to(dt)


def locks_sel(q, mu, basis, coef, num_heads, num_kv_heads):
    """Per-(query head, page) score from the rank-r summary.

    q     : [B, 1, num_heads, head_dim]
    mu    : [B, P, n_kv, head_dim]
    basis : [B, P, n_kv, head_dim, r]
    coef  : [B, P, n_kv, page_size, r]
    returns [B, num_heads, P] -- the same layout quest_sel produces, so the
    caller can hand it straight to FreeKV's GQA_policy combine.

    Score is logsumexp over the page's tokens of the reconstructed logit
        s_t = coef_t . (basis^T q) + mu . q
    i.e. the page's log attention mass under the rank-r reconstruction, which
    is the quantity the exact selector would rank by.
    """
    g = num_heads // num_kv_heads
    B, P, n_kv, hd, r = basis.shape
    # [B, n_kv, g, hd]
    qh = q.squeeze(1).reshape(B, n_kv, g, hd)
    # project the query into each page's basis: [B,P,n_kv,g,r]
    qb = torch.einsum("bpkdr,bkgd->bpkgr", basis.float(), qh.float())
    # per-token logits inside the page: [B,P,n_kv,g,page]
    tok = torch.einsum("bpktr,bpkgr->bpkgt", coef.float(), qb)
    # centroid term is constant across tokens of a page
    tok = tok + torch.einsum("bpkd,bkgd->bpkg", mu.float(), qh.float()).unsqueeze(-1)
    s = torch.logsumexp(tok, dim=-1)                       # [B,P,n_kv,g]
    return s.permute(0, 2, 3, 1).reshape(B, n_kv * g, P)   # [B,num_heads,P]


def gqa_combine(scores, GQA_policy, num_heads, num_kv_heads):
    """FreeKV's group-consistent combine, applied to our per-head page scores.

    Byte-for-byte the score-level branch of quest_sel (dynamic_attention.py:33),
    kept here so the quest/arkv path is not touched.  Only the score-level
    policies are meaningful for us: maxQ/avgQ pool the QUERY before scoring,
    which a rank-r reconstruction cannot express, and the paper's reasoning
    configuration uses avgSM anyway.
    """
    B = scores.shape[0]
    g = num_heads // num_kv_heads
    if GQA_policy == "maxS":
        return scores.reshape(B, num_kv_heads, g, -1).max(dim=-2).values
    if GQA_policy == "avgS":
        return scores.reshape(B, num_kv_heads, g, -1).mean(dim=-2)
    if GQA_policy == "maxSM":
        return torch.softmax(scores, dim=-1).reshape(
            B, num_kv_heads, g, -1).max(dim=-2).values
    if GQA_policy == "avgSM":
        return torch.softmax(scores, dim=-1).reshape(
            B, num_kv_heads, g, -1).mean(dim=-2)
    raise AssertionError(
        f"page_rep=locks does not support GQA_policy={GQA_policy}; "
        f"use one of maxS/avgS/maxSM/avgSM (the paper uses avgSM)")
