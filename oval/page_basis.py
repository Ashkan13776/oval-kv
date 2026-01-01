"""OVAL (output-aware page basis) for FreeKV's ACCURACY harness.

Mirrors source/freekv/oval_summary.py -- the perf-engine implementation -- so
the two paths compute the same summary and the same score. Kept in its own
module so their quest/arkv digest path is untouched.

eta = 0 is the plain key-PCA basis (the LOCKS special case); eta > 0 mixes in
the output-coupled metric  G = W^T W  taken from the attention output
projection. Records are UNQUANTIZED (model dtype) here.

Their digest is (max, min) per page and lives in module.min_k / module.max_k.
Ours is basis + coefficients + centroid, so it needs its own slabs, indexed by
the same page index their digest uses.
"""
import torch


def publish_metric(o_proj_weight, n_kv_heads, head_dim, n_kv_groups):
    """G^{1/2} per KV head from the attention output projection.

    o_proj maps concat(head outputs) -> model dim, so its weight is
    [d_model, n_qo_heads*head_dim]. The metric for a KV head is the sum of
    W_h^T W_h over the query heads sharing it.
    """
    W = o_proj_weight.detach().float()            # [d_model, n_qo*head_dim]
    # reorder_linear_weights concatenates W[:, full], W[:, dyn], W[:, neither];
    # if the full and dyn masks OVERLAP, columns are duplicated and o_proj comes
    # back WIDER than n_qo*head_dim (observed: 5151 vs 5120 on Qwen-14B under
    # device_map sharding). Silently flooring here would reshape garbage, so
    # refuse instead.
    if W.shape[1] % head_dim != 0:
        raise ValueError(
            f"o_proj has {W.shape[1]} input columns, not a multiple of "
            f"head_dim={head_dim}: reorder_linear_weights has duplicated "
            f"columns (overlapping full/dyn head masks). Refusing to guess.")
    n_qo = W.shape[1] // head_dim
    # Derive the group size from o_proj ITSELF. Deriving it from
    # q_proj/k_proj shapes is wrong here: reorder_linear_weights has already
    # rewritten those by the time we run, which silently gave 4 instead of 5
    # for Qwen-14B (40 qo / 8 kv) and blew up in the reshape below.
    if n_qo % n_kv_heads != 0:
        raise ValueError(
            f"o_proj implies n_qo={n_qo} which is not divisible by "
            f"n_kv_heads={n_kv_heads}; cannot form GQA groups")
    n_kv_groups = n_qo // n_kv_heads
    Wh = W.reshape(W.shape[0], n_qo, head_dim)
    G = torch.einsum('mhd,mhe->hde', Wh, Wh)      # [n_qo, d, d]
    G = G.reshape(n_kv_heads, n_kv_groups, head_dim, head_dim).sum(1)
    a, Q = torch.linalg.eigh(G)
    return (Q * a.clamp_min(0).sqrt().unsqueeze(-2)) @ Q.transpose(-1, -2)


def build_summary(paged_k, paged_v, Gh, eta, rank):
    """paged_k/paged_v: [B, P, page, n_kv, d] -> (mu, basis, coef).

    Returns mu [B,P,H,d], basis [B,P,H,d,r], coef [B,P,H,page,r].
    Line-for-line the perf path's build_summary.
    """
    dt = paged_k.dtype
    X = paged_k.permute(0, 1, 3, 2, 4).float()        # [B,P,H,page,d]
    mu = X.mean(dim=3)
    X = X - mu.unsqueeze(3)

    if eta > 0.0 and Gh is not None:
        Y = paged_v.permute(0, 1, 3, 2, 4).float()
        Y = Y - Y.mean(dim=3, keepdim=True)
        Z = Y @ Gh.to(Y.dtype)
        Mo = Z @ Z.transpose(-1, -2)
        Xg = X @ X.transpose(-1, -2)
        tr_key = (X * X).sum(dim=(-1, -2)).clamp_min(1e-30)
        tr_out = (Mo * Xg).sum(dim=(-1, -2)).clamp_min(1e-30)
        eye = torch.eye(X.shape[-2], device=X.device, dtype=X.dtype)
        A = ((1.0 - eta) / tr_key)[..., None, None] * eye \
            + (eta / tr_out)[..., None, None] * Mo
        # normalise A to unit mean diagonal -- without this the absolute scale
        # of A rides into the eigenvalues and corrupts near-degenerate pages
        A = A * (X.shape[-2] / A.diagonal(dim1=-2, dim2=-1).sum(-1)
                 .clamp_min(1e-30))[..., None, None]
        a, Q = torch.linalg.eigh(A)
        Ah = (Q * a.clamp_min(0).sqrt().unsqueeze(-2)) @ Q.transpose(-1, -2)
        Xw = Ah @ X
    else:
        Xw = X

    Gm = Xw @ Xw.transpose(-1, -2)
    S2, U = torch.linalg.eigh(Gm)
    # centring costs one dof, so rank(X) <= page_size-1
    r = min(rank, U.shape[-1] - 1, X.shape[-1])
    S = S2[..., -r:].clamp_min(1e-10).sqrt()
    U8 = U[..., -r:]
    basis = Xw.transpose(-1, -2) @ U8 / S.unsqueeze(-2)
    coef = X @ basis
    return mu.to(dt), basis.to(dt), coef.to(dt)


def oval_sel(q, basis, coef, mu, GQA_policy, num_heads, num_kv_heads):
    """Page scores, matching quest_sel's contract exactly: [bsz, n_kv, n_pages].

    q: [bsz, 1, num_heads, head_dim]; basis [bsz,P,H,d,r]; coef [bsz,P,H,page,r];
    mu [bsz,P,H,d].  The per-query-head score is the page's log attention mass
    under the rank-r reconstruction,  logsumexp_t (R_t . B^T q + mu . q),  and
    the SAME GQA reduction as quest_sel is then applied so the two are
    interchangeable downstream.
    """
    import torch.nn.functional as F
    g = num_heads // num_kv_heads
    qv = q[:, 0].float()                                     # [bsz, n_heads, d]
    qg = qv.reshape(qv.shape[0], num_kv_heads, g, qv.shape[-1])
    b_, c_, m_ = basis.float(), coef.float(), mu.float()
    u = torch.einsum('bphdr,bhgd->bphgr', b_, qg)            # [b,P,H,g,r]
    c = torch.einsum('bphd,bhgd->bphg', m_, qg)              # [b,P,H,g]
    tok = torch.einsum('bphtr,bphgr->bphgt', c_, u) + c.unsqueeze(-1)
    s = torch.logsumexp(tok, dim=-1)                         # [b,P,H,g]
    # -> [bsz, num_heads, P], the layout quest_sel produces before its reduction
    max_qk = s.permute(0, 2, 3, 1).reshape(s.shape[0], num_heads, -1)

    if GQA_policy == "maxS":
        return max_qk.reshape(max_qk.shape[0], num_kv_heads, g, -1).max(dim=-2).values
    if GQA_policy == "avgS":
        return max_qk.reshape(max_qk.shape[0], num_kv_heads, g, -1).mean(dim=-2)
    if GQA_policy == "maxSM":
        max_qk = F.softmax(max_qk, dim=-1)
        return max_qk.reshape(max_qk.shape[0], num_kv_heads, g, -1).max(dim=-2).values
    if GQA_policy == "avgSM":
        max_qk = F.softmax(max_qk, dim=-1)
        return max_qk.reshape(max_qk.shape[0], num_kv_heads, g, -1).mean(dim=-2)
    if GQA_policy == "avgSdM":
        import math
        max_qk = F.softmax(max_qk / math.sqrt(q.shape[-1]), dim=-1)
        return max_qk.reshape(max_qk.shape[0], num_kv_heads, g, -1).mean(dim=-2)
    if GQA_policy in ("maxQ", "avgQ"):
        # these pool the QUERY before scoring; with a per-page basis that would
        # change the math rather than just the reduction, so refuse loudly
        raise NotImplementedError(
            f"GQA_policy={GQA_policy} pools queries before scoring; OVAL scores "
            "per query head. Use avgSM (the paper's setting) or another S-policy.")
    raise AssertionError(f"unknown GQA_policy {GQA_policy}")
