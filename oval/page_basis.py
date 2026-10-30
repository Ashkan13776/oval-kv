"""OVAL (output-aware page basis) for FreeKV's ACCURACY harness.

Mirrors source/freekv/oab_summary.py -- the perf-engine implementation -- so
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


def build_summary(paged_k, paged_v, Gh, eta, rank, qvar=None):
    """paged_k/paged_v: [B, P, page, n_kv, d] -> (mu, basis, coef).

    Returns mu [B,P,H,d], basis [B,P,H,d,r], coef [B,P,H,page,r].
    Line-for-line the perf path's build_summary. eta is a float shared by
    every head, or a float64 tensor [n_kv] with one eta per KV head
    (--eta_table); the two agree exactly when the tensor is constant.
    """
    dt = paged_k.dtype
    X = paged_k.permute(0, 1, 3, 2, 4).float()        # [B,P,H,page,d]
    mu = X.mean(dim=3)
    X = X - mu.unsqueeze(3)

    # Optional QUERY weighting. The basis exists to preserve q.k, but plain PCA
    # weights every key direction by its own energy -- including directions no
    # query ever probes. Weighting by the per-dimension query second moment
    # aims the rank budget at the directions queries actually use.
    #
    #   minimise E_q[(q.(I-P)x)^2]  ->  PCA of X W,  W = diag(sqrt(qvar))
    #   store B = W^-1 V so scoring and reconstruction still use the RAW q
    #   and the summary format is unchanged (B is simply not orthonormal).
    #
    # qvar: [H, d] per-KV-head diagonal second moment, ridge already applied.
    Wq = None
    if qvar is not None:
        Wq = qvar.clamp_min(1e-8).sqrt().to(X.dtype)        # [H,d]
        X = X * Wq[None, None, :, None, :]                  # whiten the keys

    # eta is a python float (one value for every head) or a float64 tensor
    # [H] holding one value per KV head (--eta_table). The per-head path
    # reproduces the scalar arithmetic bit for bit: `float / tensor` is
    # Tensor.__rtruediv__, i.e. tr.reciprocal() * float32(1-eta), so the
    # table path forms the same reciprocal and multiplies it by the float64
    # entries cast to float32 at the same point (verified bitwise in
    # analysis/test_eta_rules.py, check 7). Heads whose entry is exactly 0
    # take the plain-PCA branch (Xw = X) instead of an identity whitening, so
    # they match the scalar eta=0 path exactly too.
    eta_t = None
    if torch.is_tensor(eta):
        eta_t = eta.detach().to(torch.float64).reshape(-1)
        if eta_t.numel() != X.shape[2]:
            raise ValueError(f"per-head eta has {eta_t.numel()} entries for "
                             f"{X.shape[2]} KV heads")
        use_out = bool((eta_t > 0.0).any()) and Gh is not None
    else:
        use_out = eta > 0.0 and Gh is not None
    if use_out:
        Y = paged_v.permute(0, 1, 3, 2, 4).float()
        Y = Y - Y.mean(dim=3, keepdim=True)
        Z = Y @ Gh.to(Y.dtype)
        Mo = Z @ Z.transpose(-1, -2)
        Xg = X @ X.transpose(-1, -2)
        tr_key = (X * X).sum(dim=(-1, -2)).clamp_min(1e-30)
        tr_out = (Mo * Xg).sum(dim=(-1, -2)).clamp_min(1e-30)
        eye = torch.eye(X.shape[-2], device=X.device, dtype=X.dtype)
        if eta_t is None:
            A = ((1.0 - eta) / tr_key)[..., None, None] * eye \
                + (eta / tr_out)[..., None, None] * Mo
        else:
            w1 = (1.0 - eta_t).to(X.dtype).to(X.device)          # [H]
            w2 = eta_t.to(X.dtype).to(X.device)                  # [H]
            A = (tr_key.reciprocal() * w1)[..., None, None] * eye \
                + (tr_out.reciprocal() * w2)[..., None, None] * Mo
        # normalise A to unit mean diagonal -- without this the absolute scale
        # of A rides into the eigenvalues and corrupts near-degenerate pages
        A = A * (X.shape[-2] / A.diagonal(dim1=-2, dim2=-1).sum(-1)
                 .clamp_min(1e-30))[..., None, None]
        a, Q = torch.linalg.eigh(A)
        Ah = (Q * a.clamp_min(0).sqrt().unsqueeze(-2)) @ Q.transpose(-1, -2)
        Xw = Ah @ X
        if eta_t is not None:
            zero = (eta_t == 0.0).to(X.device)                   # [H]
            if bool(zero.any()):
                # in place on the matmul output: torch.where would adopt
                # X's permuted strides and MKL then takes a different gemm
                # path, which breaks bitwise agreement with the scalar path
                Xw[:, :, zero] = X[:, :, zero]
    else:
        Xw = X

    Gm = Xw @ Xw.transpose(-1, -2)
    S2, U = torch.linalg.eigh(Gm)
    # centring costs one dof, so rank(X) <= page_size-1
    r = min(rank, U.shape[-1] - 1, X.shape[-1])
    S2r = S2[..., -r:]
    # A page whose true rank is below r leaves near-zero eigenvalues in this
    # block. Clamping the DIVISOR divides a ~0 numerator by 1e-10 and amplifies
    # round-off into a garbage basis column: at r=31 that gave a 126% key
    # reconstruction error on layer 0, whose pages are rank-deficient by ~5
    # directions. Drop those directions instead -- they carry no energy, so
    # zeroing them is exact, and the reconstruction becomes exact at full rank.
    keep = (S2r > S2[..., -1:].clamp_min(0) * 1e-8).unsqueeze(-2)
    S = S2r.clamp_min(1e-10).sqrt()
    U8 = U[..., -r:]
    basis = Xw.transpose(-1, -2) @ U8 / S.unsqueeze(-2)
    basis = basis * keep.to(basis.dtype)
    coef = X @ basis
    if Wq is not None:
        # map the basis back out of the whitened frame: B <- W^-1 V. coef is
        # already in the whitened frame (X above is whitened), so coef @ B^T
        # reconstructs the ORIGINAL centred keys and q.khat uses the raw q.
        basis = basis / Wq[None, None, :, :, None]
    return mu.to(dt), basis.to(dt), coef.to(dt)


def metric_inv_sqrt(Gh, rcond=1e-6):
    """G^{-1/2} per KV head from publish_metric's G^{1/2}, computed once per layer.

    Pseudo-inverse: directions of G below rcond * max are dropped, which
    discards centred value components invisible to W_O.
    """
    s, Q = torch.linalg.eigh(Gh.float())                    # s = eig(G)^{1/2}
    keep = s > s[..., -1:].clamp_min(0) * rcond
    si = torch.where(keep, 1.0 / s.clamp_min(1e-30), torch.zeros_like(s))
    return (Q * si.unsqueeze(-2)) @ Q.transpose(-1, -2)


def build_value_summary(paged_k, paged_v, Gh, Ghi, eta, rank, eps=1e-30):
    """Output-aware rank-r VALUE record: (nu, D, T), same layout as the key record.

    paged_k/paged_v: [B, P, page, n_kv, d] -> nu [B,P,H,d], D [B,P,H,d,r],
    T [B,P,H,page,r], reconstructed as  V_hat = 1 nu^T + T D^T.

    G = W_O^T W_O summed over each KV head's GQA group -- the same G as the key
    basis (Gh = G^{1/2} from publish_metric, Ghi = G^{-1/2}). With X, Y the
    page's centred keys and values and Z = Y G^{1/2}:

        M_rec = Z^T Z                       value reconstruction in the G metric
        M_out = Z^T X X^T Z = A A^T,  A = G^{1/2} Y^T X
                (the key output matrix is X^T Y G Y^T X = A^T A)
        M_eta = (1-eta) M_rec / tr(M_rec) + eta M_out / tr(M_out)

    U = top-r eigenvectors of M_eta; store T = Z U, D = G^{-1/2} U.

    Both terms put G on the channel side, so the hybrid objective
    (1-eta) ||E G^{1/2}||^2 / tr(M_rec) + eta tr(E^T X X^T E G) / tr(M_out)
    is || M_t^{1/2} E G^{1/2} ||^2 with M_t = (1-eta) I/tr(M_rec) + eta XX^T/tr(M_out)
    -- separable -- and this record is its exact rank-r optimum over all
    reconstructions, not only over projections. eta=0 is PCA in the G metric
    (peaked attention: one token's value error through W_O); eta=1 is the
    first-order output error under diffuse attention.

    Gh = Ghi = None is plain (Euclidean) value PCA: G = I, and eta is ignored.
    """
    dt = paged_v.dtype
    Y = paged_v.permute(0, 1, 3, 2, 4).float()            # [B,P,H,n,d]
    nu = Y.mean(dim=3)
    Y = Y - nu.unsqueeze(3)
    n, d = Y.shape[-2], Y.shape[-1]
    r = min(rank, n - 1, d)                                # centring costs one dof

    Z = Y if Gh is None else Y @ Gh.float()                # Y G^{1/2}  (Gh symmetric)
    M = Z.transpose(-1, -2) @ Z                            # M_rec
    if eta > 0.0 and Gh is not None:
        X = paged_k.permute(0, 1, 3, 2, 4).float()
        X = X - X.mean(dim=3, keepdim=True)
        AT = X.transpose(-1, -2) @ Z                       # A^T = X^T Y G^{1/2}  [.., d, d]
        Mo = AT.transpose(-1, -2) @ AT                     # M_out = A A^T
        tr_r = M.diagonal(dim1=-2, dim2=-1).sum(-1) + eps
        tr_o = Mo.diagonal(dim1=-2, dim2=-1).sum(-1) + eps
        M = ((1.0 - eta) / tr_r)[..., None, None] * M + (eta / tr_o)[..., None, None] * Mo
    lam, Ue = torch.linalg.eigh(M)                         # ascending
    U = Ue[..., -r:]                                       # top-r = LAST r columns
    # same degenerate-direction guard as build_summary: a direction with no
    # energy is zeroed, never amplified
    keep = (lam[..., -r:] > lam[..., -1:].clamp_min(0) * 1e-10).unsqueeze(-2)
    U = U * keep.to(U.dtype)
    T = Z @ U                                              # [.., n, r]
    D = U if Ghi is None else Ghi.float() @ U              # [.., d, r]
    return nu.to(dt), D.to(dt), T.to(dt)


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


# ---------------------------------------------------------------------------
# Per-(layer, KV head) eta tables (--eta_table). A table is a JSON file with
# "eta": [[...] * n_kv_heads] * n_layers in the harness's dynamic-attention
# KV-head order. With sparsity 1 (every head dynamic, the only mode this repo
# runs) reorder_linear_weights keeps the model's head order, so row h is the
# model's KV head h and matches publish_metric's G[h]. load_eta_table refuses
# anything else.

def eta_table_hash(eta):
    """8-hex content hash of a table, over the entries rounded to 4 decimals.

    Used as the output-path tag (_etab<hash>) in place of _eta<E>, so two rules
    that select the same table share one cell and tables differing anywhere
    never collide. The hash covers the VALUES only, not the file or rule name.
    """
    import hashlib, json
    canon = json.dumps([[round(float(v), 4) for v in row] for row in eta],
                       separators=(",", ":"))
    return hashlib.sha1(canon.encode()).hexdigest()[:8]


def read_eta_table(path):
    """Load and validate a table file -> (eta list of lists, hash)."""
    import json, math
    tab = json.load(open(path))
    eta = tab["eta"] if isinstance(tab, dict) else tab
    if not eta or not all(isinstance(r, list) and len(r) == len(eta[0]) for r in eta):
        raise ValueError(f"{path}: 'eta' must be a rectangular list of lists")
    for row in eta:
        for v in row:
            v = float(v)
            if not math.isfinite(v) or v < 0.0 or v > 1.0:
                raise ValueError(f"{path}: eta entries must lie in [0, 1], got {v}")
    return eta, eta_table_hash(eta)


def load_eta_table(path, layer_idx, n_layers, n_kv_heads, n_full_heads):
    """The per-head eta row for one layer as a float64 CPU tensor [n_kv_heads]."""
    eta, _ = read_eta_table(path)
    if len(eta) != n_layers:
        raise ValueError(f"{path}: table has {len(eta)} layers, model has {n_layers}")
    if len(eta[0]) != n_kv_heads:
        raise ValueError(f"{path}: table has {len(eta[0])} KV heads per layer, "
                         f"the model's dynamic attention has {n_kv_heads}")
    if n_full_heads != 0:
        raise ValueError("--eta_table assumes every KV head is dynamic (sparsity 1); "
                         f"layer {layer_idx} has {n_full_heads} full heads, after which "
                         "reorder_linear_weights permutes the KV heads and the table "
                         "rows would not line up")
    return torch.tensor([float(v) for v in eta[layer_idx]], dtype=torch.float64)
