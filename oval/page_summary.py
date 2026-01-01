"""Rank-r output-aware page summary for FreeKV's PERFORMANCE engine.

This is the same page representation as accuracy/kvc/patch/locks_rep.py, ported
into the offloading engine so it can be timed end-to-end against their min/max
digest (Fig. 6). Kept in its own module so their digest path is untouched.

Why it needs separate storage
-----------------------------
Their digest lives in the SAME KvPool as the KV cache: a digest page has shape
[2, page_size, n_kv, head_dim], i.e. exactly a (max, min) pair pretending to be
a (K, V) page, which is why `append_paged_kv_cache(*digest, ...)` works. Our
summary is basis + coefficients + centroid and does not fit that mould, so we
allocate our own buffers indexed directly by SOURCE PAGE INDEX. That is sound
because the digest sequence only ever grows -- digests are never evicted (only
the KV pages they summarise are), so page i keeps slot i for the whole run.

Cost vs their digest, per page per KV head:
    theirs  2*head_dim                      =  256 values
    ours    head_dim*r + page_size*r + head_dim = 1408 values   (5.5x)
and the score is 5.5x the arithmetic. That is the price of the representation;
it is what Fig. 6 is meant to expose.
"""
import torch


class LocksSummaryStore:
    """basis / coefficients / centroid for every (layer, batch, source page)."""

    def __init__(self, n_layers, bsz, max_pages, n_kv_heads, head_dim,
                 rank, eta, dtype, device):
        self.n_layers = n_layers
        self.bsz = bsz
        self.max_pages = max_pages
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.rank = rank
        self.eta = eta
        self.dtype = dtype
        self.device = device
        opt = dict(dtype=dtype, device=device)
        # [L, B, P, H, d, r] / [L, B, P, H, page, r] / [L, B, P, H, d]
        self.basis = torch.zeros(n_layers, bsz, max_pages, n_kv_heads,
                                 head_dim, rank, **opt)
        self.coef = None          # allocated on first build (needs page_size)
        self.mu = torch.zeros(n_layers, bsz, max_pages, n_kv_heads,
                              head_dim, **opt)
        self.n_pages = [0] * n_layers
        # per-layer [H, d, d] = G^{1/2}, only populated when eta > 0
        self.Gh = [None] * n_layers

    def _ensure_coef(self, page_size):
        if self.coef is None:
            self.coef = torch.zeros(self.n_layers, self.bsz, self.max_pages,
                                    self.n_kv, page_size, self.rank,
                                    dtype=self.dtype, device=self.device)

    def bytes(self):
        n = self.basis.numel() + self.mu.numel()
        if self.coef is not None:
            n += self.coef.numel()
        return n * self.basis.element_size()


def publish_metric(o_proj_weight, n_kv_heads, head_dim, n_kv_groups):
    """G^{1/2} per KV head from the attention output projection.

    G_k = sum_{g in group(k)} W_g^T W_g, then the symmetric PSD square root, so
    the page build can fold it into Y once and work with plain Grams after.
    """
    W = o_proj_weight.float()
    n_q = W.shape[1] // head_dim
    Wh = W.reshape(W.shape[0], n_q, head_dim)
    G = torch.einsum("dgi,dgj->gij", Wh, Wh)
    G = G.reshape(n_kv_heads, n_kv_groups, head_dim, head_dim).sum(dim=1)
    ev, Q = torch.linalg.eigh(G.double())
    Gh = (Q * ev.clamp_min(0).sqrt().unsqueeze(-2)) @ Q.transpose(-1, -2)
    return Gh.to(o_proj_weight.dtype)


def build_summary(paged_k, paged_v, Gh, eta, rank):
    """Rank-r summary of a batch of pages.

    paged_k/paged_v : [B, P, page_size, n_kv, head_dim]
    returns mu [B,P,H,d], basis [B,P,H,d,r], coef [B,P,H,page,r]

    Solved in the page_size x page_size Gram, not the d x d covariance:
    page_size (32) << head_dim (128), so the Gram route is cheaper and exactly
    equivalent on the leading directions.
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
    # centring costs one dof, so rank(X) <= page_size-1; asking for page_size
    # divides by a ~zero eigenvalue and returns garbage
    r = min(rank, U.shape[-1] - 1, X.shape[-1])
    S = S2[..., -r:].clamp_min(1e-10).sqrt()
    U8 = U[..., -r:]
    basis = Xw.transpose(-1, -2) @ U8 / S.unsqueeze(-2)   # [B,P,H,d,r]
    coef = X @ basis                                       # [B,P,H,page,r]
    return mu.to(dt), basis.to(dt), coef.to(dt)


def score_pages(q, mu, basis, coef, n_qo_heads, n_kv_heads, n_groups):
    """Per-(query head, page) score from the rank-r summary.

    q     : [B, n_qo_heads, d]     (single decode step)
    mu    : [B, P, H, d]
    basis : [B, P, H, d, r]
    coef  : [B, P, H, page, r]
    returns [B, n_groups, P] -- their contract. estimate.cu ends with
        o.reshape({bsz, n_groups, n_qo_heads/n_groups, L}).mean(2)
    so the group combine is a plain MEAN over each group's query heads (the
    perf engine, unlike the accuracy harness, has no softmax/avgSM option).

    score = logsumexp_t (coef_t . (basis^T q) + mu . q), the page's log
    attention mass under the rank-r reconstruction.
    """
    B, P, H, d, r = basis.shape
    g = n_qo_heads // n_kv_heads
    qh = q.reshape(B, H, g, d).float()
    qb = torch.einsum("bphdr,bhgd->bphgr", basis.float(), qh)
    tok = torch.einsum("bphtr,bphgr->bphgt", coef.float(), qb)
    tok = tok + torch.einsum("bphd,bhgd->bphg", mu.float(), qh).unsqueeze(-1)
    s = torch.logsumexp(tok, dim=-1)                       # [B,P,H,g]
    # -> [B, n_qo_heads, P] in their head order (head = kv_head*g + j), then
    #    their exact group reduction
    s = s.permute(0, 2, 3, 1).reshape(B, H * g, P)
    out = s.reshape(B, n_groups, (H * g) // n_groups, P).mean(2)
    # their select_topk dispatches on half/bfloat16 only
    return out.to(basis.dtype)
