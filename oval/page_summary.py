"""OVAL: our output-aware rank-r page basis, for FreeKV's PERFORMANCE engine.

eta=0 recovers the plain key-PCA basis (what LOCKS uses); eta>0 mixes in the
output-coupled metric. LOCKS is therefore the eta=0 SPECIAL CASE of this
method, not the method itself.

Ported into the offloading engine so it can be timed end-to-end against their
min/max digest (Fig. 6). Kept in its own module so their digest path is
untouched.

Why it needs separate storage
-----------------------------
Their digest lives in the SAME KvPool as the KV cache: a digest page has shape
[2, page_size, n_kv, head_dim], i.e. exactly a (max, min) pair pretending to be
a (K, V) page, which is why `append_paged_kv_cache(*digest, ...)` works. Our
summary is basis + coefficients + centroid and does not fit that mould, so we
allocate our own buffers indexed directly by SOURCE PAGE INDEX. That is sound
because the digest sequence only ever grows -- digests are never evicted (only
the KV pages they summarise are), so page i keeps slot i for the whole run.

Quantized storage (ported from LOCKS' own r8i4 records)
------------------------------------------------------
LOCKS stores this SAME rank-r information in int4/int8, not bf16 (see
locks/selection/r8i4_build.py):

    V   (d, r)     int4, per-COLUMN bf16 scale (absmax/7, clamp -8..7),
                   packed column-major, two nibbles per byte, lo = even d
    C   (page, r)  int8, per-TOKEN bf16 scale (absmax/127)
    mu  (d)        int8, per-PAGE  bf16 scale (absmax/127)

We shipped bf16 and so paid 2.88x more bytes than the method we extend, for
the same numbers. Per page per KV head:

    theirs (min/max) 2*head_dim * 2B                      =  512 B
    ours   bf16      (d*r + page*r + d) * 2B              = 2816 B   (5.50x)
    ours   quantized r*(d/2+2) + page*(r+2) + (d+2)       =  978 B   (1.91x)

The scale for each row is stored INLINE in that row's trailing 2 bytes, which
keeps the tensor count (and therefore every C++/python signature) unchanged and
puts each scale in the same cache line as the data it scales.

The scorer is bandwidth-bound in these bytes -- measured 0.1740 ms/layer for
their 2048 B/page digest vs 0.4438 ms/layer for our 11264 B/page bf16 -- so
this is simultaneously the memory fix and ~80% of the latency fix.
"""
import os

import torch


def quant_enabled():
    """int4/int8 records by default; OVAL_QUANT=0 keeps the bf16 records so the
    two can be measured head-to-head in a single build."""
    return os.environ.get("OVAL_QUANT", "1") != "0"


class OabSummaryStore:
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
        # Quantized slabs, scales inline in each row's trailing 2 bytes.
        #   basis [L,B,P,H,r,   d/2+2]  u8 : r rows of d/2 nibble-pairs + bf16 scale
        #   coef  [L,B,P,H,page,r  +2]  i8 : page rows of r int8 + bf16 scale
        #   mu    [L,B,P,H,d+2]         i8 : d int8 + bf16 scale
        self.quant = quant_enabled()
        if self.quant:
            u8 = dict(dtype=torch.uint8, device=device)
            i8 = dict(dtype=torch.int8, device=device)
            self.vrow = head_dim // 2 + 2
            self.crow = rank + 2
            self.basis = torch.zeros(n_layers, bsz, max_pages, n_kv_heads,
                                     rank, self.vrow, **u8)
            self.mu = torch.zeros(n_layers, bsz, max_pages, n_kv_heads,
                                  head_dim + 2, **i8)
        else:
            opt = dict(dtype=dtype, device=device)
            self.basis = torch.zeros(n_layers, bsz, max_pages, n_kv_heads,
                                     head_dim, rank, **opt)
            self.mu = torch.zeros(n_layers, bsz, max_pages, n_kv_heads,
                                  head_dim, **opt)
        self.coef = None          # allocated on first build (needs page_size)
        self.n_pages = [0] * n_layers
        # per-layer [H, d, d] = G^{1/2}, only populated when eta > 0
        self.Gh = [None] * n_layers

    def _ensure_coef(self, page_size):
        if self.coef is None:
            if self.quant:
                self.coef = torch.zeros(self.n_layers, self.bsz, self.max_pages,
                                        self.n_kv, page_size, self.crow,
                                        dtype=torch.int8, device=self.device)
            else:
                self.coef = torch.zeros(self.n_layers, self.bsz, self.max_pages,
                                        self.n_kv, page_size, self.rank,
                                        dtype=self.dtype, device=self.device)

    def bytes(self):
        n = self.basis.numel() + self.mu.numel()
        if self.coef is not None:
            n += self.coef.numel()
        return n if self.quant else n * self.basis.element_size()


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
    if not quant_enabled():
        return mu.to(dt), basis.to(dt), coef.to(dt)
    return quantize(mu, basis, coef, rank)


def quantize(mu, basis, coef, rank):
    """Pack (mu, basis, coef) into LOCKS' int4/int8 records, scales inline.

    Same scheme as locks/selection/r8i4_build.py:
      basis int4, per-COLUMN scale absmax/7, clamped -8..7, packed two nibbles
        per byte with the LO nibble holding the EVEN head_dim row;
      coef  int8, per-TOKEN scale absmax/127;
      mu    int8, per-PAGE  scale absmax/127.
    Each row's bf16 scale is written into that row's trailing 2 bytes, so the
    tensor count -- and every downstream signature -- is unchanged.
    """
    Bsz, P, H, d, r = basis.shape
    T = coef.shape[3]

    # ---- basis: int4, per-column (over d) scale -------------------------
    vsc = (basis.abs().amax(3) / 7.0).clamp_min(1e-8)          # [B,P,H,r]
    Vq = torch.round(basis / vsc.unsqueeze(3)).clamp(-8, 7)
    Vc = Vq.permute(0, 1, 2, 4, 3).contiguous().to(torch.int8)  # [B,P,H,r,d]
    lo = (Vc[..., 0::2] & 0xF).to(torch.uint8)
    hi = (Vc[..., 1::2] & 0xF).to(torch.uint8)
    v4 = lo | (hi << 4)                                        # [B,P,H,r,d/2]
    vrow = torch.cat([v4, vsc.to(torch.bfloat16).view(torch.uint8)
                      .reshape(Bsz, P, H, r, 2)], dim=-1)      # [B,P,H,r,d/2+2]

    # ---- coef: int8, per-token (over r) scale ---------------------------
    csc = (coef.abs().amax(-1) / 127.0).clamp_min(1e-8)        # [B,P,H,T]
    Cq = torch.round(coef / csc.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    crow = torch.cat([Cq, csc.to(torch.bfloat16).view(torch.int8)
                      .reshape(Bsz, P, H, T, 2)], dim=-1)      # [B,P,H,T,r+2]

    # ---- mu: int8, per-page scale ---------------------------------------
    msc = (mu.abs().amax(-1) / 127.0).clamp_min(1e-8)          # [B,P,H]
    Mq = torch.round(mu / msc.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    mrow = torch.cat([Mq, msc.to(torch.bfloat16).view(torch.int8)
                      .reshape(Bsz, P, H, 2)], dim=-1)         # [B,P,H,d+2]
    return mrow, vrow, crow


def dequantize(mrow, vrow, crow, rank):
    """Inverse of `quantize`, for the reference scorer and for tests."""
    Bsz, P, H, r, vw = vrow.shape
    d = (vw - 2) * 2
    T = crow.shape[3]
    vsc = vrow[..., -2:].contiguous().reshape(Bsz, P, H, r * 2).view(torch.bfloat16)[
        ..., :r].float()                                       # [B,P,H,r]
    packed = vrow[..., :-2]
    lo = (packed & 0xF).to(torch.int16)
    hi = (packed >> 4).to(torch.int16)
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    Vc = torch.stack([lo, hi], dim=-1).reshape(Bsz, P, H, r, d).float()
    basis = (Vc * vsc.unsqueeze(-1)).permute(0, 1, 2, 4, 3)    # [B,P,H,d,r]

    csc = crow[..., -2:].contiguous().reshape(Bsz, P, H, T * 2).view(torch.bfloat16)[
        ..., :T].float()
    coef = crow[..., :-2].float() * csc.unsqueeze(-1)

    msc = mrow[..., -2:].contiguous().reshape(Bsz, P, H * 2).view(torch.bfloat16)[
        ..., :H].float()
    mu = mrow[..., :-2].float() * msc.unsqueeze(-1)
    return mu, basis, coef


def score_pages(q, mu, basis, coef, n_qo_heads, n_kv_heads, n_groups):
    """PyTorch reference for the CUDA scorer. Takes the QUANTIZED records and
    dequantizes, so it validates the packing as well as the arithmetic."""
    if basis.dtype == torch.uint8:
        mu, basis, coef = dequantize(mu, basis, coef, coef.shape[-1] - 2)
    else:
        mu, basis, coef = mu.float(), basis.float(), coef.float()
    B = basis.shape[0]
    qv = q.reshape(B, n_qo_heads, -1).float()
    g = n_qo_heads // n_kv_heads
    qh = qv.reshape(B, n_kv_heads, g, -1)
    u = torch.einsum("bphdr,bhgd->bphgr", basis, qh)
    c = torch.einsum("bphd,bhgd->bphg", mu, qh)
    tok = torch.einsum("bphtr,bphgr->bphgt", coef, u) + c.unsqueeze(-1)
    s = torch.logsumexp(tok, dim=-1)                      # [B,P,H,g]
    out = s.permute(0, 2, 3, 1).reshape(B, n_qo_heads, -1)
    out = out.reshape(B, n_groups, n_qo_heads // n_groups, -1).mean(2)
    return out.to(q.dtype)
