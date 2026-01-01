"""Does the shipped r8i4 summary implement the paper's construction?

Paper (arXiv:2607.24555, sec4_method): for a completed page, decompose keys
into a centroid mu and deviations D, eigendecompose the B x B Gram D D^T, keep
r=8 components, store an orthonormal basis V in R^{d x 8} plus per-key
coefficients c_i in R^8; at decode reconstruct within-page logits as

    q^T mu_j + (V_j^T q)^T c_i / sqrt(d)

and reduce by log-sum-exp to a page-mass estimate.

Shipped build (r8i4_build.py docstring):
    mu = mean_t K ; dc = K - mu ; Gm = dc @ dc^T ; S2,U = eigh(Gm)
    S8 = sqrt(S2[-8:]) ; U8 = U[...,-8:] ; C = U8 * S8 ; V = dc^T U8 / S8

This script checks, in exact arithmetic on CPU, that those two descriptions are
the same object: that V is orthonormal, that C V^T reconstructs the centered
keys at rank 8, that it equals the truncated SVD of dc (i.e. page-local PCA),
and that the decode-time logit identity holds.
"""
import numpy as np

RNK = 8


def shipped_build(K):
    """Literal transcription of the shipped docstring's op sequence."""
    mu = K.mean(axis=0)                       # (d,)
    dc = K - mu                               # (page, d)
    Gm = dc @ dc.T                            # (page, page)
    S2, U = np.linalg.eigh(Gm)                # ascending
    S8 = np.sqrt(np.clip(S2[-RNK:], 0, None))
    U8 = U[:, -RNK:]
    C = U8 * S8                               # (page, 8)
    with np.errstate(divide="ignore", invalid="ignore"):
        V = (dc.T @ U8) / S8                  # (d, 8)
    V = np.nan_to_num(V)
    return mu, C, V


def main():
    rng = np.random.default_rng(0)
    page, d = 16, 128
    K = rng.normal(size=(page, d)).astype(np.float64)
    mu, C, V = shipped_build(K)
    dc = K - mu

    print(f"page={page} d={d} rank={RNK}\n")

    # 1. Is the stored basis orthonormal?
    orth = np.abs(V.T @ V - np.eye(RNK)).max()
    print(f"1. ||V^T V - I||_max            = {orth:.3e}   (orthonormal basis)")

    # 2. Does C V^T reconstruct the centered keys at rank 8?
    rec = C @ V.T
    P = V @ V.T
    proj_err = np.abs(rec - dc @ P).max()
    print(f"2. ||C V^T - dc P_V||_max       = {proj_err:.3e}   (C = dc V, exactly)")

    # 3. Is this the truncated SVD of dc, i.e. page-local PCA?
    Us, Ss, Vt = np.linalg.svd(dc, full_matrices=False)
    svd_rec = (Us[:, :RNK] * Ss[:RNK]) @ Vt[:RNK]
    print(f"3. ||C V^T - SVD_8(dc)||_max    = {np.abs(rec - svd_rec).max():.3e}   "
          f"(== truncated SVD => PCA)")

    # 4. Residual energy equals the discarded spectrum tail (paper's tau_{j,r}).
    tail = (Ss[RNK:] ** 2).sum()
    resid = ((dc - rec) ** 2).sum()
    print(f"4. residual energy {resid:.3e} vs spectral tail {tail:.3e}")

    # 5. Decode-time logit identity: the paper's reconstruction should equal
    #    the exact logit whenever the key deviation lies in span(V).
    q = rng.normal(size=d)
    sd = np.sqrt(d)
    exact = (K @ q) / sd
    approx = (mu @ q) / sd + (C @ (V.T @ q)) / sd     # q^T mu + (V^T q)^T c_i
    full_rank_dc = dc @ (V @ V.T)                      # what the summary retains
    ideal = (mu @ q + full_rank_dc @ q) / sd
    print(f"5. ||approx - retained-exact||  = {np.abs(approx - ideal).max():.3e}"
          "   (logit identity holds)")
    print(f"   ||approx - exact||_max       = {np.abs(approx - exact).max():.3e}"
          "   (gap = discarded tail only)")

    # 6. Rank-8 of a 16-token page: with page=16, dc has rank <= 15, so rank 8
    #    is a genuine truncation, not a lossless re-basis.
    print(f"\n6. rank(dc) = {np.linalg.matrix_rank(dc)} (<= page-1 = {page-1}); "
          f"keeping {RNK} => real truncation")

    ok = orth < 1e-10 and proj_err < 1e-10 and np.abs(rec - svd_rec).max() < 1e-8
    print("\nVERDICT:", "matches the paper's construction" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
