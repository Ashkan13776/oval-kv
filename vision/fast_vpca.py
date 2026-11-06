"""Token-domain implementation of build_value_summary for plain PCA (Gh = None).

oval.build_value_summary forms the d x d channel Gram M = Z^T Z (d = 128) and
calls a batched eigh on it. Batched 128x128 eigh has no fast CUDA path: on a
13K-token video prefill (~430 pages x 4 KV heads per layer) it takes ~11 s per
layer, ~300 s per sample, all on the first decode step. Decode-time builds are
one page at a time, so the reasoning runs never hit this.

For Gh = None (plain value PCA, the full pipeline's --vbasis pca), the top-r
eigenvectors of Z^T Z are U = Z^T V diag(lam)^{-1/2}, with (lam, V) the top-r
eigenpairs of the n x n token Gram Z Z^T (n = 32), whose nonzero eigenvalues are
the same. Then T = Z U = V diag(lam)^{1/2} and D = U, and the reconstruction
nu + T D^T = nu + Z U U^T is the same projection of the same page. Same
degenerate-direction guard. Any other case (Gh given) goes to the original.
Equivalence on real pages: vision/fast_vpca.py --check.
"""
import torch


def build_value_summary_tokdomain(orig):
    def f(paged_k, paged_v, Gh, Ghi, eta, rank, eps=1e-30):
        if Gh is not None:
            return orig(paged_k, paged_v, Gh, Ghi, eta, rank, eps)
        dt = paged_v.dtype
        Y = paged_v.permute(0, 1, 3, 2, 4).float()          # [B,P,H,n,d]
        nu = Y.mean(dim=3)
        Z = Y - nu.unsqueeze(3)
        n, d = Z.shape[-2], Z.shape[-1]
        r = min(rank, n - 1, d)
        lam, V = torch.linalg.eigh(Z @ Z.transpose(-1, -2))  # ascending, n x n
        lam_r, V_r = lam[..., -r:], V[..., -r:]
        keep = (lam_r > lam[..., -1:].clamp_min(0) * 1e-10).unsqueeze(-2)
        s = lam_r.clamp_min(1e-30).sqrt()
        U = (Z.transpose(-1, -2) @ V_r) / s.unsqueeze(-2)    # [.., d, r], orthonormal columns
        U = U * keep.to(U.dtype)
        T = Z @ U
        return nu.to(dt), U.to(dt), T.to(dt)
    f.__wrapped_original__ = orig
    return f


def install():
    """Swap the harness's value-basis builder for the token-domain one."""
    from kvc.patch import dynamic_attention as DA, oval as OV
    if not hasattr(DA._oval_vbuild, "__wrapped_original__"):
        DA._oval_vbuild = build_value_summary_tokdomain(OV.build_value_summary)


if __name__ == "__main__":
    import os, sys, time
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "FreeKV", "accuracy"))
    from kvc.patch import oval as OV
    cap = torch.load(sys.argv[2] if len(sys.argv) > 2 else
                     os.path.join(os.path.dirname(__file__), "..", "logs", "eta_rules", "capture",
                                  "llava-video-7b_VCAL_p0.pt"), map_location="cpu")
    fast = build_value_summary_tokdomain(OV.build_value_summary)
    worst = 0.0
    for li in (1, 7, 14, 21, 27):
        K, V = cap["K"][li].cuda(), cap["V"][li].cuda()          # [H, T, d] bf16
        H, T, d = K.shape
        P = (T - 256) // 32
        Kp = K[:, 128:128 + P * 32].reshape(H, P, 32, d).permute(1, 2, 0, 3)[None]
        Vp = V[:, 128:128 + P * 32].reshape(H, P, 32, d).permute(1, 2, 0, 3)[None]
        for r in (16,):
            t = time.time(); a = OV.build_value_summary(Kp, Vp, None, None, 0.0, r); torch.cuda.synchronize(); t1 = time.time() - t
            t = time.time(); b = fast(Kp, Vp, None, None, 0.0, r); torch.cuda.synchronize(); t2 = time.time() - t
            ra = a[2].float() @ a[1].float().transpose(-1, -2)
            rb = b[2].float() @ b[1].float().transpose(-1, -2)
            exact = Vp.permute(0, 1, 3, 2, 4).float() - a[0].float().unsqueeze(3)
            rel = ((ra - rb).norm() / ra.norm()).item()
            err_a = ((ra - exact).norm() / exact.norm()).item(); err_b = ((rb - exact).norm() / exact.norm()).item()
            worst = max(worst, rel)
            print(f"layer {li} r={r}: orig {t1:.2f}s fast {t2:.3f}s | recon diff {rel:.2e} | "
                  f"value error orig {err_a:.4%} fast {err_b:.4%} | mu equal {torch.equal(a[0], b[0])}")
    print("worst relative difference of the rebuilt values:", worst)
