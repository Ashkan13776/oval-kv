"""Closed-form rules for the key-basis mixing coefficient eta.

Pure math, torch, batched over any leading dims. Nothing here touches a model
or a file; analysis/eta_rules_calibrate.py feeds these functions the per-page
matrices it measures on real decode state and analysis/test_eta_rules.py
checks every rule against an independent construction on synthetic data.

Notation (one page, one KV head; r = key rank):
    A      = X^T X / tr(X^T X)             key-reconstruction second moment
    C      = X^T Y G Y^T X / tr(.)         output-sensitivity second moment
    M_eta  = (1 - eta) A + eta C
    P_eta  = Proj_r(M_eta)                 projector onto the top-r eigenvectors
X are the page's centred keys, Y its centred values, G = sum_i W_O,i^T W_O,i
over the KV head's GQA group. All rules clip eta to [0, 1].

Every rule returns (eta, degenerate) where `degenerate` is a bool tensor
flagging entries that hit a documented fallback; the caller counts them.
"""
import torch


def clip01(x):
    return x.clamp(0.0, 1.0)


def unit_trace(M, eps=1e-30):
    tr = M.diagonal(dim1=-2, dim2=-1).sum(-1)
    return M / tr.clamp_min(eps)[..., None, None]


def eigh_desc(M):
    """Eigenpairs in DESCENDING order (torch.linalg.eigh is ascending)."""
    lam, U = torch.linalg.eigh(M)
    return lam.flip(-1), U.flip(-1)


def proj_top_r(M, r):
    """Projector onto the top-r eigenvectors of a symmetric M."""
    _, U = eigh_desc(M)
    Ur = U[..., :r]
    return Ur @ Ur.transpose(-1, -2)


def mix(A, C, eta):
    """M_eta = (1-eta) A + eta C; eta is a float or a tensor broadcastable to
    A's batch dims (per page / per head)."""
    eta = torch.as_tensor(eta, dtype=A.dtype, device=A.device)
    while eta.dim() < A.dim():
        eta = eta.unsqueeze(-1)
    return (1.0 - eta) * A + eta * C


def page_matrices(X, Y, Gh):
    """A and C (unit trace, d x d) from centred keys X [.., n, d], centred
    values Y [.., n, d] and Gh = G^{1/2} [.., d, d]. Reference construction for
    the tests; the calibration script builds them in reduced coordinates."""
    A = X.transpose(-1, -2) @ X
    Z = Y @ Gh
    C = X.transpose(-1, -2) @ (Z @ Z.transpose(-1, -2)) @ X
    return unit_trace(A), unit_trace(C)


# ---------------------------------------------------------------- rule 1 ---
def rule1_endpoint(a, b, c, default=0.5, rel_tol=1e-9):
    """Endpoint error interpolation.

    a = E||D0||^2, b = E||D1||^2, c = E[D0 . D1] for the output errors D0, D1
    of the method at eta = 0 and eta = 1. Modelling the error at eta as the
    blend (1-eta) D0 + eta D1 gives the quadratic
        f(eta) = (1-eta)^2 a + eta^2 b + 2 eta (1-eta) c,
    minimised at eta = (a - c) / (a + b - 2c). The denominator is
    E||D0 - D1||^2 >= 0; when it is (numerically) zero the objective is flat
    and `default` is returned.
    """
    a, b, c = (torch.as_tensor(v, dtype=torch.float64) for v in (a, b, c))
    a, b, c = torch.broadcast_tensors(a, b, c)
    den = a + b - 2.0 * c
    scale = torch.maximum(a, b).clamp_min(1e-300)
    degenerate = den <= rel_tol * scale
    eta = torch.where(degenerate, torch.full_like(den, default),
                      (a - c) / torch.where(degenerate, torch.ones_like(den), den))
    return clip01(eta), degenerate


# ---------------------------------------------------------------- rule 2 ---
def rule2_quadratic(L0, Lm, L1):
    """Quadratic interpolation through (0, L0), (1/2, Lm), (1, L1).

    L(eta) = L0 + v eta + (u/2) eta^2 with u = 2(L0 + L1 - 2 Lm),
    v = 4 Lm - 3 L0 - L1. If u > 0 (convex) the vertex
    (3 L0 + L1 - 4 Lm) / (4 (L0 + L1 - 2 Lm)) is clipped to [0, 1]; otherwise
    the endpoint with the smaller loss is returned (ties -> 0).
    `degenerate` flags the non-convex case.
    """
    L0, Lm, L1 = (torch.as_tensor(v, dtype=torch.float64) for v in (L0, Lm, L1))
    L0, Lm, L1 = torch.broadcast_tensors(L0, Lm, L1)
    u = 2.0 * (L0 + L1 - 2.0 * Lm)
    convex = u > 0
    safe_u = torch.where(convex, u, torch.ones_like(u))
    vertex = clip01((3.0 * L0 + L1 - 4.0 * Lm) / (2.0 * safe_u))
    endpoint = torch.where(L1 < L0, torch.ones_like(u), torch.zeros_like(u))
    return torch.where(convex, vertex, endpoint), ~convex


# ---------------------------------------------------------------- rule 3 ---
def projector_derivative(A, C, eta0, r, gap_tol=1e-6):
    """P0 = Proj_r(M_eta0) and its derivative P0' = dP/deta at eta0.

    First-order perturbation of an invariant subspace: with (lambda_i, u_i)
    the eigenpairs of M0 = (1-eta0) A + eta0 C in descending order and
    D = dM/deta = C - A,
        P0' = sum_{i<=r} sum_{j>r} (u_j^T D u_i) / (lambda_i - lambda_j)
                                   (u_j u_i^T + u_i u_j^T).
    Valid when the eigengap lambda_r - lambda_{r+1} is open; pages with
    gap <= gap_tol * lambda_1 are flagged degenerate (their P0' is returned
    as computed but the caller should drop them).

    Returns P0, P0', gap, degenerate.
    """
    M0 = mix(A, C, eta0)
    D = C - A
    lam, U = eigh_desc(M0)
    Uin, Uout = U[..., :r], U[..., r:]
    lin, lout = lam[..., :r], lam[..., r:]
    # F[j, i] = u_j^T D u_i  for j > r (rows), i <= r (cols)
    F = Uout.transpose(-1, -2) @ D @ Uin                     # [.., d-r, r]
    gapmat = lin.unsqueeze(-2) - lout.unsqueeze(-1)          # lambda_i - lambda_j
    gap = lin[..., -1] - lout[..., 0]
    degenerate = gap <= gap_tol * lam[..., 0].clamp_min(0)
    safe = torch.where(gapmat.abs() > 0, gapmat, torch.ones_like(gapmat))
    K = F / safe
    cross = Uout @ K @ Uin.transpose(-1, -2)
    P0p = cross + cross.transpose(-1, -2)
    P0 = Uin @ Uin.transpose(-1, -2)
    return P0, P0p, gap, degenerate


def rule3_step(num, den, eta0, delta=None, rel_tol=1e-12, gnorm=None):
    """eta = eta0 + E[g.h] / E[h.h] from the linearised error g - (eta-eta0) h.

    num = E[g^T G h], den = E[h^T G h]. The linear model
    Err(eta) = g - (eta - eta0) h gives E||Err||_G^2 minimised at
    eta - eta0 = num / den. `delta` (if given) bounds the step to
    [max(0, eta0-delta), min(1, eta0+delta)]; without it the result is clipped
    to [0, 1]. A vanishing denominator (den <= rel_tol * gnorm, gnorm =
    E||g||^2 if given, else absolute) keeps eta0.
    """
    num, den = torch.broadcast_tensors(torch.as_tensor(num, dtype=torch.float64),
                                       torch.as_tensor(den, dtype=torch.float64))
    ref = torch.as_tensor(1.0 if gnorm is None else gnorm, dtype=torch.float64)
    degenerate = den <= rel_tol * ref
    step = torch.where(degenerate, torch.zeros_like(den),
                       num / torch.where(degenerate, torch.ones_like(den), den))
    eta = eta0 + step
    if delta is None:
        return clip01(eta), degenerate
    lo, hi = max(0.0, eta0 - delta), min(1.0, eta0 + delta)
    return eta.clamp(lo, hi), degenerate


# ---------------------------------------------------------------- rule 4 ---
def rule4_sacrifices(A, C, r):
    """a = tr((I-P_C) A) - tr((I-P_A) A) >= 0: key energy lost by using C's
    basis instead of A's; b = tr((I-P_A) C) - tr((I-P_C) C) >= 0: output
    energy lost by using A's basis instead of C's. Both are Ky Fan gaps and
    non-negative up to round-off (clamped)."""
    PA, PC = proj_top_r(A, r), proj_top_r(C, r)
    trPA_A = (PA * A).sum((-1, -2))
    trPC_A = (PC * A).sum((-1, -2))
    trPC_C = (PC * C).sum((-1, -2))
    trPA_C = (PA * C).sum((-1, -2))
    return (trPA_A - trPC_A).clamp_min(0), (trPC_C - trPA_C).clamp_min(0)


def rule4_eta(a, b, w=None, default=0.5, dim=None):
    """eta = sum_p w_p b_p / sum_p w_p (a_p + b_p) over the pages in `dim`
    (all of the last dim if None); default when the denominator is 0."""
    a, b = torch.broadcast_tensors(torch.as_tensor(a, dtype=torch.float64),
                                   torch.as_tensor(b, dtype=torch.float64))
    w = torch.ones_like(a) if w is None else torch.as_tensor(w, dtype=torch.float64).expand_as(a)
    d = -1 if dim is None else dim
    num = (w * b).sum(d)
    den = (w * (a + b)).sum(d)
    degenerate = den <= 0
    eta = torch.where(degenerate, torch.full_like(den, default),
                      num / torch.where(degenerate, torch.ones_like(den), den))
    return clip01(eta), degenerate


# ---------------------------------------------------------------- rule 5 ---
def rule5_terms(T, A, C):
    """Per-page numerator <T-A, C-A>_F and denominator ||C-A||_F^2 after
    normalising T, A, C to unit trace. eta = num/den is the least-squares
    coefficient of the mixture (1-eta) A + eta C closest to T."""
    T, A, C = unit_trace(T), unit_trace(A), unit_trace(C)
    DA = C - A
    num = ((T - A) * DA).sum((-1, -2))
    den = (DA * DA).sum((-1, -2))
    return num, den


def rule5_eta(num, den, default=0.5, dim=None, rel_tol=1e-12):
    """Shared eta over the pages in `dim`: the summed Frobenius objective
    sum_p ||T_p - A_p - eta (C_p - A_p)||^2 is minimised by
    sum_p num_p / sum_p den_p. Degenerate (C = A everywhere) -> default."""
    num, den = torch.broadcast_tensors(torch.as_tensor(num, dtype=torch.float64),
                                       torch.as_tensor(den, dtype=torch.float64))
    if dim is not None or num.dim() > 0:
        d = -1 if dim is None else dim
        num, den = num.sum(d), den.sum(d)
    degenerate = den <= rel_tol
    eta = torch.where(degenerate, torch.full_like(den, default),
                      num / torch.where(degenerate, torch.ones_like(den), den))
    return clip01(eta), degenerate


# ---------------------------------------------------------------- rule 6 ---
def rule6_features(A, C, r, eps=1e-8):
    """Per-page features [.., 3]: log((b+eps)/(a+eps)), rho, gamma_C - gamma_A
    with (a, b) the rule-4 sacrifices, rho = tr(AC)/(||A|| ||C||) and
    gamma_M = (sum of the top-r eigenvalues of M) / tr(M)."""
    a, b = rule4_sacrifices(A, C, r)
    f1 = torch.log((b + eps) / (a + eps))
    rho = (A * C).sum((-1, -2)) / (A.norm(dim=(-1, -2)) * C.norm(dim=(-1, -2))).clamp_min(1e-30)
    lamA = eigh_desc(A)[0][..., :r].sum(-1) / A.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-30)
    lamC = eigh_desc(C)[0][..., :r].sum(-1) / C.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-30)
    return torch.stack([f1, rho, lamC - lamA], -1)


def rule6_predict(beta, F):
    """sigmoid(beta0 + beta[1:] . F) for features F [.., 3]."""
    beta = torch.as_tensor(beta, dtype=F.dtype, device=F.device)
    return torch.sigmoid(beta[0] + (F * beta[1:]).sum(-1))


def rule6_fit(F, t, w=None, ridge=0.0, iters=100, tol=1e-12):
    """Fit beta (intercept + 3 slopes) by weighted soft-label logistic
    regression: minimise sum_p w_p [-t_p log s_p - (1-t_p) log(1-s_p)] +
    ridge/2 ||beta[1:]||^2 with s = sigmoid(beta . [1, F]). Convex; Newton's
    method with step halving. Targets t in [0, 1] (the offline-best eta per
    page). Returns beta [4] (float64)."""
    F = torch.as_tensor(F, dtype=torch.float64)
    t = torch.as_tensor(t, dtype=torch.float64)
    w = torch.ones_like(t) if w is None else torch.as_tensor(w, dtype=torch.float64)
    w = w / w.sum().clamp_min(1e-300)
    X = torch.cat([torch.ones_like(t)[:, None], F], 1)      # [n, 4]
    beta = torch.zeros(X.shape[1], dtype=torch.float64)
    reg = torch.zeros(X.shape[1], dtype=torch.float64); reg[1:] = ridge

    def loss(b):
        z = X @ b
        # soft-label cross-entropy, numerically stable
        ce = torch.nn.functional.softplus(z) - t * z
        return (w * ce).sum() + 0.5 * (reg * b * b).sum()

    cur = loss(beta)
    for _ in range(iters):
        s = torch.sigmoid(X @ beta)
        grad = X.T @ (w * (s - t)) + reg * beta
        H = (X * (w * s * (1 - s))[:, None]).T @ X + torch.diag(reg)
        step = torch.linalg.solve(H + 1e-12 * torch.eye(X.shape[1], dtype=torch.float64), grad)
        alpha = 1.0
        while alpha > 1e-6:
            cand = beta - alpha * step
            new = loss(cand)
            if new <= cur:
                break
            alpha *= 0.5
        if new > cur:
            break
        improved = cur - new
        beta, cur = cand, new
        if improved < tol:
            break
    return beta
