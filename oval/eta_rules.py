"""Rule 1: closed-form choice of the key-basis mixing coefficient eta.

Pure math, torch, batched over any leading dims. Nothing here touches a model
or a file; analysis/eta_rules_calibrate.py feeds rule1_endpoint the endpoint
output errors it measures on real decode state per (layer, KV head), and
analysis/test_eta_rules.py checks it against a brute-force minimiser.

eta mixes the key-reconstruction and output-sensitivity second moments of a
page in the key basis (oval/page_basis.py::build_summary). The rule returns
(eta, degenerate) where `degenerate` flags entries that hit the documented
fallback; the caller counts them.
"""
import torch


def clip01(x):
    return x.clamp(0.0, 1.0)


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
