"""CPU unit tests for oval/eta_rules.py and the per-head eta path of
oval/page_basis.py::build_summary. Run: python analysis/test_eta_rules.py

Each check compares a rule against an independent construction:
  1. rule 1 is exact when the error is a linear blend of the endpoint errors
  2. rule 2 recovers the vertex of an exact convex quadratic; non-convex ->
     the better endpoint
  3. rule 3: P0' matches a central finite difference of Proj_r(M_eta) at eta0
     (eigengap open), and the eta step is the numerical minimiser of the
     linearised quadratic E||g - (eta-eta0) h||^2; vanishing denominator and
     closed eigengap are flagged
  4. rule 4: a, b >= 0; A = C -> a = b = 0 -> default; C's basis orthogonal to
     A's -> eta = tr_r(C)/(tr_r(A)+tr_r(C)) by construction, limiting cases
  5. rule 5 recovers eta exactly when T is itself the mixture (1-eta)A + eta C,
     per page and shared over pages with different (A, C)
  6. rule 6: the soft-label logistic fit recovers known betas from exact
     sigmoid targets, weighted and unweighted
  7. build_summary with a constant per-head eta table equals the scalar path
     EXACTLY (bitwise), including eta = 0 heads and non-grid eta; a mixed
     table equals building every head separately with its own scalar eta
  8. eta_table_hash: same values -> same tag, any difference -> different tag,
     file/rule name irrelevant; read_eta_table rejects bad tables
  9. the `rules` stage of eta_rules_calibrate.py on synthetic stats files:
     rule 6 deployed = within-model fit (own pages, tasks pooled), recorded
     r6x = cross-model fit; the deployed table is the uniform mean of the
     per-page sigmoid predictions; all variant tables are written; a single
     model gets no r6x; and eta_rules_queue.py finds variant tables, aliases a
     variant equal to a deployed table, and queues nothing not named in --rules
"""
import glob, importlib.util, json, math, os, sys, tempfile
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
R = _load("eta_rules", "oval/eta_rules.py")
PB = _load("page_basis", "oval/page_basis.py")

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
FAILS = []
def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}  {detail}")
    if not ok: FAILS.append(name)

d, n, r = 24, 32, 8

def rand_page(dd=d, nn=n, scale_v=0.5):
    X = torch.randn(nn, dd); X -= X.mean(0, keepdim=True)
    Y = torch.randn(nn, dd) * scale_v; Y -= Y.mean(0, keepdim=True)
    W = torch.randn(3 * dd, dd) / math.sqrt(dd)
    G = W.T @ W
    lam, Q = torch.linalg.eigh(G); Gh = (Q * lam.clamp_min(0).sqrt()) @ Q.T
    return X, Y, Gh

# 1 ------------------------------------------------------------------------
D0 = torch.randn(200, 16); D1 = torch.randn(200, 16) * 0.7 + 0.3 * D0
eta_true = 0.37
blend = (1 - eta_true) * D0 + eta_true * D1          # error at eta_true is zero-ish? no: test the quadratic
a, b, c = (D0 * D0).sum(1).mean(), (D1 * D1).sum(1).mean(), (D0 * D1).sum(1).mean()
eta1, deg = R.rule1_endpoint(a, b, c)
grid = torch.linspace(0, 1, 100001)
f = torch.stack([(((1 - e) * D0 + e * D1) ** 2).sum(1).mean() for e in grid])
check("1 rule1 == argmin of blended error", abs(eta1 - grid[f.argmin()]) < 2e-5 and not deg,
      f"eta={eta1:.5f} grid={grid[f.argmin()]:.5f}")
eta1c, degc = R.rule1_endpoint(a, a, a)              # D0 == D1: flat objective
check("1 rule1 flat -> default 0.5, flagged", eta1c == 0.5 and bool(degc))
check("1 rule1 clips", R.rule1_endpoint(1.0, 10.0, 3.0)[0] == 0.0)   # vertex negative -> 0

# 2 ------------------------------------------------------------------------
alpha, beta_, gamma = 0.3, -0.5, 0.8                  # vertex at 0.3125
Lq = lambda e: alpha + beta_ * e + gamma * e * e
eta2, deg2 = R.rule2_quadratic(Lq(0), Lq(0.5), Lq(1))
check("2 rule2 recovers vertex", abs(eta2 - 0.3125) < 1e-12 and not deg2, f"{eta2:.6f}")
eta2b, deg2b = R.rule2_quadratic(1.0, 1.5, 0.8)      # concave: pick endpoint L1 < L0 -> 1
check("2 rule2 non-convex -> better endpoint", eta2b == 1.0 and bool(deg2b))
check("2 rule2 clips vertex outside", R.rule2_quadratic(0.0, 0.1, 0.3)[0] == 0.0)
# batched
eb, _ = R.rule2_quadratic(torch.tensor([Lq(0), 1.0]), torch.tensor([Lq(.5), 1.5]), torch.tensor([Lq(1), .8]))
check("2 rule2 batched", torch.allclose(eb, torch.tensor([0.3125, 1.0])))

# 3 ------------------------------------------------------------------------
X, Y, Gh = rand_page()
A, C = R.page_matrices(X, Y, Gh)
eta0 = 0.5
P0, P0p, gap, deg3 = R.projector_derivative(A, C, eta0, r)
h = 1e-5
Pp = R.proj_top_r(R.mix(A, C, eta0 + h), r); Pm = R.proj_top_r(R.mix(A, C, eta0 - h), r)
fd = (Pp - Pm) / (2 * h)
check("3 P0' == finite difference of Proj_r(M_eta)", (fd - P0p).abs().max() < 1e-5 * max(1.0, P0p.abs().max()),
      f"max|diff|={(fd - P0p).abs().max():.2e} gap={gap:.3e}")
check("3 P0 is the projector", torch.allclose(P0, R.proj_top_r(R.mix(A, C, eta0), r)) and not deg3)
# eta step == numerical minimiser of the linearised quadratic
q = torch.randn(64, d)
Jx = torch.randn(d, n) * 0.3                           # a fixed stand-in Jacobian
g = (Jx @ X @ (torch.eye(d) - P0) @ q.T).T / math.sqrt(d)
hh = (Jx @ X @ P0p @ q.T).T / math.sqrt(d)
num, den = (g * hh).sum(1).mean(), (hh * hh).sum(1).mean()
eta3, deg = R.rule3_step(num, den, eta0)
etas = torch.linspace(0, 1, 200001)
obj = torch.stack([((g - (e - eta0) * hh) ** 2).sum(1).mean() for e in etas])
check("3 eta step == argmin of linearised quadratic", abs(eta3 - etas[obj.argmin()]) < 1e-5 and not deg,
      f"eta={eta3:.5f} grid={etas[obj.argmin()]:.5f}")
etab, _ = R.rule3_step(num, den, eta0, delta=0.25)
check("3 bounded step within [eta0-d, eta0+d]", 0.25 - 1e-12 <= etab <= 0.75 + 1e-12 and
      (etab == eta3 or etab in (0.25, 0.75)))
e0, dg = R.rule3_step(0.3, 0.0, eta0, gnorm=1.0)
check("3 vanishing denominator keeps eta0, flagged", e0 == eta0 and bool(dg))
# closed eigengap: A with a degenerate r-th / (r+1)-th eigenvalue
Q, _ = torch.linalg.qr(torch.randn(d, d))
lam = torch.ones(d); lam[:r - 1] = torch.arange(r - 1, 0, -1) + 2.0
Adeg = R.unit_trace(Q @ torch.diag(lam) @ Q.T)
_, _, gapd, degd = R.projector_derivative(Adeg, Adeg, 0.5, r)
check("3 closed eigengap flagged", bool(degd), f"gap={gapd:.2e}")
# batched over pages
Ab = torch.stack([A, Adeg]); Cb = torch.stack([C, Adeg])
_, P0pb, _, degb = R.projector_derivative(Ab, Cb, eta0, r)
check("3 batched == single", torch.allclose(P0pb[0], P0p) and degb.tolist() == [False, True])

# 4 ------------------------------------------------------------------------
a4, b4 = R.rule4_sacrifices(A, C, r)
check("4 a, b >= 0 on a random page", a4 >= 0 and b4 >= 0, f"a={a4:.4f} b={b4:.4f}")
a_eq, b_eq = R.rule4_sacrifices(A, A, r)
check("4 A == C -> a = b = 0", a_eq < 1e-12 and b_eq < 1e-12)
e4, dg4 = R.rule4_eta(a_eq, b_eq)
check("4 a + b = 0 -> default 0.5, flagged", e4 == 0.5 and bool(dg4))
# C's top-r subspace orthogonal to A's: a = tr_r(A) - (A energy in C's space)
QA, _ = torch.linalg.qr(torch.randn(d, d))
lamA = torch.linspace(2, 0.1, d); lamC = lamA.flip(0)       # top-r of C == bottom-r of A
Ao = R.unit_trace(QA @ torch.diag(lamA) @ QA.T); Co = R.unit_trace(QA @ torch.diag(lamC) @ QA.T)
ao, bo = R.rule4_sacrifices(Ao, Co, r)
ao_ref = (lamA[:r].sum() - lamA[-r:].sum()) / lamA.sum(); bo_ref = (lamC[-r:].sum() - lamC[:r].sum()).abs() / lamC.sum()
check("4 orthogonal bases: a, b match closed form", abs(ao - ao_ref) < 1e-10 and abs(bo - bo_ref) < 1e-10)
# weighted aggregate over pages
ap = torch.tensor([1.0, 3.0]); bp = torch.tensor([1.0, 1.0]); w = torch.tensor([1.0, 0.0])
check("4 weighted aggregate", R.rule4_eta(ap, bp, w)[0] == 0.5 and abs(R.rule4_eta(ap, bp)[0] - 2 / 6) < 1e-12)
check("4 limiting cases", R.rule4_eta(torch.tensor([0.0]), torch.tensor([1.0]))[0] == 1.0 and
      R.rule4_eta(torch.tensor([1.0]), torch.tensor([0.0]))[0] == 0.0)

# 5 ------------------------------------------------------------------------
eta_t = 0.3127
T = R.mix(A, C, eta_t) * 3.7                                # scale must not matter
num5, den5 = R.rule5_terms(T, A, C)
e5, dg5 = R.rule5_eta(num5, den5)
check("5 rule5 recovers eta from a mixture", abs(e5 - eta_t) < 1e-10 and not dg5, f"{e5:.6f}")
# shared over pages with different A, C, same eta
pages = [rand_page() for _ in range(5)]
As = torch.stack([R.page_matrices(*p)[0] for p in pages]); Cs = torch.stack([R.page_matrices(*p)[1] for p in pages])
Ts = R.mix(As, Cs, eta_t) * torch.rand(5)[:, None, None]
n5, d5 = R.rule5_terms(Ts, As, Cs)
e5s, _ = R.rule5_eta(n5, d5)
check("5 shared eta over pages", abs(e5s - eta_t) < 1e-10, f"{e5s:.6f}")
check("5 clips", R.rule5_eta(*R.rule5_terms(R.mix(A, C, 1.4), A, C))[0] == 1.0)
check("5 C == A -> default, flagged", bool(R.rule5_eta(*R.rule5_terms(A, A, A))[1]))

# 6 ------------------------------------------------------------------------
F6 = torch.stack([R.rule6_features(*R.page_matrices(*rand_page()), r) for _ in range(400)])
F6 = (F6 - F6.mean(0)) / F6.std(0).clamp_min(1e-9) * torch.tensor([1.0, 0.6, 0.8])   # spread the features
beta_true = torch.tensor([0.2, -0.9, 0.5, 1.3])
t6 = R.rule6_predict(beta_true, F6)
bhat = R.rule6_fit(F6, t6)
check("6 sigmoid fit recovers betas (unweighted)", (bhat - beta_true).abs().max() < 1e-6, f"{bhat.tolist()}")
w6 = torch.rand(400) + 0.1
bhat_w = R.rule6_fit(F6, t6, w6)
check("6 sigmoid fit recovers betas (weighted)", (bhat_w - beta_true).abs().max() < 1e-6)
tn = (t6 + 0.05 * torch.randn(400)).clamp(0, 1)
bn = R.rule6_fit(F6, tn)
check("6 fit stable under target noise", (bn - beta_true).abs().max() < 0.3, f"{bn.tolist()}")
f_one = R.rule6_features(A, A, r)
check("6 features: A == C -> log-ratio 0, rho 1, dgamma 0",
      abs(f_one[0]) < 1e-9 and abs(f_one[1] - 1) < 1e-9 and abs(f_one[2]) < 1e-9)

# 7 ------------------------------------------------------------------------
torch.manual_seed(1)
B, P, page, nkv, hd, g, rk = 1, 5, 32, 4, 64, 3, 8
K = (torch.randn(B, P, page, nkv, hd) * 1.6).float()
V = (torch.randn(B, P, page, nkv, hd) * 0.4).float()
Wo = (torch.randn(nkv * g * hd, nkv * g * hd) / 32).float()
Gh32 = PB.publish_metric(Wo, nkv, hd, g)
def same(x, y): return all(torch.equal(a_, b_) for a_, b_ in zip(x, y))
for E in (0.0, 0.25, 0.3127, 1.0):
    ref = PB.build_summary(K, V, Gh32, E, rk)
    tab = PB.build_summary(K, V, Gh32, torch.full((nkv,), E, dtype=torch.float64), rk)
    check(f"7 constant table eta={E} == scalar (bitwise)", same(ref, tab))
# A mixed table against each head built on its own with that head's scalar
# eta. The arithmetic is identical, but a single-head build hands MKL a
# different batch of GEMMs than the 4-head build, and MKL's result then differs
# at fp32 round-off (we measured ~6e-6 on unit basis vectors, ~1e-4 on coef
# whose scale is ~10). So: compare the quantities the harness consumes --
# reconstructed keys and page scores -- to 1e-4 relative, and require the
# eta=0 / eta=1 heads (no mixing arithmetic at all) to agree bitwise.
mixed = torch.tensor([0.0, 0.25, 0.3127, 1.0], dtype=torch.float64)
mix_out = PB.build_summary(K, V, Gh32, mixed, rk)
q = torch.randn(1, 1, nkv * g, hd).float()
ok_close, ok_exact = True, True
for hh_ in range(nkv):
    Kh, Vh, Gm = K[:, :, :, hh_:hh_ + 1], V[:, :, :, hh_:hh_ + 1], Gh32[hh_:hh_ + 1]
    one = PB.build_summary(Kh, Vh, Gm, float(mixed[hh_]), rk)
    mix_h = tuple(m[:, :, hh_:hh_ + 1] for m in mix_out)
    rec_a = one[2].float() @ one[1].float().transpose(-1, -2) + one[0].float().unsqueeze(-2)
    rec_b = mix_h[2].float() @ mix_h[1].float().transpose(-1, -2) + mix_h[0].float().unsqueeze(-2)
    qh = q[:, :, hh_ * g:(hh_ + 1) * g]
    s_a = PB.oval_sel(qh, one[1], one[2], one[0], "avgSM", g, 1)
    s_b = PB.oval_sel(qh, mix_h[1], mix_h[2], mix_h[0], "avgSM", g, 1)
    ok_close &= torch.allclose(rec_a, rec_b, rtol=1e-4, atol=1e-4 * rec_a.abs().max()) and \
                torch.allclose(s_a, s_b, rtol=1e-4, atol=1e-6)
    if float(mixed[hh_]) in (0.0, 1.0):
        ok_exact &= all(torch.equal(m, o) for m, o in zip(mix_h, one))
check("7 mixed table: reconstructions and scores == per-head scalar builds (1e-4)", ok_close)
check("7 mixed table: eta=0 and eta=1 heads == per-head scalar builds (bitwise)", ok_exact)
try:
    PB.build_summary(K, V, Gh32, torch.tensor([0.1, 0.2]), rk); check("7 wrong table length refused", False)
except ValueError:
    check("7 wrong table length refused", True)
# the per-head path also goes through oval_sel unchanged (shape contract)
q = torch.randn(1, 1, nkv * g, hd).float()
s = PB.oval_sel(q, mix_out[1], mix_out[2], mix_out[0], "avgSM", nkv * g, nkv)
check("7 scores from per-head records have quest_sel's shape", tuple(s.shape) == (1, nkv, P))

# 8 ------------------------------------------------------------------------
t1 = [[0.25, 0.5], [0.31274, 1.0]]
t2 = [[0.25, 0.5], [0.31270, 1.0]]            # same after rounding to 4 decimals
t3 = [[0.25, 0.5], [0.3128, 1.0]]
check("8 hash: equal after 4-dec rounding -> same tag; 1e-4 change -> different",
      PB.eta_table_hash(t1) == PB.eta_table_hash(t2) != PB.eta_table_hash(t3))
with tempfile.TemporaryDirectory() as td:
    pa, pb_ = os.path.join(td, "ruleA.json"), os.path.join(td, "ruleB.json")
    json.dump({"rule": "A", "eta": t1}, open(pa, "w")); json.dump({"rule": "B", "eta": t2}, open(pb_, "w"))
    check("8 tag ignores file and rule name", PB.read_eta_table(pa)[1] == PB.read_eta_table(pb_)[1])
    row = PB.load_eta_table(pa, 1, 2, 2, 0)
    check("8 load_eta_table row", row.dtype == torch.float64 and row.tolist() == [0.31274, 1.0])
    for bad, why in (([[0.5, 1.2]], "out of range"), ([[0.5], [0.1, 0.2]], "ragged")):
        json.dump({"eta": bad}, open(pa, "w"))
        try: PB.read_eta_table(pa); check(f"8 rejects {why}", False)
        except ValueError: check(f"8 rejects {why}", True)
    json.dump({"eta": t1}, open(pa, "w"))
    for args, why in (((0, 3, 2, 0), "layer count"), ((0, 2, 3, 0), "head count"), ((0, 2, 2, 1), "full heads")):
        try: PB.load_eta_table(pa, *args); check(f"8 rejects wrong {why}", False)
        except ValueError: check(f"8 rejects wrong {why}", True)

# 9 ------------------------------------------------------------------------
import argparse, subprocess
import numpy as np
CAL = _load("eta_rules_calibrate", "analysis/eta_rules_calibrate.py")
COLS = ["layer", "head", "f_logba", "f_rho", "f_dgamma", "target", "weight", "mass", "a", "b", "emin_over_emax"]

def fake_cell(out, model, task, nl, H, beta_true, seed):
    """A stats json + pages npz whose per-page targets follow sigmoid(beta_true . F)."""
    rng = np.random.default_rng(seed)
    rows = []
    for li in range(nl):
        for h in range(H):
            F = rng.normal(size=(40, 3)) * [1.0, 0.3, 0.2]
            t = 1 / (1 + np.exp(-(beta_true[0] + F @ beta_true[1:])))
            w = rng.random(40) + 0.1; w /= w.sum()
            m = rng.random(40)
            rows.append(np.column_stack([np.full(40, li), np.full(40, h), F, t, w, m, rng.random(40), rng.random(40), rng.random(40)]))
    pages = np.concatenate(rows, 0)
    np.savez_compressed(os.path.join(out, f"{model}_{task}_pages.npz"), pages=pages, columns=np.array(COLS))
    e = lambda: rng.random((nl, H)).round(4).tolist()
    st = {"model": model, "task": task, "n_layers": nl, "n_kv_heads": H, "rules": {
        "r1": {"eta": e(), "degenerate": [[False] * H] * nl},
        "r2": {"eta": e(), "degenerate": [[False] * H] * nl},
        "r3": {"eta": e(), "eta_unbounded": e(), "degenerate": [[False] * H] * nl},
        "r4": {"eta": e(), "eta_mass_weighted": e(), "degenerate": [[False] * H] * nl},
        "r5": {"eta": e(), "degenerate": [[False] * H] * nl}}}
    st["rules"]["r4"]["eta_mass_weighted"] = st["rules"]["r4"]["eta"]        # r4w == r4 on purpose (alias test)
    json.dump(st, open(os.path.join(out, f"{model}_{task}.json"), "w"))
    return pages

def rules_args(out, models, tasks):
    return argparse.Namespace(models=models, tasks=tasks, out=out, tag="", rule6_ridge=1e-3, overwrite=True)

with tempfile.TemporaryDirectory() as td:
    out = os.path.join(td, "eta_rules"); os.makedirs(out)
    bA, bB = np.array([0.3, -1.0, 0.8, 1.5]), np.array([-0.4, 0.9, -0.5, -1.0])
    pg = {}
    for task, sd in (("T1", 1), ("T2", 2)):
        pg[("mA", task)] = fake_cell(out, "mA", task, 3, 2, bA, sd)
        pg[("mB", task)] = fake_cell(out, "mB", task, 4, 2, bB, 10 + sd)
    CAL.stage_rules(rules_args(out, ["mA", "mB"], ["T1", "T2"]))
    st = json.load(open(os.path.join(out, "mA_T1.json")))
    r6 = st["rules"]["r6"]
    check("9 rule 6 deployed fit is within-model, pooled over tasks",
          r6["train"] == "within-model" and r6["train_models"] == ["mA"] and r6["train_tasks"] == ["T1", "T2"]
          and r6["cross_train_models"] == ["mB"] and r6["train_rows"] == 2 * 2 * 2 * 40)       # layers 1-2, 2 heads, 2 tasks
    check("9 within-model fit recovers model A's betas; cross fit recovers model B's",
          np.abs(np.array(r6["beta"]) - bA).max() < 0.3 and np.abs(np.array(r6["beta_cross_model"]) - bB).max() < 0.3,
          f"{np.round(r6['beta'], 3).tolist()} vs {bA.tolist()}; {np.round(r6['beta_cross_model'], 3).tolist()} vs {bB.tolist()}")
    P = pg[("mA", "T1")]
    def agg(beta, li, h):
        sel = (P[:, 0] == li) & (P[:, 1] == h)
        return float(np.mean(1 / (1 + np.exp(-(beta[0] + P[sel, 2:5] @ np.array(beta[1:]))))))
    ok = all(abs(r6["eta"][li][h] - agg(r6["beta"], li, h)) < 1e-9 and
             abs(r6["eta_cross_model"][li][h] - agg(r6["beta_cross_model"], li, h)) < 1e-9
             for li in range(3) for h in range(2))
    check("9 deployed / r6x tables are uniform means of the per-page predictions under their own betas", ok)
    tabs = sorted(os.path.basename(f) for f in glob.glob(os.path.join(out, "tables", "*.json")))
    vars_ = sorted(os.path.basename(f) for f in glob.glob(os.path.join(out, "tables_variants", "*.json")))
    check("9 six deployed + four variant tables per cell",
          len(tabs) == 4 * 6 and len(vars_) == 4 * 4 and
          all(f"mA_T1_{v}.json" in vars_ for v in ("r3u", "r4w", "r6w", "r6x")), f"{len(tabs)} {len(vars_)}")
    tx = json.load(open(os.path.join(out, "tables_variants", "mA_T1_r6x.json")))
    t6 = json.load(open(os.path.join(out, "tables", "mA_T1_r6.json")))
    check("9 r6x table differs from r6 and carries variant_of", tx["hash"] != t6["hash"] and tx["variant_of"] == "r6"
          and tx["deploy"] is False and t6["deploy"] is True)
    fit = json.load(open(os.path.join(out, "rule6_fit.json")))["fits"]["mA_T1"]
    check("9 rule6_fit.json records both fits", fit["deployed"]["train"] == "within-model" and
          fit["r6x"]["train"] == "cross-model" and fit["r6x"]["train_models"] == ["mB"])
    # queue builder: variants found in tables_variants, r4w aliases r4, nothing unnamed
    codec = os.path.join(td, "codec"); os.makedirs(codec)
    qpy = str(ROOT / "analysis/eta_rules_queue.py")
    def queue(rules, extra=()):
        return subprocess.run([sys.executable, qpy, "--models", "mA", "mB", "--datasets", "T1", "--seeds", "1", "2",
                               "--rules", *rules, "--tables_dir", os.path.join(out, "tables"), "--root", codec, *extra],
                              capture_output=True, text=True, env={**os.environ, "KV_ROOT": str(ROOT)})
    NP = {"T1": 1}
    qpy_src = open(qpy).read()
    # the queue script's NPROB knows only the real datasets; patch a copy for the synthetic one
    qpy_tmp = os.path.join(td, "q.py")
    open(qpy_tmp, "w").write(qpy_src.replace('NPROB = {"MATH50": 50, "AIME24": 30, "GPQA50c": 50}', 'NPROB = {"T1": 1}')
                             .replace("from paths import ROOT", f"ROOT = __import__('pathlib').Path({str(ROOT)!r})"))
    qpy = qpy_tmp
    r = queue(["r1", "r2", "r3", "r4", "r5", "r6"], ["--count"])
    check("9 queue: 6 deployed rules x 2 models x 2 seeds = 24 seed-cells", r.stdout.strip() == "24", r.stdout + r.stderr)
    r = queue(["r1", "r2", "r3", "r4", "r5", "r6", "r3u", "r4w", "r6x"])
    check("9 queue: variants found; r4w aliases r4 (same hash); r3u and r6x add cells (8 tables x 2 x 2 = 32)",
          "r4w: same table as r4" in r.stdout and "r3u: hash" in r.stdout and "r6x: hash" in r.stdout
          and "=== 32 seed-cells to run" in r.stdout and "MISSING" not in r.stdout, r.stdout[-600:] + r.stderr)
    r = queue(["r1", "r6w"], ["--count"])
    check("9 queue: only named rules are queued (r1 + r6w = 2 x 2 x 2)", r.stdout.strip() == "8", r.stdout + r.stderr)
    # single model: within-model deployed fit, no cross-model variant
    out1 = os.path.join(td, "one"); os.makedirs(out1)
    fake_cell(out1, "mA", "T1", 3, 2, bA, 5)
    CAL.stage_rules(rules_args(out1, ["mA"], ["T1"]))
    st1 = json.load(open(os.path.join(out1, "mA_T1.json")))
    check("9 single model: deployed r6 within-model, r6x skipped",
          st1["rules"]["r6"]["train"] == "within-model" and st1["rules"]["r6"]["eta_cross_model"] is None
          and not os.path.exists(os.path.join(out1, "tables_variants", "mA_T1_r6x.json"))
          and os.path.exists(os.path.join(out1, "tables_variants", "mA_T1_r6w.json")))

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}"); sys.exit(1)
print("all checks passed")
