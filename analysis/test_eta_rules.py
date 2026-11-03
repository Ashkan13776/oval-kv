"""CPU unit tests for oval/eta_rules.py (rule 1) and the per-head eta path of
oval/page_basis.py::build_summary. Run: python analysis/test_eta_rules.py

  1. rule 1 equals the brute-force argmin of the blended endpoint error; a
     flat objective falls back to 0.5 and is flagged; the vertex is clipped
  2. build_summary with a constant per-head eta table equals the scalar path
     EXACTLY (bitwise), including eta = 0 heads and non-grid eta; a mixed
     table equals building every head separately with its own scalar eta
  3. eta_table_hash: same values -> same tag, any difference -> different tag,
     file/rule name irrelevant; read_eta_table rejects bad tables
  4. the `rules` stage of eta_rules_calibrate.py writes rule 1's table from a
     stats file, unchanged tables are not rewritten, and eta_rules_queue.py
     queues one cell per (model, dataset, seed)
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

# 1 ------------------------------------------------------------------------
D0 = torch.randn(200, 16); D1 = torch.randn(200, 16) * 0.7 + 0.3 * D0
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
    check(f"2 constant table eta={E} == scalar (bitwise)", same(ref, tab))
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
check("2 mixed table: reconstructions and scores == per-head scalar builds (1e-4)", ok_close)
check("2 mixed table: eta=0 and eta=1 heads == per-head scalar builds (bitwise)", ok_exact)
try:
    PB.build_summary(K, V, Gh32, torch.tensor([0.1, 0.2]), rk); check("2 wrong table length refused", False)
except ValueError:
    check("2 wrong table length refused", True)
# the per-head path also goes through oval_sel unchanged (shape contract)
q = torch.randn(1, 1, nkv * g, hd).float()
s = PB.oval_sel(q, mix_out[1], mix_out[2], mix_out[0], "avgSM", nkv * g, nkv)
check("2 scores from per-head records have quest_sel's shape", tuple(s.shape) == (1, nkv, P))

# 3 ------------------------------------------------------------------------
t1 = [[0.25, 0.5], [0.31274, 1.0]]
t2 = [[0.25, 0.5], [0.31270, 1.0]]            # same after rounding to 4 decimals
t3 = [[0.25, 0.5], [0.3128, 1.0]]
check("3 hash: equal after 4-dec rounding -> same tag; 1e-4 change -> different",
      PB.eta_table_hash(t1) == PB.eta_table_hash(t2) != PB.eta_table_hash(t3))
with tempfile.TemporaryDirectory() as td:
    pa, pb_ = os.path.join(td, "ruleA.json"), os.path.join(td, "ruleB.json")
    json.dump({"rule": "A", "eta": t1}, open(pa, "w")); json.dump({"rule": "B", "eta": t2}, open(pb_, "w"))
    check("3 tag ignores file and rule name", PB.read_eta_table(pa)[1] == PB.read_eta_table(pb_)[1])
    row = PB.load_eta_table(pa, 1, 2, 2, 0)
    check("3 load_eta_table row", row.dtype == torch.float64 and row.tolist() == [0.31274, 1.0])
    for bad, why in (([[0.5, 1.2]], "out of range"), ([[0.5], [0.1, 0.2]], "ragged")):
        json.dump({"eta": bad}, open(pa, "w"))
        try: PB.read_eta_table(pa); check(f"3 rejects {why}", False)
        except ValueError: check(f"3 rejects {why}", True)
    json.dump({"eta": t1}, open(pa, "w"))
    for args, why in (((0, 3, 2, 0), "layer count"), ((0, 2, 3, 0), "head count"), ((0, 2, 2, 1), "full heads")):
        try: PB.load_eta_table(pa, *args); check(f"3 rejects wrong {why}", False)
        except ValueError: check(f"3 rejects wrong {why}", True)

# 4 ------------------------------------------------------------------------
import argparse, subprocess
CAL = _load("eta_rules_calibrate", "analysis/eta_rules_calibrate.py")
with tempfile.TemporaryDirectory() as td:
    out = os.path.join(td, "eta_rules"); os.makedirs(out)
    gen = torch.Generator().manual_seed(4)
    for m, nl in (("mA", 3), ("mB", 4)):
        eta = torch.rand(nl, 2, generator=gen).tolist()
        json.dump({"model": m, "task": "T1", "n_layers": nl, "n_kv_heads": 2,
                   "rules": {"r1": {"eta": eta, "degenerate": [[False, False]] * nl}}},
                  open(os.path.join(out, f"{m}_T1.json"), "w"))
    ns = argparse.Namespace(models=["mA", "mB"], tasks=["T1"], out=out, tag="", overwrite=False)
    CAL.stage_rules(ns)
    tabs = sorted(os.path.basename(f) for f in glob.glob(os.path.join(out, "tables", "*.json")))
    check("4 rules stage writes one rule-1 table per (model, task)", tabs == ["mA_T1_r1.json", "mB_T1_r1.json"], str(tabs))
    t = json.load(open(os.path.join(out, "tables", "mA_T1_r1.json")))
    st = json.load(open(os.path.join(out, "mA_T1.json")))
    check("4 table = stats eta rounded to 4 decimals, hash matches",
          t["eta"] == [[round(v, 4) for v in row] for row in st["rules"]["r1"]["eta"]]
          and t["hash"] == PB.eta_table_hash(t["eta"]) and t["rule"] == "r1" and t["deploy"] is True)
    mt = os.path.getmtime(os.path.join(out, "tables", "mA_T1_r1.json"))
    CAL.stage_rules(ns)                                    # same table, no --overwrite: no error, not rewritten
    check("4 re-running with an unchanged table is a no-op", os.path.getmtime(os.path.join(out, "tables", "mA_T1_r1.json")) == mt)
    # queue builder on the synthetic dataset (patch a copy: NPROB knows only the real datasets)
    codec = os.path.join(td, "codec"); os.makedirs(codec)
    qpy = os.path.join(td, "q.py")
    open(qpy, "w").write(open(ROOT / "analysis/eta_rules_queue.py").read()
                         .replace('NPROB = {"MATH50": 50, "AIME24": 30, "GPQA50c": 50}', 'NPROB = {"T1": 1}')
                         .replace("from paths import ROOT", f"ROOT = __import__('pathlib').Path({str(ROOT)!r})"))
    r = subprocess.run([sys.executable, qpy, "--models", "mA", "mB", "--datasets", "T1", "--seeds", "1", "2",
                        "--tables_dir", os.path.join(out, "tables"), "--root", codec, "--count"],
                       capture_output=True, text=True)
    check("4 queue: 2 models x 2 seeds = 4 seed-cells", r.stdout.strip() == "4", r.stdout + r.stderr)

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}"); sys.exit(1)
print("all checks passed")
