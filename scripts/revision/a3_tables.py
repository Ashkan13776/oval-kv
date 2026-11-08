"""Block A3: rule variants built from the endpoint statistics behind the Tab. 1 tables.

For every (model, task) with a value-rank-16 stats file in results/eta_rules/vr16/,
read the per-(layer, KV head) endpoint terms a = E||D0||^2, b = E||D1||^2,
c = E[D0.D1] (calibration traces 1-4) and write three new tables:

  argmin01  eta_h = 0 if a_h <= b_h else 1                       (per head)
  layer     rule 1 on a, b, c summed over the KV heads of a layer (per layer)
  shared    one eta for all heads: argmin over the 21-point grid of
            sum_h f_h(eta), f_h(eta) = (1-eta)^2 a + eta^2 b + 2 eta (1-eta) c

The deployed rule-1 table is re-derived from the same a, b, c and its hash is
checked against results/eta_rules/vr16/tables/<model>_<task>_r1.json, so the
variants are built from exactly the statistics behind Tab. 1.

Outputs never overwrite: results/revision/A/tables/<model>_<task>_r1_<variant>.json

  python scripts/revision/a3_tables.py
"""
import json, os, sys, importlib.util, datetime
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
PB = _load("page_basis", "oval/page_basis.py")
R = _load("eta_rules", "oval/eta_rules.py")

STATS = ROOT / "results/eta_rules/vr16"
OUT = ROOT / "results/revision/A/tables"
GRID = np.round(np.linspace(0, 1, 21), 4)


def write(model, task, variant, eta, info):
    OUT.mkdir(parents=True, exist_ok=True)
    eta = [[round(float(v), 4) for v in row] for row in eta]
    h = PB.eta_table_hash(eta)
    path = OUT / f"{model}_{task}_r1_{variant}.json"
    if path.exists():
        old = json.load(open(path))
        if old.get("hash") != h:
            sys.exit(f"{path} exists with a different table; refusing to overwrite")
        return path, h
    json.dump({"model": model, "task": task, "rule": f"r1_{variant}", "deploy": True, "hash": h,
               "n_layers": len(eta), "n_kv_heads": len(eta[0]), "eta": eta,
               "created": datetime.datetime.now().isoformat(timespec="seconds"),
               "source": str(info.pop("_src", "")), **info},
              open(path, "w"), indent=1)
    return path, h


def main():
    rows = []
    files = sorted(STATS.glob("*.json")) + sorted((ROOT / "results/revision/A/calib_vr16").glob("*.json"))
    for f in files:
        if f.stem.endswith("_disjoint"):
            continue
        st = json.load(open(f))
        model, task = st["model"], st["task"]
        r1 = st["rules"]["r1"]
        a, b, c = (np.asarray(r1[k], dtype=np.float64) for k in ("a", "b", "c"))
        # 0) re-derive the deployed table and check its hash against Tab. 1's table
        eta_r1, _ = R.rule1_endpoint(torch.as_tensor(a), torch.as_tensor(b), torch.as_tensor(c))
        eta_r1 = [[round(float(v), 4) for v in row] for row in eta_r1.numpy()]
        dep = json.load(open(f.parent / "tables" / f"{model}_{task}_r1.json"))
        ok = PB.eta_table_hash(eta_r1) == dep["hash"]
        # layer 0 is dense in the harness; keep whatever the deployed table holds there
        # 1) per-head argmin over the two endpoints
        argmin = np.where(a <= b, 0.0, 1.0)
        # 2) per-layer rule on summed terms
        el, _ = R.rule1_endpoint(torch.as_tensor(a.sum(1)), torch.as_tensor(b.sum(1)), torch.as_tensor(c.sum(1)))
        layer = np.repeat(el.numpy()[:, None], a.shape[1], axis=1)
        # 3) one shared eta per (model, task): minimise the summed quadratic on the grid
        g = GRID[:, None, None]
        f_sum = ((1 - g) ** 2 * a + g ** 2 * b + 2 * g * (1 - g) * c)[:, 1:, :].sum((1, 2))  # layers >= 1
        e_shared = float(GRID[int(np.argmin(f_sum))])
        shared = np.full_like(a, e_shared)
        out = {}
        for name, tab, info in (("argmin01", argmin, {"note": "eta=0 if a<=b else 1"}),
                                ("layer", layer, {"note": "rule 1 on per-layer sums of a,b,c"}),
                                ("shared", shared, {"note": f"argmin_grid sum_h f_h, eta={e_shared}"})):
            p, h = write(model, task, name, tab, {**info, "_src": str(f.relative_to(ROOT))})
            out[name] = h
        rows.append((model, task, ok, dep["hash"], e_shared, out,
                     float(np.mean(argmin[1:])), float(np.mean(layer[1:]))))
    print("| model | task | r1 hash reproduced | r1 hash | shared eta | argmin01: frac eta=1 | layer: mean eta | new hashes |")
    print("|---|---|---|---|---|---|---|---|")
    for m, t, ok, h, e, out, fa, fl in rows:
        print(f"| {m} | {t} | {ok} | {h} | {e} | {fa:.2f} | {fl:.3f} | {out} |")


if __name__ == "__main__":
    main()
