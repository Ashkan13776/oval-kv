"""Output error vs eta, and what rule 1 picks.

Everything is read from the calibration stats (results/eta_rules/<model>_<task>.json):
`loss_grid` is the measured relative output error L_h(eta) per (layer, KV head)
on a 21-point grid, on the calibration traces (1-4) and the held-out traces
(5-6); `rules.r1` holds the endpoint sums a, b, c and the chosen eta*.
L_h(eta) = sum_q ||Delta_h(q)||^2 / sum_q ||u_h(q)||^2, Delta_h the KV head's
residual-stream error vs dense attention (docs/eta-rules.md).

Row per model, four panels:
  (a) three example heads: measured L(eta) on the calibration split (dots), the
      rule-1 quadratic f(eta) = (1-eta)^2 a + eta^2 b + 2 eta (1-eta) c fitted
      from the two endpoints only (line, same scale as L), and eta* (star)
  (b) held-out output error, layers >= 1, relative to the per-head grid oracle:
      one curve per task for a single eta shared by every head, with rule 1's
      per-head eta* as a point at its mean eta
  (c) distribution of eta* over (layer, head), layers >= 1, all tasks
  (d) eta* vs the held-out best grid eta per head

  python analysis/plot_eta_rule1.py [--out results/eta_rules/figures/eta_rule1_error]
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["ds-r1-qwen-7b", "ds-r1-llama-8b"])
ap.add_argument("--tasks", nargs="+", default=["MATH50", "AIME24", "GPQA50c"])
ap.add_argument("--eta_dir", default=str(ROOT / "results/eta_rules"))
ap.add_argument("--example_task", default="AIME24")
ap.add_argument("--out", default=str(ROOT / "results/eta_rules/figures/eta_rule1_error"))
A = ap.parse_args()

NAME = {"ds-r1-qwen-7b": "R1-Distill-Qwen-7B", "ds-r1-llama-8b": "R1-Distill-Llama-8B"}
COL = {"MATH50": "#1f77b4", "AIME24": "#d62728", "GPQA50c": "#2ca02c"}


def load(model, task):
    st = json.load(open(os.path.join(A.eta_dir, f"{model}_{task}.json")))
    r = st["rules"]["r1"]
    return dict(grid=np.array(st["grid"]),
                Lc=np.array(st["loss_grid"]["calib"]), Lt=np.array(st["loss_grid"]["test"]),
                a=np.array(r["a"]), b=np.array(r["b"]), c=np.array(r["c"]), eta=np.array(r["eta"]),
                best_t=np.array(st["regret"]["best_test_eta"]),
                r1_t=np.array(st["regret"]["tables"]["r1"]["loss"]),
                orac_t=np.array(st["regret"]["grid_min_test"]))


fig, axes = plt.subplots(len(A.models), 4, figsize=(19, 4.4 * len(A.models)), squeeze=False)
for row, model in zip(axes, A.models):
    D = {t: load(model, t) for t in A.tasks}

    # (a) example heads: interior / near-0 / near-1 eta*, picked by their curvature
    ax = row[0]; d = D[A.example_task]; g = d["grid"]
    e = d["eta"][1:]; Lc = d["Lc"][1:]
    span = (Lc.max(-1) - Lc.min(-1)) / Lc.min(-1)                  # how much eta matters
    picks = []
    for lo, hi in [(0.3, 0.7), (-0.01, 0.01), (0.99, 1.01)]:
        idx = np.argwhere((e >= lo) & (e <= hi))
        if len(idx):
            picks.append(tuple(idx[np.argmax(span[tuple(idx.T)])]))
    xs = np.linspace(0, 1, 201)
    for (l, h), cc in zip(picks, ["#9467bd", "#ff7f0e", "#17becf"]):
        L = Lc[l, h]; ref = L.min()
        a, b, c = d["a"][1:][l, h], d["b"][1:][l, h], d["c"][1:][l, h]
        fx = ((1 - xs) ** 2 * a + xs ** 2 * b + 2 * xs * (1 - xs) * c) * (L[0] / a)
        ax.plot(g, L / ref, "o", color=cc, ms=4)
        ax.plot(xs, fx / ref, "-", color=cc, lw=1.4, label=f"layer {l + 1}, head {h}: eta*={e[l, h]:.2f}")
        es = e[l, h]
        ax.plot([es], [((1 - es) ** 2 * a + es ** 2 * b + 2 * es * (1 - es) * c) * (L[0] / a) / ref],
                "*", color=cc, ms=15, mec="k", mew=0.6)
    ax.set_title(f"{NAME[model]} / {A.example_task}: example heads\n"
                 "dots: measured L(eta); line: rule-1 quadratic", fontsize=10)
    ax.set_xlabel("eta"); ax.set_ylabel("output error / min over eta")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) held-out error: one shared eta vs rule 1 per head, relative to the per-head oracle
    ax = row[1]
    for t, d in D.items():
        orac = d["orac_t"][1:].mean()
        shared = d["Lt"][1:].mean((0, 1))
        ax.plot(d["grid"], 100 * (shared / orac - 1), "-", color=COL[t], lw=1.6, label=f"{t}: one eta for all heads")
        ax.plot([d["eta"][1:].mean()], [100 * (d["r1_t"][1:].mean() / orac - 1)], "*", color=COL[t],
                ms=16, mec="k", mew=0.6, label=f"{t}: rule 1 per head (x = mean eta*)")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_title(f"{NAME[model]}: held-out output error, layers >= 1\n"
                 "vs. the best grid eta per head (oracle)", fontsize=10)
    ax.set_xlabel("eta"); ax.set_ylabel("excess output error (%)")
    ax.legend(fontsize=7.5, ncol=1); ax.grid(alpha=0.3)

    # (c) eta* distribution
    ax = row[2]
    bins = np.linspace(0, 1, 21)
    ax.hist([D[t]["eta"][1:].ravel() for t in A.tasks], bins=bins, color=[COL[t] for t in A.tasks],
            label=A.tasks, stacked=False)
    ax.set_title(f"{NAME[model]}: rule-1 eta*\nper (layer, KV head)", fontsize=10)
    ax.set_xlabel("eta*"); ax.set_ylabel("heads"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (d) eta* vs held-out best grid eta
    ax = row[3]
    rng = np.random.default_rng(0)
    for t, d in D.items():
        x = d["eta"][1:].ravel(); y = d["best_t"][1:].ravel()
        ax.plot(x + rng.normal(0, 0.008, x.shape), y + rng.normal(0, 0.008, y.shape), ".",
                color=COL[t], ms=4, alpha=0.6, label=f"{t}  (corr {np.corrcoef(x, y)[0, 1]:.2f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_title(f"{NAME[model]}: rule-1 eta*\nvs. held-out best grid eta", fontsize=10)
    ax.set_xlabel("rule-1 eta* (calibration traces)"); ax.set_ylabel("best grid eta (held-out traces)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.tight_layout()
os.makedirs(os.path.dirname(A.out), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{A.out}.{ext}", dpi=150)
print(f"wrote {A.out}.png/.pdf")
