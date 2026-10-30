"""Output error vs eta, and what rule 1 picks.

Everything is read from the calibration stats (results/eta_rules/<model>_<task>.json):
`loss_grid` is the measured relative output error L_h(eta) per (layer, KV head)
on a 21-point grid, on the calibration traces (1-4) and the held-out traces
(5-6); `rules.r1` holds the endpoint sums a, b, c and the chosen eta*.
L_h(eta) = sum_q ||Delta_h(q)||^2 / sum_q ||u_h(q)||^2, Delta_h the KV head's
residual-stream error vs dense attention (docs/eta-rules.md).

Row per model, two panels (no titles; the caption is in docs/eta-rules.md):
  left   three example heads: measured L(eta) on the calibration split (dots),
         the rule-1 quadratic f(eta) = (1-eta)^2 a + eta^2 b + 2 eta (1-eta) c
         fitted from the two endpoints only (line, same scale as L), and eta* (star)
  right  held-out output error, layers >= 1, relative to the per-head grid
         oracle: one curve per task for a single eta shared by every head, with
         rule 1's per-head eta* as a star at its mean eta

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
# pastel palette, in deeper shades for lines and markers so they stay readable
COL = {"MATH50": "#6FA3D2", "AIME24": "#E06C75", "GPQA50c": "#7FBF8E"}
HEAD = ["#A58BCB", "#F5A25D", "#5FB7B2"]
plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "legend.fontsize": 12, "axes.spines.top": False, "axes.spines.right": False})


def panel_letter(ax, s):
    ax.text(-0.11, 1.02, f"({s})", transform=ax.transAxes, fontsize=17, fontweight="bold", va="bottom")


def load(model, task):
    st = json.load(open(os.path.join(A.eta_dir, f"{model}_{task}.json")))
    r = st["rules"]["r1"]
    return dict(grid=np.array(st["grid"]),
                Lc=np.array(st["loss_grid"]["calib"]), Lt=np.array(st["loss_grid"]["test"]),
                a=np.array(r["a"]), b=np.array(r["b"]), c=np.array(r["c"]), eta=np.array(r["eta"]),
                r1_t=np.array(st["regret"]["tables"]["r1"]["loss"]),
                orac_t=np.array(st["regret"]["grid_min_test"]))


fig, axes = plt.subplots(len(A.models), 2, figsize=(14, 5.6 * len(A.models)), squeeze=False)
letters = iter("abcdefgh")
for row, model in zip(axes, A.models):
    D = {t: load(model, t) for t in A.tasks}

    # left: example heads -- interior / near-0 / near-1 eta*, picked by how much eta matters
    ax = row[0]; d = D[A.example_task]; g = d["grid"]
    e = d["eta"][1:]; Lc = d["Lc"][1:]
    span = (Lc.max(-1) - Lc.min(-1)) / Lc.min(-1)
    picks = []
    for lo, hi in [(0.3, 0.7), (-0.01, 0.01), (0.99, 1.01)]:
        idx = np.argwhere((e >= lo) & (e <= hi))
        if len(idx):
            picks.append(tuple(idx[np.argmax(span[tuple(idx.T)])]))
    xs = np.linspace(0, 1, 201)
    for (l, h), cc in zip(picks, HEAD):
        L = Lc[l, h]; ref = L.min()
        a, b, c = d["a"][1:][l, h], d["b"][1:][l, h], d["c"][1:][l, h]
        f = lambda t: ((1 - t) ** 2 * a + t ** 2 * b + 2 * t * (1 - t) * c) * (L[0] / a) / ref
        ax.plot(g, L / ref, "o", color=cc, ms=6, alpha=0.9)
        ax.plot(xs, f(xs), "-", color=cc, lw=2.4,
                label=f"layer {l + 1}, head {h}: $\\eta^*$ = {e[l, h]:.2f}")
        ax.plot([e[l, h]], [f(e[l, h])], "*", color=cc, ms=20, mec="#333", mew=0.8, zorder=5)
    ax.plot([], [], "o", color="#999", ms=6, label="measured")
    ax.plot([], [], "-", color="#999", lw=2.4, label="rule-1 quadratic")
    ax.set_xlabel("$\\eta$"); ax.set_ylabel("output error / min over $\\eta$")
    ax.legend(frameon=False, fontsize=11.5, loc="upper center")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
    ax.text(0.97, 0.6, f"{NAME[model]}\n{A.example_task}", transform=ax.transAxes, ha="right", va="center",
            fontsize=13, color="#444")
    ax.grid(alpha=0.3); panel_letter(ax, next(letters))

    # right: held-out error, one shared eta vs rule 1 per head, relative to the per-head oracle
    ax = row[1]
    for t, d in D.items():
        orac = d["orac_t"][1:].mean()
        shared = d["Lt"][1:].mean((0, 1))
        ax.plot(d["grid"], 100 * (shared / orac - 1), "-", color=COL[t], lw=2.6, label=t)
        ax.plot([d["eta"][1:].mean()], [100 * (d["r1_t"][1:].mean() / orac - 1)], "*", color=COL[t],
                ms=22, mec="#333", mew=0.8, zorder=5)
    ax.plot([], [], "*", color="#bbb", ms=16, mec="#333", mew=0.8, label="rule 1, per head")
    ax.axhline(0, color="#333", lw=0.8, ls=":")
    ax.set_xlabel("shared $\\eta$  (rule 1: mean $\\eta^*$)"); ax.set_ylabel("excess output error (%)")
    ax.legend(frameon=False, loc="upper center")
    ax.text(0.97, 0.05 if model.endswith("qwen-7b") else 0.9, NAME[model], transform=ax.transAxes,
            ha="right", fontsize=13, color="#444")
    ax.grid(alpha=0.3); panel_letter(ax, next(letters))

fig.tight_layout(h_pad=2.5, w_pad=3)
os.makedirs(os.path.dirname(A.out), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{A.out}.{ext}", dpi=200, bbox_inches="tight")
print(f"wrote {A.out}.png/.pdf")
