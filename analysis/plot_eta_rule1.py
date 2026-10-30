"""Output error vs eta, and what rule 1 picks.

Everything is read from the calibration stats (results/eta_rules/<model>_<task>.json):
`loss_grid` is the measured relative output error L_h(eta) per (layer, KV head)
on a 21-point grid, on the calibration traces (1-4) and the held-out traces
(5-6); `rules.r1` holds the endpoint sums a, b, c and the chosen eta*.
L_h(eta) = sum_q ||Delta_h(q)||^2 / sum_q ||u_h(q)||^2, Delta_h the KV head's
residual-stream error vs dense attention (docs/eta-rules.md).

One row per model, one panel per layer (spread over the depth), the first
--heads KV heads of each: measured L(eta) on the calibration split as % above
the head's minimum over eta (dots), rule 1's quadratic
f(eta) = (1-eta)^2 a + eta^2 b + 2 eta (1-eta) c built from the two endpoints
only (line, same scale), and its eta* (star). No titles; the caption is in
docs/eta-rules.md.

  python analysis/plot_eta_rule1.py [--out results/eta_rules/figures/eta_rule1_error]
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.lines, matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["ds-r1-qwen-7b", "ds-r1-llama-8b"])
ap.add_argument("--eta_dir", default=str(ROOT / "results/eta_rules"))
ap.add_argument("--task", default="AIME24")
ap.add_argument("--layer_fracs", type=float, nargs="+", default=[0.05, 0.3, 0.55, 0.8, 1.0],
                help="layers shown, as fractions of the depth")
ap.add_argument("--heads", type=int, default=4, help="first N KV heads of each layer")
ap.add_argument("--out", default=str(ROOT / "results/eta_rules/figures/eta_rule1_error"))
A = ap.parse_args()

NAME = {"ds-r1-qwen-7b": "R1-Distill-Qwen-7B", "ds-r1-llama-8b": "R1-Distill-Llama-8B"}
HEADC = ["#6FA3D2", "#E06C75", "#7FBF8E", "#A58BCB"]       # pastel, deep enough for lines
plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "legend.fontsize": 11.5, "axes.spines.top": False, "axes.spines.right": False})


def load(model, task):
    st = json.load(open(os.path.join(A.eta_dir, f"{model}_{task}.json")))
    r = st["rules"]["r1"]
    return dict(grid=np.array(st["grid"]), Lc=np.array(st["loss_grid"]["calib"]),
                a=np.array(r["a"]), b=np.array(r["b"]), c=np.array(r["c"]), eta=np.array(r["eta"]))


ncol = len(A.layer_fracs)
fig, axes = plt.subplots(len(A.models), ncol, figsize=(4.3 * ncol, 4.6 * len(A.models)), squeeze=False)
xs = np.linspace(0, 1, 201)
for r, (row, model) in enumerate(zip(axes, A.models)):
    d = load(model, A.task); g = d["grid"]
    nl, nh = d["eta"].shape
    # layers spread over the depth; layer 0 is dense in the harness and has no records
    layers = sorted({min(nl - 1, max(1, int(round(f * (nl - 1))))) for f in A.layer_fracs})
    for ci, (ax, li) in enumerate(zip(row, layers)):
        for k, h in enumerate(range(min(A.heads, nh))):
            L = d["Lc"][li, h]; ref = L.min()
            a, b, c = d["a"][li, h], d["b"][li, h], d["c"][li, h]
            # rule 1's quadratic in a, b, c, scaled to L at eta = 0 (both are sums
            # of squared errors; L is normalised by the dense output energy)
            # plotted as % above the head's best measured error over the grid
            f = lambda t: 100 * (((1 - t) ** 2 * a + t ** 2 * b + 2 * t * (1 - t) * c) * (L[0] / a) / ref - 1)
            e = d["eta"][li, h]
            ax.plot(g, 100 * (L / ref - 1), "o", color=HEADC[k], ms=5, alpha=0.85)
            ax.plot(xs, f(xs), "-", color=HEADC[k], lw=2.2)
            ax.plot([e], [f(e)], "*", color=HEADC[k], ms=17, mec="#333", mew=0.7, zorder=5)
        ax.text(0.5, 1.02, f"layer {li}", transform=ax.transAxes, ha="center", va="bottom", fontsize=14)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, integer=True))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.axhline(0, color="#333", lw=0.8, ls=":")
        ax.grid(alpha=0.3)
        if r == len(A.models) - 1:
            ax.set_xlabel("$\\eta$")
        if ci == 0:
            ax.set_ylabel(f"{NAME[model]}\nerror above head's best")
    for ax in row[len(layers):]:
        ax.axis("off")
# shared key: head colours, then the marker types
fig.legend(handles=[matplotlib.lines.Line2D([], [], ls="-", lw=3, color=HEADC[k], label=f"KV head {k}")
                    for k in range(A.heads)] + [matplotlib.lines.Line2D([], [], marker="o", ls="", color="#888", ms=6, label="measured"),
                    matplotlib.lines.Line2D([], [], ls="-", color="#888", lw=2.2, label="rule-1 quadratic (from $\\eta$=0, 1)"),
                    matplotlib.lines.Line2D([], [], marker="*", ls="", color="#bbb", mec="#333", ms=15, label="rule-1 $\\eta^*$")],
           loc="lower center", ncol=A.heads + 3, frameon=False, fontsize=14, bbox_to_anchor=(0.5, -0.05))
fig.tight_layout(h_pad=2.0, w_pad=1.5)
os.makedirs(os.path.dirname(A.out), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{A.out}.{ext}", dpi=200, bbox_inches="tight")
print(f"wrote {A.out}.png/.pdf")
