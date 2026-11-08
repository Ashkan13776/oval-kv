"""Block A5 figures for the supplementary (no GPU). Each figure: vector PDF + the CSV behind it.

All figures are in the OVAL setting (rank-8 keys, rank-16 values) once
results/revision/A/grid_vr16/ exists; until then F2/F3 fall back to value rank 8
and say so in the axis title.

F1  eta_h per (layer, KV head) for every model x task: one figure, rows = models,
    columns = tasks, single-hue sequential ramp, one shared colorbar.
F2  six heads per model (MATH500), picked by quantile of b_h - a_h: held-out output
    error on the 21-point eta grid (dots), the endpoint quadratic f_h from the
    calibration terms (line), the rule's eta_h (star). Rows = models.
F3  ECDF over (layer >= 1, KV head), three tasks pooled, of the held-out error at
    eta_h above the per-head grid optimum (%), one line per model, log x.

  python analysis/fig_eta_revision.py
"""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/revision/A/figures"; OUT.mkdir(parents=True, exist_ok=True)
TASKS = ["MATH50", "AIME24", "GPQA50c"]
MODELS = ["ds-r1-qwen-7b", "ds-r1-llama-8b", "ds-r1-qwen-14b"]
NICE = {"ds-r1-qwen-7b": "R1-Distill Qwen-7B", "ds-r1-llama-8b": "R1-Distill Llama-8B",
        "ds-r1-qwen-14b": "R1-Distill Qwen-14B", "MATH50": "MATH500", "AIME24": "AIME24", "GPQA50c": "GPQA"}

# --- style: paper font (Times clone), recessive axes, validated reference palette ---
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"                 # categorical slots 1-2 (validated all-pairs)
SEQ = LinearSegmentedColormap.from_list("blue_seq", [  # sequential blue ramp, steps 100 -> 700
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
WIDTH = 6.875                                         # CVPR text width (in)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "figure.dpi": 150})


def stats_path(m, t):
    """Prefer the OVAL-setting grid (rank-8 keys, rank-16 values) when it exists."""
    g = ROOT / f"results/revision/A/grid_vr16/{m}_{t}.json"
    if g.exists():
        st = json.load(open(g))
        if len(st["grid"]) == 21 and "regret" in st:
            return g, 16
    return ROOT / f"results/eta_rules/{m}_{t}.json", 8


def deployed_tables():
    found = {}
    for d in [ROOT / "results/eta_rules/vr16/tables", ROOT / "results/revision/A/calib_vr16/tables"]:
        if d.exists():
            for f in sorted(d.glob("*_r1.json")):
                t = json.load(open(f)); found[(t["model"], t["task"])] = (np.array(t["eta"]), t["hash"])
    return found


def f1():
    found = deployed_tables()
    models = [m for m in MODELS if all((m, t) in found for t in TASKS)]
    heights = [found[(m, TASKS[0])][0].shape[1] for m in models]
    fig = plt.figure(figsize=(WIDTH, 0.35 + 0.2 * sum(heights) + 0.55 * len(models)))
    gs = fig.add_gridspec(len(models), len(TASKS) + 1, height_ratios=heights,
                          width_ratios=[1, 1, 1, 0.03], hspace=0.55, wspace=0.08)
    rows = []
    for r, m in enumerate(models):
        for c, t in enumerate(TASKS):
            eta, h = found[(m, t)]
            ax = fig.add_subplot(gs[r, c])
            L, H = eta.shape
            im = ax.imshow(eta[1:].T, aspect="auto", origin="lower", cmap=SEQ, vmin=0, vmax=1,
                           interpolation="nearest", extent=[0.5, L - 0.5, -0.5, H - 0.5])
            v = eta[1:].ravel()
            stat = f"median {np.median(v):.2f} · {(v == 0).mean()*100:.0f}% at 0 · {(v == 1).mean()*100:.0f}% at 1"
            ax.set_title(stat, fontsize=6.5, color=INK2, pad=2)
            if r == 0:
                ax.text(0.5, 1.0, NICE[t], transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=8, color=INK, fontweight="bold").set_y(1.16 if H <= 4 else 1.08)
            ax.set_yticks(range(H)) if H <= 4 else ax.set_yticks(range(0, H, 2))
            ax.tick_params(length=0)
            for s in ax.spines.values(): s.set_visible(False)
            if c == 0: ax.set_ylabel(f"{NICE[m].replace('R1-Distill ', '')}\nKV head")
            else: ax.set_yticklabels([])
            if r == len(models) - 1: ax.set_xlabel("layer (layer 0 is dense)", labelpad=1)
            else: ax.set_xticklabels([])
            for li in range(L):
                for hi in range(H):
                    rows.append([m, t, h, li, hi, eta[li, hi]])
    cax = fig.add_subplot(gs[:, -1])
    cb = fig.colorbar(im, cax=cax); cb.set_label("$\\eta_h$"); cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    fig.savefig(OUT / "F1_eta_heatmaps.pdf"); plt.close(fig)
    with open(OUT / "F1_eta_heatmaps.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["model", "task", "table_hash", "layer", "kv_head", "eta"]); w.writerows(rows)
    return models


def f2():
    rows, vrs = [], set()
    ms = ["ds-r1-qwen-7b", "ds-r1-llama-8b"]
    fig, axs = plt.subplots(len(ms), 6, figsize=(WIDTH, 1.55 * len(ms) + 0.35), sharex=True)
    for r, m in enumerate(ms):
        sp, vr = stats_path(m, "MATH50"); vrs.add(vr); st = json.load(open(sp))
        grid = np.array(st["grid"]); test = np.array(st["loss_grid"]["test"])
        r1 = st["rules"]["r1"]; a, b, c = (np.array(r1[k]) for k in "abc"); eta = np.array(r1["eta"])
        L, H = a.shape
        idx = [(li, hi) for li in range(1, L) for hi in range(H)]
        diff = np.array([b[i] - a[i] for i in idx]); order = np.argsort(diff); n = len(order)
        pick = [order[1], order[3], order[n // 2 - 1], order[n // 2 + 1], order[-4], order[-2]]
        for col, (ax, k) in enumerate(zip(axs[r], pick)):
            li, hi = idx[k]
            y = test[li, hi]; ymin = y.min(); yp = 100 * (y - ymin) / ymin
            fg = (1 - grid) ** 2 * a[li, hi] + grid ** 2 * b[li, hi] + 2 * grid * (1 - grid) * c[li, hi]
            fp = 100 * (fg - fg.min()) / fg.min()
            e = eta[li, hi]; ye = 100 * (np.interp(e, grid, y) - ymin) / ymin
            ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
            ax.plot(grid, fp, "-", color=BLUE, lw=1.6, label="endpoint quadratic $f_h$", zorder=2)
            ax.plot(grid, yp, "o", ms=3.2, color=INK, mec="white", mew=0.4, label="held-out error", zorder=3)
            ax.plot([e], [ye], marker="*", ms=10, color=ORANGE, mec="white", mew=0.6, ls="none",
                    label="rule's $\\eta_h$", zorder=4)
            ax.set_title(f"layer {li}, head {hi}", fontsize=7, color=INK2, pad=2)
            ax.set_xticks([0, 0.5, 1]); ax.set_ylim(bottom=-2)
            if col == 0: ax.set_ylabel(f"{NICE[m].replace('R1-Distill ', '')}\n% above minimum")
            if r == len(ms) - 1: ax.set_xlabel("$\\eta$")
            for gi, gv in enumerate(grid):
                rows.append([m, vr, li, hi, float(diff[k]), gv, y[gi], yp[gi], fg[gi], fp[gi], e])
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    fig.savefig(OUT / "F2_eta_fit.pdf"); plt.close(fig)
    with open(OUT / "F2_eta_fit.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "value_rank", "layer", "kv_head", "b_minus_a", "eta", "test_error",
                    "test_pct_above_min", "f_h", "f_h_pct_above_min", "eta_h"]); w.writerows(rows)
    return vrs


def f3():
    rows, vrs, summ = [], set(), {}
    fig, ax = plt.subplots(figsize=(WIDTH * 0.5, 2.1))
    floor = 1e-3
    for m, colr in (("ds-r1-qwen-7b", BLUE), ("ds-r1-llama-8b", ORANGE)):
        vals = []
        for t in TASKS:
            sp, vr = stats_path(m, t); vrs.add(vr); st = json.load(open(sp))
            loss = np.array(st["regret"]["tables"]["r1"]["loss"]); gmin = np.array(st["regret"]["grid_min_test"])
            rel = 100 * (loss - gmin) / gmin
            for li in range(1, rel.shape[0]):
                for hi in range(rel.shape[1]):
                    vals.append(rel[li, hi]); rows.append([m, vr, t, li, hi, loss[li, hi], gmin[li, hi], rel[li, hi]])
        v = np.sort(np.maximum(np.array(vals), floor)); ycdf = np.arange(1, len(v) + 1) / len(v)
        med, p90 = float(np.median(vals)), float(np.percentile(vals, 90)); summ[m] = (med, p90, len(vals))
        ax.step(v, ycdf, where="post", color=colr, lw=1.6,
                label=f"{NICE[m].replace('R1-Distill ', '')}: median {med:.2f}%, 90th pct. {p90:.1f}%")
    ax.set_xscale("log"); ax.set_xlim(floor, None); ax.set_ylim(0, 1.0)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_xlabel("held-out output error of $\\eta_h$ above the per-head optimum (%)")
    ax.set_ylabel("fraction of KV heads")
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(OUT / "F3_eta_regret_ecdf.pdf"); plt.close(fig)
    with open(OUT / "F3_eta_regret_ecdf.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "value_rank", "task", "layer", "kv_head", "loss_eta_h", "grid_min", "pct_above"])
        w.writerows(rows)
    return vrs, summ


if __name__ == "__main__":
    print("F1 models:", f1()); print("F2 value rank:", f2()); print("F3:", f3()); print("->", OUT)
