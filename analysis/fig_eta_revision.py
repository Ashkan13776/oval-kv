"""Block A5 figures (no GPU). Each figure is written as PDF plus the CSV behind it.

F1  eta_h heatmap (layers x KV heads) + histogram, per (model, task), from the
    deployed value-rank-16 tables (results/eta_rules/vr16/tables, and
    results/revision/A/calib_vr16/tables for Llama-8B once calibrated).
F2  six heads picked by quantile of b_h - a_h (two low, two middle, two high), per
    model on MATH500: held-out output error on the 21-point eta grid (dots, %% above
    the head's grid minimum), the endpoint quadratic f_h built from the calibration
    terms a, b, c (line, %% above its own minimum), and the rule's eta_h (star).
    Value rank 8, the only rank with the full grid (results/eta_rules/<model>_MATH50.json).
F3  histogram over (layer >= 1, KV head) of (L(eta_h) - grid min) / grid min on the
    held-out traces, per model, all three tasks pooled; value rank 8.

  python analysis/fig_eta_revision.py
"""
import csv, json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/revision/A/figures"; OUT.mkdir(parents=True, exist_ok=True)
TASKS = ["MATH50", "AIME24", "GPQA50c"]
NICE = {"ds-r1-qwen-7b": "R1-Distill Qwen-7B", "ds-r1-llama-8b": "R1-Distill Llama-8B",
        "ds-r1-qwen-14b": "R1-Distill Qwen-14B", "MATH50": "MATH500", "AIME24": "AIME24", "GPQA50c": "GPQA"}


def tables_vr16():
    found = {}
    for d in [ROOT / "results/eta_rules/vr16/tables", ROOT / "results/revision/A/calib_vr16/tables"]:
        if d.exists():
            for f in sorted(d.glob("*_r1.json")):
                t = json.load(open(f)); found[(t["model"], t["task"])] = (np.array(t["eta"]), t["hash"], str(f))
    return found


def f1():
    found = tables_vr16()
    rows = []
    for (m, t), (eta, h, src) in sorted(found.items()):
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(7.5, 3.2), gridspec_kw={"width_ratios": [1.3, 1]})
        im = a0.imshow(eta[1:].T, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis",
                       extent=[0.5, eta.shape[0] - 0.5, -0.5, eta.shape[1] - 0.5])
        a0.set_xlabel("layer (layer 0 is dense)"); a0.set_ylabel("KV head")
        a0.set_title(f"{NICE[m]}, {NICE[t]}: $\\eta_h$", fontsize=9)
        fig.colorbar(im, ax=a0, fraction=0.05)
        v = eta[1:].ravel()
        a1.hist(v, bins=np.linspace(0, 1, 21), color="0.35")
        a1.set_xlabel("$\\eta_h$"); a1.set_ylabel("KV heads")
        a1.set_title(f"mean {v.mean():.2f}, median {np.median(v):.2f}; "
                     f"{(v == 0).mean()*100:.0f}% at 0, {(v == 1).mean()*100:.0f}% at 1", fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / f"F1_eta_heatmap_{m}_{t}.pdf"); plt.close(fig)
        for li in range(eta.shape[0]):
            for hi in range(eta.shape[1]):
                rows.append([m, t, h, li, hi, eta[li, hi]])
    with open(OUT / "F1_eta_heatmap.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["model", "task", "table_hash", "layer", "kv_head", "eta"]); w.writerows(rows)
    return sorted(found)


def stats_path(m, t):
    """Prefer the OVAL-setting grid (rank-8 keys, rank-16 values) when it exists."""
    g = ROOT / f"results/revision/A/grid_vr16/{m}_{t}.json"
    if g.exists():
        st = json.load(open(g))
        if len(st["grid"]) == 21 and "regret" in st:
            return g, 16
    return ROOT / f"results/eta_rules/{m}_{t}.json", 8


def f2():
    rows = []
    for m in ["ds-r1-qwen-7b", "ds-r1-llama-8b"]:
        sp, vr = stats_path(m, "MATH50"); st = json.load(open(sp))
        grid = np.array(st["grid"]); test = np.array(st["loss_grid"]["test"])     # [L][H][21]
        r1 = st["rules"]["r1"]; a, b, c = (np.array(r1[k]) for k in "abc"); eta = np.array(r1["eta"])
        L, H = a.shape
        idx = [(li, hi) for li in range(1, L) for hi in range(H)]
        diff = np.array([b[li, hi] - a[li, hi] for li, hi in idx])
        order = np.argsort(diff); n = len(order)
        pick = [order[1], order[3], order[n // 2 - 1], order[n // 2 + 1], order[-4], order[-2]]
        fig, axs = plt.subplots(1, 6, figsize=(13, 2.6), sharey=False)
        for ax, k in zip(axs, pick):
            li, hi = idx[k]
            y = test[li, hi]; ymin = y.min(); yp = 100 * (y - ymin) / ymin
            fg = (1 - grid) ** 2 * a[li, hi] + grid ** 2 * b[li, hi] + 2 * grid * (1 - grid) * c[li, hi]
            fp = 100 * (fg - fg.min()) / fg.min()
            ax.plot(grid, yp, "o", ms=3, color="0.2", label="held-out error")
            ax.plot(grid, fp, "-", color="tab:orange", label="$f_h$ (calibration)")
            e = eta[li, hi]; ye = 100 * (np.interp(e, grid, y) - ymin) / ymin
            ax.plot([e], [ye], "*", ms=11, color="tab:red", label="$\\eta_h$")
            ax.set_title(f"L{li} H{hi}, $b-a$={diff[k]:+.1e}", fontsize=8); ax.set_xlabel("$\\eta$")
            for gi, gv in enumerate(grid):
                rows.append([m, li, hi, float(diff[k]), gv, y[gi], yp[gi], fg[gi], fp[gi], e])
        axs[0].set_ylabel("% above minimum"); axs[0].legend(fontsize=7)
        fig.suptitle(f"{NICE[m]}, MATH500, rank-8 keys, value rank {vr}", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / f"F2_eta_fit_{m}_MATH50.pdf"); plt.close(fig)
    with open(OUT / "F2_eta_fit.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "layer", "kv_head", "b_minus_a", "eta", "test_error", "test_pct_above_min",
                    "f_h", "f_h_pct_above_min", "eta_h"]); w.writerows(rows)


def f3():
    rows = []; fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.0))
    for ax, m in zip(axs, ["ds-r1-qwen-7b", "ds-r1-llama-8b"]):
        vals = []
        for t in TASKS:
            sp, vr = stats_path(m, t); st = json.load(open(sp))
            loss = np.array(st["regret"]["tables"]["r1"]["loss"]); gmin = np.array(st["regret"]["grid_min_test"])
            rel = (loss - gmin) / gmin
            for li in range(1, rel.shape[0]):
                for hi in range(rel.shape[1]):
                    vals.append(100 * rel[li, hi]); rows.append([m, t, li, hi, loss[li, hi], gmin[li, hi], 100 * rel[li, hi]])
        v = np.array(vals)
        ax.hist(v, bins=40, color="0.35")
        ax.set_xlabel("held-out error of $\\eta_h$ above the grid minimum (%)"); ax.set_ylabel("KV heads")
        ax.set_title(f"{NICE[m]} (3 tasks, value rank {vr})\nmedian {np.median(v):.2f}%, "
                     f"90th percentile {np.percentile(v, 90):.2f}%", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "F3_eta_regret_hist.pdf"); plt.close(fig)
    with open(OUT / "F3_eta_regret_hist.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["model", "task", "layer", "kv_head", "loss_eta_h", "grid_min", "pct_above"]); w.writerows(rows)


if __name__ == "__main__":
    print("F1 tables:", f1()); f2(); f3(); print("written to", OUT)
