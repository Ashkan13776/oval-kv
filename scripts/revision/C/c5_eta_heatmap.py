"""C5: eta_h on video vs reasoning (no GPU). F1-style heatmaps (same ramp and layout as
analysis/fig_eta_revision.py on main): one strip per table, KV head x layer, layer 0 dense and
not shown. Tables: R1-Distill Qwen-7B reasoning (OVAL setting, value rank 16; MATH500, AIME24,
GPQA), LLaVA-Video-7B-Qwen2 (Tab. 5, video calibration), Qwen2.5-VL-7B (C1, video calibration).
All five are 28 layers x 4 KV heads. Also prints, per pair, Pearson r over (layer >= 1, head) and
the mean eta difference, so "does visual prefill move eta_h" is a number.

    python scripts/revision/C/c5_eta_heatmap.py
-> results/revision/C/figures/c5_eta_heatmaps.{pdf,csv}, results/revision/C/tables/c5_eta_pairs.csv
"""
import csv, json, os
import numpy as np

import figstyle as S
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
RES = os.path.join(ROOT, "results", "revision", "C")
TABLES = [
    ("Qwen-7B reasoning, MATH500", "results/eta_rules/vr16/tables/ds-r1-qwen-7b_MATH50_r1.json"),
    ("Qwen-7B reasoning, AIME24", "results/eta_rules/vr16/tables/ds-r1-qwen-7b_AIME24_r1.json"),
    ("Qwen-7B reasoning, GPQA", "results/eta_rules/vr16/tables/ds-r1-qwen-7b_GPQA50c_r1.json"),
    ("LLaVA-Video-7B, video", "results/vision/eta/tables/llava-video-7b_VCAL_r1.json"),
    ("Qwen2.5-VL-7B, video", "results/revision/C/eta/tables/qwen2.5-vl-7b_VCAL_r1.json"),
]


def main():
    os.makedirs(os.path.join(RES, "figures"), exist_ok=True); os.makedirs(os.path.join(RES, "tables"), exist_ok=True)
    tabs = [(n, json.load(open(os.path.join(ROOT, p)))) for n, p in TABLES]
    fig, axs = plt.subplots(len(tabs), 1, figsize=(S.COL, 0.78 * len(tabs) + 0.3), sharex=True,
                            gridspec_kw=dict(hspace=0.75))
    rows = []
    for ax, (n, t) in zip(axs, tabs):
        eta = np.array(t["eta"]); L, H = eta.shape
        im = ax.imshow(eta[1:].T, aspect="auto", origin="lower", cmap=S.SEQ, vmin=0, vmax=1,
                       interpolation="nearest", extent=[0.5, L - 0.5, -0.5, H - 0.5])
        v = eta[1:].ravel()
        ax.set_title(f"{n}: median {np.median(v):.2f}, {(v == 0).mean()*100:.0f}% at 0, "
                     f"{(v == 1).mean()*100:.0f}% at 1", fontsize=6.5, color=S.INK, pad=2, loc="left")
        ax.set_yticks(range(H)); ax.tick_params(length=0); ax.set_ylabel("head", labelpad=1)
        for s in ax.spines.values():
            s.set_visible(False)
        rows += [[n, t["hash"], li, hi, eta[li, hi]] for li in range(L) for hi in range(H)]
    axs[-1].set_xlabel("layer (layer 0 is dense)", labelpad=1)
    cb = fig.colorbar(im, ax=axs, fraction=0.03, pad=0.02); cb.set_label("$\\eta_h$"); cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    fig.savefig(os.path.join(RES, "figures", "c5_eta_heatmaps.pdf")); plt.close(fig)
    with open(os.path.join(RES, "figures", "c5_eta_heatmaps.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["table", "hash", "layer", "kv_head", "eta"]); w.writerows(rows)
    pr = []
    for i in range(len(tabs)):
        for j in range(i + 1, len(tabs)):
            a = np.array(tabs[i][1]["eta"])[1:].ravel(); b = np.array(tabs[j][1]["eta"])[1:].ravel()
            pr.append([tabs[i][0], tabs[j][0], f"{np.corrcoef(a, b)[0, 1]:.3f}", f"{(b - a).mean():+.3f}",
                       f"{np.abs(b - a).mean():.3f}"])
    with open(os.path.join(RES, "tables", "c5_eta_pairs.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["table_a", "table_b", "pearson_r", "mean_b_minus_a", "mean_abs_diff"]); w.writerows(pr)
    for p in pr:
        print(" | ".join(p))


if __name__ == "__main__":
    main()
