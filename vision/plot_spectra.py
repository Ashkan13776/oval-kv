"""Plot results/vision/page_spectra.npz (written by vision/page_spectra.py) -> figures/page_spectra.pdf.
Runs with any python that has numpy + matplotlib (the harness env has no matplotlib)."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results", "vision")
z = np.load(os.path.join(R, "page_spectra.npz")); meta = json.load(open(os.path.join(R, "page_spectra.json")))
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
for ax, (lab, a, b) in zip(axs, [(f"keys, rank {meta['key_rank']}", z["vis_key"], z["text_key"]),
                                 (f"values, rank {meta['value_rank']}", z["vis_value"], z["text_value"])]):
    bp = ax.boxplot([a, b], labels=["video pages", "text pages"], widths=0.5, showfliers=False,
                    patch_artist=True, medianprops=dict(color="black"))
    for patch, c in zip(bp["boxes"], ["#E8806F", "#5F6B7D"]):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    ax.scatter([1, 2], [a.mean(), b.mean()], marker="D", color="black", s=14, zorder=3)
    ax.set_title(lab, fontsize=10); ax.set_ylabel("centred energy captured per page"); ax.grid(alpha=0.3)
fig.tight_layout()
out = os.path.join(HERE, "..", "figures", "page_spectra.pdf"); os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out); print("wrote", os.path.abspath(out))
