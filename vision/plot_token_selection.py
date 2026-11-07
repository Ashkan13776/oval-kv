"""Plot results/vision/token_selection.npz -> results/figures/videomme_token_selection.pdf

(a) per-frame retention of OVAL's page selection, layer x frame;
(b) the marginal over layers, against the uniform share the budget would give.
Runs with any python that has numpy + matplotlib.
"""
import json, os

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results", "vision")
OUT = os.path.join(HERE, "..", "results", "figures", "videomme_token_selection.pdf")

meta = json.load(open(os.path.join(R, "token_selection.json")))
frac = np.load(os.path.join(R, "token_selection.npz"))["retention"]      # [layers>=1, frames]
nl, nf = frac.shape
uniform = min(1.0, meta["page_budget"] / np.mean([m["pages"] for m in meta["videos"]]))

# sequential ramp: one hue, light -> dark (the coral of the other vision figures)
cmap = LinearSegmentedColormap.from_list("coral", ["#FFF4F1", "#F3B3A5", "#E8806F", "#A8412F", "#5C1C13"])

# a 2x2 grid so the colorbar takes width from its own column, not from the shared x axis
fig = plt.figure(figsize=(7.2, 4.4))
gs = fig.add_gridspec(2, 2, height_ratios=[2.6, 1.0], width_ratios=[1, 0.022],
                      hspace=0.12, wspace=0.02)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
cax = fig.add_subplot(gs[0, 1])
fig.add_subplot(gs[1, 1]).axis("off")

im = ax0.imshow(frac, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=frac.max(),
                extent=[-0.5, nf - 0.5, 0.5, nl + 0.5], interpolation="nearest")
ax0.set_ylabel("layer")
ax0.tick_params(labelbottom=False)
cb = fig.colorbar(im, cax=cax)
cb.set_label("fraction of the frame's tokens kept", fontsize=8)
cb.ax.tick_params(labelsize=8)
ax0.set_title(f"OVAL page selection over the {nf} sampled frames  "
              f"({meta['n_videos']} Video-MME Long videos, budget {meta['budget']} of "
              f"{int(np.mean([m['n_visual'] for m in meta['videos']]))} visual tokens)",
              fontsize=8.5, color="#333333")

m = frac.mean(0)
ax1.fill_between(np.arange(nf), frac.min(0), frac.max(0), color="#E8806F", alpha=0.18,
                 lw=0, zorder=2, label="min-max over layers")
ax1.plot(np.arange(nf), m, color="#E8806F", lw=2, zorder=3, label="mean over layers")
ax1.axhline(uniform, color="#5F6B7D", lw=1, ls="--", alpha=0.85, zorder=1,
            label=f"uniform share ({uniform*100:.0f}%)")
ax1.set_xlabel(f"frame index (0 = start of video, {nf - 1} = end)")
ax1.set_ylabel("retention")
ax1.set_xlim(-0.5, nf - 0.5)
ax1.grid(alpha=0.3)
ax1.set_axisbelow(True)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)
ax1.legend(fontsize=8, frameon=False, loc="upper center", ncol=3)

fig.tight_layout()
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
fig.savefig(OUT)
fig.savefig(os.path.splitext(OUT)[0] + ".png", dpi=200)
print("wrote", os.path.abspath(OUT))
