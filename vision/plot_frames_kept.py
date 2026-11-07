"""Filmstrip: the 64 sampled frames of one video, shaded by how much of each
frame OVAL's page selection keeps. -> results/figures/videomme_frames_kept.pdf

Reads results/vision/token_selection.npz (per-video retention, written by
vision/token_selection.py) and the PNG frame cache in $VMME_ROOT/frames.

    python vision/plot_frames_kept.py                 # first video in the npz
    python vision/plot_frames_kept.py --video <id>    # a specific videoID
    python vision/plot_frames_kept.py --list          # which videos are available

Frames are shown at full colour in proportion to their retention and washed out
toward the page background when the selector drops them; the border carries the
same number on a one-hue ramp. The bar strip under the grid is the per-frame
retention against the uniform share the budget allows.
Runs with any python that has numpy + matplotlib + PIL.
"""
import argparse, json, os

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results", "vision")
VMME = os.environ.get("VMME_ROOT", os.path.expanduser("~/vmme"))
OUT = os.path.join(HERE, "..", "results", "figures", "videomme_frames_kept.pdf")
CMAP = LinearSegmentedColormap.from_list("coral", ["#FFF4F1", "#F3B3A5", "#E8806F", "#A8412F", "#5C1C13"])
NCOL = 8

ap = argparse.ArgumentParser()
ap.add_argument("--video", default=None)
ap.add_argument("--list", action="store_true")
ap.add_argument("--out", default=OUT)
a = ap.parse_args()

meta = json.load(open(os.path.join(R, "token_selection.json")))
z = np.load(os.path.join(R, "token_selection.npz"))
ids = [str(v) for v in z["video_ids"]]
if a.list:
    for i, v in enumerate(ids):
        print(f"{v}  mean retention {z['per_video'][i].mean()*100:.1f}%")
    raise SystemExit
vid = a.video or ids[0]
r = z["per_video"][ids.index(vid)]
nf = len(r)
uniform = min(1.0, meta["page_budget"] / np.mean([m["pages"] for m in meta["videos"]]))

cache = os.path.join(VMME, "frames", f"{vid}_n{nf}")
if not os.path.isdir(cache):
    raise SystemExit(f"no frame cache at {cache} (set VMME_ROOT, or run vision/frames.py)")

nrow = int(np.ceil(nf / NCOL))
vmax = float(r.max())
norm = Normalize(0, vmax)
# tiles are square: height follows the tile width so no vertical slack is left over
tile = 7.2 / (NCOL + 0.45)
fig = plt.figure(figsize=(7.2, nrow * tile + 1.55))
# explicit margins: tight_layout fights a gridspec whose colorbar spans rows
gs = fig.add_gridspec(nrow + 1, NCOL + 1, height_ratios=[1] * nrow + [0.80],
                      width_ratios=[1] * NCOL + [0.035], hspace=0.06, wspace=0.06,
                      left=0.055, right=0.885, top=0.945, bottom=0.075)

for i in range(nf):
    ax = fig.add_subplot(gs[i // NCOL, i % NCOL])
    img = np.asarray(Image.open(os.path.join(cache, f"{i:03d}.png")).convert("RGB"), float) / 255
    keep = r[i] / vmax if vmax > 0 else 0.0
    # wash dropped frames toward white; kept frames stay at full colour
    alpha = 0.22 + 0.78 * keep
    ax.imshow(np.clip(img * alpha + (1 - alpha), 0, 1))
    for s in ax.spines.values():
        s.set_edgecolor(CMAP(norm(r[i]))); s.set_linewidth(2.0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.04, 0.96, f"{i}", transform=ax.transAxes, fontsize=5.5, va="top", ha="left",
            color="white", bbox=dict(boxstyle="square,pad=0.12", fc="#00000088", ec="none"))
    ax.text(0.96, 0.04, f"{r[i]*100:.0f}%", transform=ax.transAxes, fontsize=5.5, va="bottom",
            ha="right", color="white", bbox=dict(boxstyle="square,pad=0.12", fc="#00000088", ec="none"))

cax = fig.add_subplot(gs[:nrow, NCOL])
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
cb.set_label("fraction of the frame's tokens kept", fontsize=7.5)
cb.ax.tick_params(labelsize=7)

axb = fig.add_subplot(gs[nrow, :NCOL])
axb.bar(np.arange(nf), r * 100, width=0.86, color=[CMAP(norm(x)) for x in r])
axb.axhline(uniform * 100, color="#5F6B7D", lw=1, ls="--", alpha=0.85,
            label=f"uniform share ({uniform*100:.0f}%)")
axb.set_xlim(-0.6, nf - 0.4)
axb.set_xlabel(f"frame index (0 = start of video, {nf - 1} = end)", fontsize=8)
axb.set_ylabel("kept (%)", fontsize=8)
axb.tick_params(labelsize=7)
axb.grid(alpha=0.3, axis="y")
axb.set_axisbelow(True)
for s in ("top", "right"):
    axb.spines[s].set_visible(False)
axb.legend(fontsize=7, frameon=False, loc="upper center")

fig.suptitle(f"What OVAL keeps of {vid}: {nf} sampled frames, budget {meta['budget']} of "
             f"{meta['videos'][ids.index(vid)]['n_visual']} visual tokens",
             fontsize=9, color="#333333", y=0.985)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
fig.savefig(a.out)
fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=200)
print("wrote", os.path.abspath(a.out))
