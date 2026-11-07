"""Filmstrip: the 64 sampled frames of one video, shaded by how much of each
frame OVAL's page selection keeps. -> results/figures/videomme_frames_kept.pdf

Reads results/vision/token_selection.npz (per-video retention, written by
vision/token_selection.py) and the PNG frame cache in $VMME_ROOT/frames.

    python vision/plot_frames_kept.py                 # first video in the npz
    python vision/plot_frames_kept.py --video <id>    # a specific videoID
    python vision/plot_frames_kept.py --list          # which videos are available

Frames keep their colour in proportion to their retention and wash out toward
the page when the selector drops them; the border and the per-tile label carry
the same number. No title and no axes -- the script prints a caption instead.
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
WIDTH = 9.0          # generous, so each 384x384 frame is barely downsampled in the raster

ap = argparse.ArgumentParser()
ap.add_argument("--video", default=None)
ap.add_argument("--list", action="store_true")
ap.add_argument("--dpi", type=int, default=300)
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
tile = WIDTH / (NCOL + 0.30)
fig = plt.figure(figsize=(WIDTH, nrow * tile + 0.08))
# explicit margins: tight_layout fights a gridspec whose colorbar spans every row
gs = fig.add_gridspec(nrow, NCOL + 1, width_ratios=[1] * NCOL + [0.030],
                      hspace=0.035, wspace=0.035,
                      left=0.005, right=0.932, top=0.995, bottom=0.005)

for i in range(nf):
    ax = fig.add_subplot(gs[i // NCOL, i % NCOL])
    img = np.asarray(Image.open(os.path.join(cache, f"{i:03d}.png")).convert("RGB"), float) / 255
    keep = r[i] / vmax if vmax > 0 else 0.0
    # wash dropped frames toward white; the floor keeps them legible, and the
    # border plus the label carry the exact value
    alpha = 0.30 + 0.70 * keep
    ax.imshow(np.clip(img * alpha + (1 - alpha), 0, 1), interpolation="lanczos")
    for s in ax.spines.values():
        s.set_edgecolor(CMAP(norm(r[i]))); s.set_linewidth(2.4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.035, 0.965, f"{i}", transform=ax.transAxes, fontsize=6.5, va="top", ha="left",
            color="white", bbox=dict(boxstyle="square,pad=0.14", fc="#000000AA", ec="none"))
    ax.text(0.965, 0.035, f"{r[i]*100:.0f}%", transform=ax.transAxes, fontsize=6.5, va="bottom",
            ha="right", color="white", bbox=dict(boxstyle="square,pad=0.14", fc="#000000AA", ec="none"))

cax = fig.add_subplot(gs[:, NCOL])
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
cb.set_label("fraction of the frame's tokens kept", fontsize=8)
cb.ax.tick_params(labelsize=7.5)

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
fig.savefig(a.out)
fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=a.dpi)
print("wrote", os.path.abspath(a.out))
print("caption: The %d frames sampled from Video-MME video %s, shaded by the fraction of each "
      "frame's visual tokens that OVAL's page selection keeps (border and label give the same "
      "value; a uniform selection would keep %.0f%% of every frame). Query-agnostic protocol, "
      "budget %d of %d visual tokens, rank-8 key records, per-head eta table %s."
      % (nf, vid, uniform * 100, meta["budget"],
         meta["videos"][ids.index(vid)]["n_visual"], meta["eta_table_hash"]))
