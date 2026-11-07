"""Filmstrip: the sampled frames of one video, shaded by how much of each frame
OVAL's page selection keeps. -> results/figures/videomme_frames_kept.pdf

Reads results/vision/token_selection.npz (per-video retention, written by
vision/token_selection.py) and the PNG frame cache in $VMME_ROOT/frames.

    python vision/plot_frames_kept.py                    # single column: 3.4 x 4.1in, 8 cols, all 64
    python vision/plot_frames_kept.py --cols 8 --width 7 # two-column width
    python vision/plot_frames_kept.py --cols 4 --every 2 # bigger tiles, every other frame
                                                         # (NB: --every 2 drops frames 1 and 63,
                                                         #  the two highest-retention frames)
    python vision/plot_frames_kept.py --video <id>       # a specific videoID
    python vision/plot_frames_kept.py --list             # which videos are available

Frames keep their colour in proportion to their retention and wash out toward
the page when the selector drops them. Each tile carries one solid footer strip
"<frame index>  <percent kept>", filled with the retention colour, which is the
only legend the figure needs. No title, no axes, no colourbar -- the script
prints a caption instead.
Runs with any python that has numpy + matplotlib + PIL.
"""
import argparse, json, os

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results", "vision")
VMME = os.environ.get("VMME_ROOT", os.path.expanduser("~/vmme"))
OUT = os.path.join(HERE, "..", "results", "figures", "videomme_frames_kept.pdf")
CMAP = LinearSegmentedColormap.from_list("coral", ["#FFF4F1", "#F3B3A5", "#E8806F", "#A8412F", "#5C1C13"])

ap = argparse.ArgumentParser()
ap.add_argument("--video", default=None)
ap.add_argument("--list", action="store_true")
ap.add_argument("--cols", type=int, default=8)
ap.add_argument("--width", type=float, default=3.4, help="figure width in inches (3.4 = single column)")
ap.add_argument("--every", type=int, default=1, help="show every Nth frame")
ap.add_argument("--dpi", type=int, default=400)
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
r_all = z["per_video"][ids.index(vid)]
nf_all = len(r_all)
shown = list(range(0, nf_all, a.every))
uniform = min(1.0, meta["page_budget"] / np.mean([m["pages"] for m in meta["videos"]]))

cache = os.path.join(VMME, "frames", f"{vid}_n{nf_all}")
if not os.path.isdir(cache):
    raise SystemExit(f"no frame cache at {cache} (set VMME_ROOT, or run vision/frames.py)")

NCOL = a.cols
nrow = int(np.ceil(len(shown) / NCOL))
vmax = float(r_all.max())
norm = Normalize(0, vmax)
tile = a.width / NCOL                       # inches per tile, edge to edge
FOOT = 0.20                                 # footer strip height, as a fraction of the tile
fig = plt.figure(figsize=(a.width, nrow * tile * (1 + FOOT)))
gs = fig.add_gridspec(nrow, NCOL, hspace=0.035, wspace=0.035,
                      left=0.004, right=0.996, top=0.997, bottom=0.003)
# label size follows the tile so the strip is always near-filled
fs = np.clip(tile * 10.5, 5.0, 13.0)

for k, i in enumerate(shown):
    ax = fig.add_subplot(gs[k // NCOL, k % NCOL])
    img = np.asarray(Image.open(os.path.join(cache, f"{i:03d}.png")).convert("RGB"), float) / 255
    keep = r_all[i] / vmax if vmax > 0 else 0.0
    alpha = 0.30 + 0.70 * keep
    img = np.clip(img * alpha + (1 - alpha), 0, 1)
    h, w = img.shape[:2]
    # paint the footer strip into the image itself so it scales exactly with the tile
    fh = max(1, int(round(h * FOOT)))
    strip = np.empty((fh, w, 3))
    strip[:] = np.array(CMAP(norm(r_all[i]))[:3])
    ax.imshow(np.vstack([img, strip]), interpolation="lanczos")
    for s in ax.spines.values():
        s.set_edgecolor(CMAP(norm(r_all[i]))); s.set_linewidth(1.2)
    ax.set_xticks([]); ax.set_yticks([])
    # white on the dark end of the ramp, near-black on the light end
    ink = "white" if norm(r_all[i]) > 0.45 else "#2A0D08"
    ax.text(0.5, (fh / 2) / (h + fh), f"{i}   {r_all[i]*100:.0f}%", transform=ax.transAxes,
            fontsize=fs, va="center", ha="center", color=ink, fontweight="bold")

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
fig.savefig(a.out)
fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=a.dpi)
print("wrote", os.path.abspath(a.out), f"({len(shown)} frames, {NCOL}x{nrow}, {a.width}in)")
print("caption: Frames sampled from Video-MME video %s, shaded by the fraction of each frame's "
      "visual tokens that OVAL's page selection keeps; the strip under each frame gives the frame "
      "index and that fraction (a uniform selection would keep %.0f%% of every frame). "
      "Query-agnostic protocol, budget %d of %d visual tokens, rank-8 key records, per-head eta "
      "table %s." % (vid, uniform * 100, meta["budget"],
                     meta["videos"][ids.index(vid)]["n_visual"], meta["eta_table_hash"]))
