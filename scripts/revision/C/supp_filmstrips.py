"""Supplementary filmstrips: OVAL's per-frame selection on one video per benchmark, for both models
(the paper's earlier single-video figure, extended to Qwen2.5-VL and LVBench).

Tile = one Qwen2.5-VL temporal group (frames 2g and 2g+1; the image shown is frame 2g). Value = share of
the tile's visual tokens selected by OVAL, averaged over decoding steps, layers, heads, and all questions
of the video (from the read-only selection log, `sellog`, of the real runs). For LLaVA-Video, which
selects per frame, the two frames of a tile are pooled (selected / available), so both models are shown
at the same granularity. Both models sample the same 64 frame indices (checked from the frame-cache meta).

Video choice (no hand-picking): per benchmark, among videos whose questions all ran for both models
(Video-MME: Long split), the one whose two per-tile profiles are closest (L2, summed over models) to the
benchmark-wide mean profiles.

    python scripts/revision/C/supp_filmstrips.py  ->  results/revision/C/figures/supp_filmstrips.pdf (+ .csv)
"""
import csv, glob, json, os
from collections import defaultdict

import numpy as np
from PIL import Image
from matplotlib.colors import Normalize

from figstyle import plt, SEQ, INK, INK2, WIDTH

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
RUNS = os.path.join(ROOT, "results", "revision", "C", "runs")
OUT = os.path.join(ROOT, "results", "revision", "C", "figures", "supp_filmstrips")
SRC = {  # (bench, model) -> run dir with sellog
    ("videomme", "LLaVA-Video"): "c5_sellog_llava_videomme/oval_n64",
    ("videomme", "Qwen2.5-VL"): "c1_qwen25vl/oval_n64",
    ("lvbench", "LLaVA-Video"): "c2_lvbench_llava/oval_n64",
    ("lvbench", "Qwen2.5-VL"): "c2_lvbench_qwen25vl/oval_n64",
}
FRAMES = {"videomme": os.path.expanduser("~/vmme/frames"), "lvbench": os.path.expanduser("~/lvbench/frames")}
BENCH = [("videomme", "Video-MME (Long)"), ("lvbench", "LVBench")]
MODELS = ["LLaVA-Video", "Qwen2.5-VL"]
G = 32


def tiles(rows):
    """per-video tile profile [32]: selected / available pooled over the video's questions."""
    sel, av = defaultdict(lambda: np.zeros(G)), defaultdict(lambda: np.zeros(G))
    for r in rows:
        s, a = np.array(r["sellog"]["sel_frame"], float), np.array(r["sellog"]["avail_frame"], float)
        if len(s) == 2 * G:                      # per-frame (LLaVA) -> pool pairs
            s, a = s.reshape(G, 2).sum(1), a.reshape(G, 2).sum(1)
        sel[r["videoID"]] += s; av[r["videoID"]] += a
    return {v: sel[v] / np.maximum(av[v], 1) for v in sel}


def load(d, long_only):
    out = []
    for f in sorted(glob.glob(os.path.join(RUNS, d, "shard?of4.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            if long_only and (r.get("group") or r.get("duration")) != "long":
                continue
            if r.get("sellog"):
                out.append(r)
    return out


def main():
    prof, pick = {}, {}
    for b, _ in BENCH:
        for m in MODELS:
            prof[b, m] = tiles(load(SRC[b, m], b == "videomme"))
        common = sorted(set(prof[b, MODELS[0]]) & set(prof[b, MODELS[1]]))
        mean = {m: np.mean([prof[b, m][v] for v in common], 0) for m in MODELS}
        pick[b] = min(common, key=lambda v: sum(np.linalg.norm(prof[b, m][v] - mean[m]) for m in MODELS))
    vmax = max(prof[b, m][pick[b]].max() for b, _ in BENCH for m in MODELS)
    norm = Normalize(0, vmax)
    NC, NR = 16, 2                                # 32 tiles per strip
    tile = WIDTH / NC
    FOOT = 0.24
    strip_h = NR * tile * (9 / 16) * (1 + FOOT)   # frames are ~16:9
    lab_h = 0.22
    H = len(BENCH) * (2 * (strip_h + lab_h) + 0.12)
    fig = plt.figure(figsize=(WIDTH, H))
    y = 1.0
    out = []
    for b, btitle in BENCH:
        vid = pick[b]
        cache = os.path.join(FRAMES[b], f"{vid}_n64")
        meta = json.load(open(os.path.join(cache, "meta.json")))
        for m in MODELS:
            p = prof[b, m][vid]
            y -= lab_h / H
            fig.text(0.0, y + 0.25 * lab_h / H, f"{btitle}, video {vid}  ({meta['video_time'] / 60:.0f} min)  -  {m}",
                     fontsize=7.5, color=INK, va="bottom")
            gs = fig.add_gridspec(NR, NC, left=0.0, right=1.0, top=y, bottom=y - strip_h / H, hspace=0.04, wspace=0.03)
            for g in range(G):
                ax = fig.add_subplot(gs[g // NC, g % NC])
                img = np.asarray(Image.open(os.path.join(cache, f"{2 * g:03d}.png")).convert("RGB").resize((160, 90)), float) / 255
                k = p[g] / vmax
                alpha = 0.30 + 0.70 * k
                img = np.clip(img * alpha + (1 - alpha), 0, 1)
                fh = int(round(90 * FOOT))
                col = np.array(SEQ(norm(p[g]))[:3])
                ax.imshow(np.vstack([img, np.tile(col, (fh, 160, 1))]), interpolation="lanczos", aspect="auto")
                for s in ax.spines.values():
                    s.set_edgecolor(col); s.set_linewidth(0.9)
                ax.set_xticks([]); ax.set_yticks([])
                ax.text(0.5, (fh / 2) / (90 + fh), f"{2 * g}  {100 * p[g]:.0f}%", transform=ax.transAxes, fontsize=5,
                        ha="center", va="center", color="white" if norm(p[g]) > 0.45 else INK)
                out.append([b, vid, m, g, 2 * g, f"{p[g]:.5f}"])
            y -= strip_h / H
        y -= 0.12 / H
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + ".pdf")
    with open(OUT + ".csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["bench", "videoID", "model", "tile", "first_frame", "selected_fraction"]); w.writerows(out)
    for b, _ in BENCH:
        print(b, pick[b], {m: f"{100 * prof[b, m][pick[b]].mean():.1f}%" for m in MODELS},
              "n videos", len(set(prof[b, MODELS[0]]) & set(prof[b, MODELS[1]])))


if __name__ == "__main__":
    main()
