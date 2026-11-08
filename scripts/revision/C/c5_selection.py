"""C5: where does OVAL's page selection go? Per-frame selection maps from the REAL OVAL runs on
the Video-MME Long split (no GPU; reads the `sellog` field written by sellog.py).

Per question: fraction of frame f's tokens that sat inside a selected page, over all
(decode step, layer >= 1, KV head) decisions = sel_frame / avail_frame. Averaged over questions
(each question weighs 1). LLaVA-Video: 64 frames. Qwen2.5-VL: 32 temporal groups (each holds
frames 2g and 2g+1 in one token block), plotted at frame 2g + 0.5 and expanded to both frames in
the map. Uniform selection would keep ~ page_budget / selectable pages (~14%) of every frame.

95% CI per frame: bootstrap over videos (Video-MME has 3 questions per video), 2,000 draws.
Start/end bias as numbers: mean selected fraction of the first 4 and last 4 frames vs the middle.

    python scripts/revision/C/c5_selection.py
-> results/revision/C/figures/c5_selection_{map,line}.{pdf,csv}, results/revision/C/c5_selection.md
"""
import csv, os, sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_c import load

RES = os.path.join(HERE, "..", "..", "..", "results", "revision", "C")
SRC = [("LLaVA-Video-7B", "c5_sellog_llava_videomme/oval_n64", 900),
       ("Qwen2.5-VL-7B", "c1_qwen25vl/oval_n64", 900)]


def per_question(rows):
    out = {}
    for q, r in rows.items():
        if r.get("group", "long") != "long" or "sellog" not in r:
            continue
        s = r["sellog"]
        f = np.array(s["sel_frame"]) / np.maximum(np.array(s["avail_frame"]), 1)
        if len(f) == 32:                        # Qwen temporal groups -> 64 frames
            f = np.repeat(f, 2)
        out[q] = (r["videoID"], f)
    return out


def boot_v(pq, n=2000, seed=0):
    byv = defaultdict(list)
    for v, f in pq.values():
        byv[v].append(f)
    vids = list(byv)
    S = np.array([np.sum(byv[v], 0) for v in vids]); C = np.array([len(byv[v]) for v in vids])
    idx = np.random.default_rng(seed).integers(0, len(vids), (n, len(vids)))
    bs = S[idx].sum(1) / C[idx].sum(1)[:, None]
    return S.sum(0) / C.sum(), np.quantile(bs, 0.025, 0), np.quantile(bs, 0.975, 0), bs


def main():
    os.makedirs(os.path.join(RES, "figures"), exist_ok=True)
    L = ["# C5: per-frame selection (OVAL, Video-MME Long, real runs)", ""]
    res = {}
    for model, d, tot in SRC:
        pq = per_question(load(os.path.join(RES, "runs", d)))
        if not pq:
            L += [f"- {model}: no selection logs yet (not run).", ""]; continue
        m, lo, hi, bs = boot_v(pq)
        ends = lambda x: np.concatenate([x[..., :4], x[..., -4:]], -1).mean(-1)
        mid = lambda x: x[..., 4:-4].mean(-1)
        ratio = ends(bs) / mid(bs)
        res[model] = (m, lo, hi)
        L += [f"## {model}: {len(pq)} of {tot} Long questions{'' if len(pq) >= tot else ' (INCOMPLETE)'}", "",
              f"- mean selected fraction, all frames: {100 * m.mean():.2f}%",
              f"- first 4 + last 4 frames: {100 * ends(m):.2f}%; frames 4-59: {100 * mid(m):.2f}%; "
              f"ratio {ends(m) / mid(m):.2f} [{np.quantile(ratio, 0.025):.2f}, {np.quantile(ratio, 0.975):.2f}] "
              "(bootstrap over videos)",
              f"- frame 0: {100 * m[0]:.2f}%, frame 63: {100 * m[-1]:.2f}%; max at frame {int(m.argmax())} "
              f"({100 * m.max():.2f}%), min at frame {int(m.argmin())} ({100 * m.min():.2f}%)", ""]
    open(os.path.join(RES, "c5_selection.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    if not res:
        return
    import figstyle as S
    import matplotlib.pyplot as plt
    with open(os.path.join(RES, "figures", "c5_selection_line.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "frame", "mean_selected_fraction", "lo", "hi"])
        for model, (m, lo, hi) in res.items():
            w.writerows([[model, i, f"{m[i]:.5f}", f"{lo[i]:.5f}", f"{hi[i]:.5f}"] for i in range(64)])
    # line plot
    fig, ax = plt.subplots(figsize=(S.COL, 1.6))
    for (model, (m, lo, hi)), c in zip(res.items(), [S.BLUE, S.ORANGE]):
        x = np.arange(64)
        ax.fill_between(x, 100 * lo, 100 * hi, color=c, alpha=0.2, lw=0)
        ax.plot(x, 100 * m, color=c, lw=1.0, label=model)
    ax.set_xlabel("frame index"); ax.set_ylabel("selected (%)"); ax.set_xlim(0, 63)
    ax.grid(axis="y", color=S.GRID, lw=0.5); ax.legend(frameon=False)
    fig.savefig(os.path.join(RES, "figures", "c5_selection_line.pdf")); plt.close(fig)
    # map: one 1 x 64 strip per model, shared colour scale
    vmax = max(m.max() for m, _, _ in res.values()) * 100
    fig, axs = plt.subplots(len(res), 1, figsize=(S.COL, 0.42 * len(res) + 0.35), squeeze=False)
    for ax, (model, (m, _, _)) in zip(axs[:, 0], res.items()):
        im = ax.imshow(100 * m[None], aspect="auto", cmap=S.SEQ, vmin=0, vmax=vmax, interpolation="nearest",
                       extent=[-0.5, 63.5, 0, 1])
        ax.set_yticks([]); ax.set_title(model, fontsize=7, loc="left", pad=2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axs[-1, 0].set_xlabel("frame index")
    cb = fig.colorbar(im, ax=axs[:, 0], fraction=0.04, pad=0.02); cb.set_label("selected (%)"); cb.outline.set_visible(False)
    fig.savefig(os.path.join(RES, "figures", "c5_selection_map.pdf")); plt.close(fig)
    with open(os.path.join(RES, "figures", "c5_selection_map.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "frame", "mean_selected_fraction"])
        for model, (m, _, _) in res.items():
            w.writerows([[model, i, f"{m[i]:.5f}"] for i in range(64)])


if __name__ == "__main__":
    main()
