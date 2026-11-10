"""Video summary figure for the paper (replaces the single-video selection figure).
(a) OVAL per-frame selection share, mean over all 900 Long Video-MME questions, both models (95% CI band,
    bootstrap over videos); from figures/c5_selection_line.csv.
(b) Paired accuracy change vs FullKV-64 on LVBench by where the annotated evidence lies in the video (thirds),
    OVAL and FreeKV, one panel per model (95% CI, bootstrap over questions); from tables/c5_evidence_paired.csv.
    python scripts/revision/C/fig_video_summary.py  ->  results/revision/C/figures/video_summary.pdf (+ .csv)
"""
import csv, os
from figstyle import plt, BLUE, ORANGE, INK2, GRID, COL, SEQ

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results", "revision", "C")
MODELS = [("LLaVA-Video-7B", "LLaVA-Video"), ("Qwen2.5-VL-7B", "Qwen2.5-VL")]


def rows(p):
    return list(csv.DictReader(open(p)))


def main():
    sel = rows(os.path.join(R, "figures", "c5_selection_line.csv"))
    ev = rows(os.path.join(R, "tables", "c5_evidence_paired.csv"))
    fig = plt.figure(figsize=(COL, 3.55))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.95, wspace=0.12)
    ax = fig.add_subplot(gs[0, :])
    out = []
    for (m, lab), c, ls in zip(MODELS, [SEQ(1.0), SEQ(0.45)], ["-", "--"]):
        r = [x for x in sel if x["model"] == m]
        f = [int(x["frame"]) for x in r]
        y, lo, hi = ([100 * float(x[k]) for x in r] for k in ("mean_selected_fraction", "lo", "hi"))
        ax.fill_between(f, lo, hi, color=c, alpha=0.2, lw=0)
        ax.plot(f, y, color=c, lw=1.1, ls=ls, label=lab)
        out += [["a", lab, "OVAL", a, b, l, h] for a, b, l, h in zip(f, y, lo, hi)]
    ax.set_xlim(0, 63); ax.set_ylim(0, None); ax.set_xticks([0, 16, 32, 48, 63])
    ax.set_xlabel("frame index (64 frames)"); ax.set_ylabel("tokens selected (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", ncol=2, handlelength=1.4)
    ax.set_title("(a) Video-MME Long: per-frame selection share of OVAL", loc="left")
    B = ["first", "middle", "last"]
    axs = [fig.add_subplot(gs[1, 0]), None]
    axs[1] = fig.add_subplot(gs[1, 1], sharey=axs[0])
    for k, ((m, lab), a) in enumerate(zip(MODELS, axs)):
        for j, (arm, c, dx) in enumerate([("OVAL", BLUE, -0.12), ("FreeKV", ORANGE, 0.12)]):
            r = {x["bucket"]: x for x in ev if x["model"] == m and x["arm"] == arm and x["reference"] == "FullKV-64"}
            y = [float(r[b]["diff"]) for b in B]; lo = [float(r[b]["lo"]) for b in B]; hi = [float(r[b]["hi"]) for b in B]
            xs = [i + dx for i in range(3)]
            a.errorbar(xs, y, yerr=[[v - l for v, l in zip(y, lo)], [h - v for v, h in zip(y, hi)]], fmt="o", ms=3,
                       color=c, ecolor=c, elinewidth=0.9, capsize=1.5, label=arm if k == 0 else None)
            out += [["b", lab, arm, b, v, l, h] for b, v, l, h in zip(B, y, lo, hi)]
        a.axhline(0, color=INK2, lw=0.6)
        a.set_xticks(range(3)); a.set_xticklabels(B); a.set_xlim(-0.5, 2.5)
        a.set_title(lab, fontsize=7.5)
        a.yaxis.grid(True, color=GRID, lw=0.5); a.set_axisbelow(True)
    axs[0].set_ylabel("$\\Delta$ acc. vs FullKV (pp)")
    plt.setp(axs[1].get_yticklabels(), visible=False)
    axs[0].legend(frameon=False, loc="lower left", handletextpad=0.2, borderaxespad=0.1)
    fig.text(0.5, 0.0, "LVBench: position of the annotated evidence in the video (thirds)", ha="center", fontsize=7, color=INK2)
    fig.text(0.0, 0.445, "(b) LVBench: accuracy change by evidence position", fontsize=8)
    os.makedirs(os.path.join(R, "figures"), exist_ok=True)
    fig.savefig(os.path.join(R, "figures", "video_summary.pdf"))
    with open(os.path.join(R, "figures", "video_summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["panel", "model", "arm", "x", "value", "lo", "hi"]); w.writerows(out)


if __name__ == "__main__":
    main()
