"""C5 on LVBench: does the selection's position bias cost accuracy? (no GPU)

For every LVBench question: midpoint of `time_reference` relative to the annotated
`duration_minutes` -> first / middle / last third. Accuracy of FullKV-64, FreeKV and OVAL per
bucket, 95% CI from a bootstrap over QUESTIONS (as the brief asks), 10,000 draws, seed 0; and the
paired difference OVAL - FullKV and FreeKV - FullKV per bucket (same questions).

Span coverage: how many of the 64 sampled frames (frame_idx / fps from the frame cache meta of each
model) fall inside the question's span [start, end]; accuracy restricted to questions whose span
contains >= 1 sampled frame. Questions whose span is annotated with 'None' (2 of 1,549) are
excluded from C5. Spans whose end exceeds the annotated duration, or whose end is before
their start, are counted and kept in the bucket analysis (midpoint clipped to [0, 1]).

    python scripts/revision/C/c5_evidence.py
-> results/revision/C/tables/c5_evidence_{buckets,paired,coverage}.csv, results/revision/C/c5_evidence.md,
   results/revision/C/figures/c5_evidence_buckets.{pdf,csv}
"""
import csv, json, os, sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_c import load

ROOT = os.path.join(HERE, "..", "..", "..")
RES = os.path.join(ROOT, "results", "revision", "C")
LVB = os.environ.get("LVB_ROOT", os.path.expanduser("~/lvbench"))
MODELS = [("LLaVA-Video-7B", "c2_lvbench_llava", "frames"), ("Qwen2.5-VL-7B", "c2_lvbench_qwen25vl", "frames_qwen")]
ARMS = [("full_n64", "FullKV-64"), ("freekv_n64", "FreeKV"), ("oval_n64", "OVAL")]
BUCKETS = ["first", "middle", "last"]
T = 1549


def secs(s):
    p = [float(x) for x in s.strip().split(":")]
    v = 0.0
    for x in p:
        v = 60 * v + x
    return v


def span(tr):
    a, b = tr.split("-")
    return secs(a), secs(b)


def boot_q(x, n=10000, seed=0):
    x = np.asarray(x, float)
    idx = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return x.mean(), np.quantile(m, 0.025), np.quantile(m, 0.975)


def main():
    os.makedirs(os.path.join(RES, "tables"), exist_ok=True); os.makedirs(os.path.join(RES, "figures"), exist_ok=True)
    L = ["# C5: evidence position on LVBench", "",
         "Bucket = midpoint of `time_reference` / (60 x `duration_minutes`), thirds. 95% CIs: bootstrap over "
         "questions (10,000 draws). Paired: same questions.", ""]
    brows, prows, crows, fig = [], [], [], {}
    for model, rdir, fdir in MODELS:
        runs = {a: load(os.path.join(RES, "runs", rdir, a)) for a, _ in ARMS}
        runs = {a: r for a, r in runs.items() if r}
        if "full_n64" not in runs:
            L += [f"## {model}: not run yet", ""]; continue
        common = sorted(set.intersection(*[set(r) for r in runs.values()]))
        status = ", ".join(f"{lab} {len(runs[a])}/{T}{'' if len(runs[a]) >= T else ' INCOMPLETE'}"
                           for a, lab in ARMS if a in runs)
        ref = runs["full_n64"]
        bucket, cover, odd = {}, {}, Counter()
        unparsed = [q for q in common if "None" in ref[q]["time_reference"]]
        common = [q for q in common if q not in unparsed]   # no usable span ("None" in the annotation)
        for q in common:
            r = ref[q]; s, e = span(r["time_reference"]); D = 60 * r["duration_minutes"]
            if e < s: odd["end before start"] += 1
            if e > D: odd["end after duration"] += 1
            mid = min(max((s + e) / 2 / D, 0.0), 1.0)
            bucket[q] = BUCKETS[min(int(mid * 3), 2)]
            meta = json.load(open(os.path.join(LVB, fdir, f"{r['videoID']}_n64", "meta.json")))
            ft = np.asarray(meta["frame_idx"]) / meta["fps"]
            cover[q] = int(((ft >= s) & (ft <= e)).sum())
        L += [f"## {model}", "", f"Runs: {status}. Scored on {len(common)} common questions. "
              f"Excluded (span annotated as None): {len(unparsed)} questions ({', '.join(unparsed)}). "
              f"Span anomalies kept: {dict(odd) or 'none'}.", "",
              "| arm | " + " | ".join(f"{b} (n={sum(v == b for v in bucket.values())})" for b in BUCKETS)
              + " | all |", "|---|---|---|---|---|"]
        for a, lab in ARMS:
            if a not in runs:
                continue
            cells = []
            for b in BUCKETS + ["all"]:
                x = [runs[a][q]["score"] * 100 for q in common if b == "all" or bucket[q] == b]
                m, lo, hi = boot_q(x); cells.append(f"{m:.2f} [{lo:.2f}, {hi:.2f}]")
                brows.append([model, lab, b, len(x), f"{m:.3f}", f"{lo:.3f}", f"{hi:.3f}", len(runs[a]) >= T])
                fig[model, lab, b] = (m, lo, hi)
            L.append(f"| {lab} | " + " | ".join(cells) + " |")
        L += ["", "Paired difference vs FullKV-64 (pp); * = CI excludes 0", "", "| arm | first | middle | last | all |",
              "|---|---|---|---|---|"]
        for a, lab in ARMS[1:]:
            if a not in runs:
                continue
            cells = []
            for b in BUCKETS + ["all"]:
                x = [100 * (runs[a][q]["score"] - ref[q]["score"]) for q in common if b == "all" or bucket[q] == b]
                m, lo, hi = boot_q(x); cells.append(f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]{'*' if lo > 0 or hi < 0 else ''}")
                prows.append([model, lab, "FullKV-64", b, len(x), f"{m:.3f}", f"{lo:.3f}", f"{hi:.3f}"])
            L.append(f"| {lab} | " + " | ".join(cells) + " |")
        cnt = Counter(cover.values()); inside = [q for q in common if cover[q] > 0]
        L += ["", f"Sampled frames inside the span: 0 frames for {cnt[0]} of {len(common)} questions "
              f"({100 * cnt[0] / len(common):.2f}%); median {int(np.median(list(cover.values())))}, "
              f"max {max(cover.values())}. Distribution: "
              + ", ".join(f"{k}: {cnt[k]}" for k in sorted(cnt)[:8]) + (" ..." if len(cnt) > 8 else ""), "",
              f"Accuracy on the {len(inside)} questions whose span contains >= 1 sampled frame "
              f"(vs the {len(common) - len(inside)} with none):", "",
              "| arm | span has a frame | span has no frame |", "|---|---|---|"]
        for a, lab in ARMS:
            if a not in runs:
                continue
            c1 = boot_q([runs[a][q]["score"] * 100 for q in inside])
            c0 = boot_q([runs[a][q]["score"] * 100 for q in common if cover[q] == 0])
            L.append(f"| {lab} | {c1[0]:.2f} [{c1[1]:.2f}, {c1[2]:.2f}] | {c0[0]:.2f} [{c0[1]:.2f}, {c0[2]:.2f}] |")
            crows += [[model, lab, "span>=1 frame", len(inside), *[f"{v:.3f}" for v in c1]],
                      [model, lab, "span 0 frames", len(common) - len(inside), *[f"{v:.3f}" for v in c0]]]
        L.append("")
    w = lambda n, h, rows: csv.writer(open(os.path.join(RES, "tables", n), "w", newline="")).writerows([h] + rows)
    w("c5_evidence_buckets.csv", ["model", "arm", "bucket", "n", "acc", "lo", "hi", "complete"], brows)
    w("c5_evidence_paired.csv", ["model", "arm", "reference", "bucket", "n", "diff", "lo", "hi"], prows)
    w("c5_evidence_coverage.csv", ["model", "arm", "subset", "n", "acc", "lo", "hi"], crows)
    open(os.path.join(RES, "c5_evidence.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    if fig:
        import figstyle as S
        import matplotlib.pyplot as plt
        models = [m for m, _, _ in MODELS if any(k[0] == m for k in fig)]
        f, axs = plt.subplots(1, len(models), figsize=(S.COL * len(models) / 1.2, 1.9), squeeze=False, sharey=True)
        col = {"FullKV-64": S.INK2, "FreeKV": S.ORANGE, "OVAL": S.BLUE}
        for ax, m in zip(axs[0], models):
            for i, (_, lab) in enumerate(ARMS):
                pts = [fig.get((m, lab, b)) for b in BUCKETS]
                if None in pts:
                    continue
                x = np.arange(3) + (i - 1) * 0.18
                y = [p[0] for p in pts]
                ax.errorbar(x, y, yerr=[[p[0] - p[1] for p in pts], [p[2] - p[0] for p in pts]], fmt="o", ms=3,
                            color=col[lab], capsize=1.5, lw=0.8, label=lab)
            ax.set_xticks(range(3)); ax.set_xticklabels(["first third", "middle third", "last third"])
            ax.set_title(m); ax.grid(axis="y", color=S.GRID, lw=0.5)
        axs[0][0].set_ylabel("LVBench accuracy (%)"); axs[0][-1].legend(frameon=False)
        f.savefig(os.path.join(RES, "figures", "c5_evidence_buckets.pdf"))
        w("../figures/c5_evidence_buckets.csv", ["model", "arm", "bucket", "n", "acc", "lo", "hi", "complete"], brows)


if __name__ == "__main__":
    main()
