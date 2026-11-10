"""Paired difference between two arms (same questions), cluster bootstrap over videos (10k, seed 0),
as summarize_c.py. Used for budget-matched C3 comparisons (e.g. OVAL vs ReKV / InfiniPot-V 2k).

    python scripts/revision/C/pair_arms.py --a RUN_DIR_A --b RUN_DIR_B [--groups]
"""
import argparse
from collections import defaultdict

from summarize_c import boot, load


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--groups", action="store_true", help="also per group field (duration / video type)")
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    qs = sorted(set(A) & set(B))
    keys = [None] + (sorted({A[q].get("group") or A[q]["duration"] for q in qs}) if a.groups else [])
    for k in keys:
        dv = defaultdict(list)
        for q in qs:
            if k is None or (A[q].get("group") or A[q]["duration"]) == k:
                dv[A[q]["videoID"]].append(100 * (A[q]["score"] - B[q]["score"]))
        m, lo, hi = boot(dv)
        print(f"{k or 'overall'}\tn={sum(map(len, dv.values()))}\t{m:+.2f} [{lo:+.2f}, {hi:+.2f}]{'*' if lo > 0 or hi < 0 else ''}")


if __name__ == "__main__":
    main()
