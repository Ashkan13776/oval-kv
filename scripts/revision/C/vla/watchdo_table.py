"""LIBERO Watch-and-Do results: per planner model x KV arm, the share of instructions where the planner chose the
correct command (recall) and the closed-loop task success of the two-level agent (OpenVLA executes the chosen
command; success = simulator goal check of the target task). References: oracle (always the correct command:
policy ceiling) and random (no memory). Statistics as the video tables: cluster bootstrap over sessions
(10,000 draws, seed 0); paired 95% CIs vs FullKV (N) and vs OVAL on the same instructions. An arm with a
missing rollout is reported as incomplete ("--"), never filled.

    python scripts/revision/C/vla/watchdo_table.py -> results/revision/C/vla/watchdo_{table.md,table.csv,pairs.csv}
"""
import csv, glob, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from summarize_c import boot

R = os.path.join(HERE, "..", "..", "..", "..", "results", "revision", "C", "vla")
ARMS = ["full_n64", "quest_n64", "freekv_n64", "rekv", "infinipot_2k", "snapkv_n64", "oval_n64"]
LAB = {"full_n64": "FullKV (N)", "quest_n64": "Quest", "freekv_n64": "FreeKV", "rekv": "ReKV", "infinipot_2k": "InfiniPot-V 2k",
       "snapkv_n64": "SnapKV (upper bound)", "oval_n64": "OVAL", "oracle": "Oracle command", "random": "Random command"}


def main():
    roll = {}
    for f in glob.glob(os.path.join(R, "rollouts", "shard*.jsonl")):
        for l in open(f):
            if l.endswith("\n"):
                r = json.loads(l); roll[r["key"]] = r["success"]
    ch = defaultdict(dict)
    for l in open(os.path.join(R, "choices.jsonl")):
        c = json.loads(l); ch[c["model"], c["arm"]][c["uid"]] = c
    def score(c):
        """(recall, success) for one choice; None if its rollout has not run."""
        rec = float(c["command"] == c["target"])
        if c["command"] is None:
            return rec, 0.0
        k = f'{c["target"]}|{c["command"]}|{c["init_idx"]}'
        return (rec, float(roll[k])) if k in roll else None
    S = {a: {u: score(c) for u, c in d.items()} for a, d in ch.items()}
    rows, pairs, md = [], [], ["| Planner | Arm | n | Recall % [95% CI] | Success % [95% CI] | Success - FullKV | Success - OVAL |", "|---|---|---|---|---|---|---|"]
    order = [("-", "oracle"), ("-", "random")] + [(m, a) for m in ("Qwen2.5-VL", "LLaVA-Video") for a in ARMS]
    for m, a in order:
        s = S.get((m, a))
        if s is None or any(v is None for v in s.values()):
            n = 0 if s is None else sum(v is not None for v in s.values())
            md.append(f"| {m} | {LAB[a]} | {n}/240 | incomplete | incomplete | | |"); continue
        cell = {}
        for i, name in ((0, "recall"), (1, "success")):
            dv = defaultdict(list)
            for u, v in s.items():
                dv[u.split("-")[0]].append(100 * v[i])
            cell[name] = boot(dv)
        diffs = {}
        for ref in ("full_n64", "oval_n64"):
            r = S.get((m, ref))
            if m == "-" or a == ref or r is None or any(v is None for v in r.values()):
                continue
            dd = defaultdict(list)
            for u in s:
                dd[u.split("-")[0]].append(100 * (s[u][1] - r[u][1]))
            diffs[ref] = boot(dd)
            pairs.append([m, f"{a} - {ref}", *(f"{x:.3f}" for x in diffs[ref])])
        f = lambda t: f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"
        sig = lambda t: "*" if t[1] > 0 or t[2] < 0 else ""
        md.append(f"| {m} | {LAB[a]} | {len(s)} | {f(cell['recall'])} | {f(cell['success'])} | "
                  + " | ".join(f"{diffs[r][0]:+.2f}{sig(diffs[r])}" if r in diffs else "" for r in ("full_n64", "oval_n64")) + " |")
        rows.append([m, a, len(s), *(f"{x:.3f}" for x in cell["recall"]), *(f"{x:.3f}" for x in cell["success"])])
    open(os.path.join(R, "watchdo_table.md"), "w").write("\n".join(md) + "\n")
    with open(os.path.join(R, "watchdo_table.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["model", "arm", "n", "recall", "rlo", "rhi", "success", "slo", "shi"]); w.writerows(rows)
    with open(os.path.join(R, "watchdo_pairs.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["model", "pair", "diff", "lo", "hi"]); w.writerows(pairs)
    print("\n".join(md)); print(len(roll), "rollouts")


if __name__ == "__main__":
    main()
