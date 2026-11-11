"""Fig. 1(b) v2: per-task selected share on LIBERO-Recall (Qwen2.5-VL, real runs, all 100 sessions).
For each session and question, share of task k = mean over the frames of task k of the per-frame
selected share (sellog). Averaged over sessions, for first/last questions, OVAL and SnapKV.
Output: fig1b_tasks.json"""
import json, glob, os, numpy as np
exec(open("scripts/revision/C/fig1b_extract.py").read().split("cands = []")[0])
out = {}
for name, d in [("oval", ov), ("snapkv", sk)]:
    for t in ["first", "last"]:
        T, C = [], []
        for ses in S["sessions"]:
            s2 = ses["sid"]; fi = np.array(json.load(open(os.path.expanduser(f"~/liberorecall/frames/{s2}_n64/meta.json")))["frame_idx"])
            r = d.get(f"{s2}-{t}")
            if r is None: continue
            p = prof(r); row = []
            for sp in ses["spans"]:
                m = (fi >= sp["start"]) & (fi < sp["end"])
                row.append(float(p[m].mean()) if m.sum() else np.nan)
            T.append(row); C.append(score(r))
        out[f"{name}_{t}"] = {"task_share": np.nanmean(np.array(T), 0).tolist(), "acc": 100 * float(np.mean(C)), "n": len(C)}
json.dump(out, open("results/revision/C/fig1b/fig1b_tasks.json", "w"))
for k, v in out.items(): print(k, "acc %.0f" % v["acc"], " ".join("%.3f" % x for x in v["task_share"]))
