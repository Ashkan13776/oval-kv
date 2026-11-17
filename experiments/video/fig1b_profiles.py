"""Fig. 1(b): per-frame selected share on LIBERO-Recall (Qwen2.5-VL, real runs, all 100 sessions).
For first and last questions: mean profile over sessions (64 frames), accuracy, and the mean
gold-task mask. SnapKV pages come from the video's last prefill tokens, before the question
(query-agnostic, run without eviction). Output: fig1b_profiles.json"""
import json, glob, os, numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LR = os.environ.get("LR_ROOT", os.path.expanduser("~/liberorecall"))
src = open("experiments/video/fig1b_extract.py").read().split("cands = []")[0]
exec(src)
out = {}
for name, d in [("oval", ov), ("snapkv", sk)]:
    for t, ix in [("first", 0), ("last", -1)]:
        P, C, M = [], [], []
        for ses in S["sessions"]:
            s2 = ses["sid"]; fi = np.array(json.load(open(os.path.join(LR, f"frames/{s2}_n64/meta.json")))["frame_idx"])
            sp = ses["spans"][ix]; r = d.get(f"{s2}-{t}")
            if r is None: continue
            P.append(prof(r)); C.append(score(r)); M.append(((fi >= sp["start"]) & (fi < sp["end"])).astype(float))
        out[f"{name}_{t}"] = {"profile": np.mean(P, 0).tolist(), "acc": 100 * float(np.mean(C)), "n": len(C), "gold_mask_mean": np.mean(M, 0).tolist()}
json.dump(out, open(os.path.join(REPO, "results/video/fig1b/fig1b_profiles.json"), "w"))
for k, v in out.items(): print(k, "acc %.1f n %d" % (v["acc"], v["n"]))
