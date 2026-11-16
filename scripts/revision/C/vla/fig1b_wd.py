"""Fig. 1(b) data: one LIBERO Watch-and-Do episode (Qwen2.5-VL planner with OVAL, OpenVLA executes), real runs.
Episode choice (rule, no hand-picking): first/last episodes where the OVAL planner picks the correct command
and the OpenVLA rollout succeeds; among them, the one where OVALs read share inside the target tasks span,
relative to its mean read share over the video, is largest. Read share per frame = selected / available visual
tokens of that frame, averaged over decoding steps, layers and KV heads (sellog). Output: fig1b_wd.json"""
import glob, json, os
import numpy as np
H = os.path.expanduser("~")
RUN = f"{H}/oval-kv/results/revision/C/runs/wd2_qwen25vl"
WD = f"{H}/libero_watchdo2"
def load(a):
    out = {}
    for f in sorted(glob.glob(f"{RUN}/{a}/shard*.jsonl")):
        if "_timing" in f: continue
        for l in open(f):
            if l.endswith("\n"): r = json.loads(l); out[r["question_id"]] = r
    return out
arms = {a: load(a) for a in ["oval_n64", "full_n64", "freekv_n64", "infinipot_2k", "quest_n64", "snapkv_n64", "rekv"]}
roll = {}
for f in glob.glob(f"{H}/oval-kv/results/revision/C/vla/rollouts/shard*.jsonl"):
    for l in open(f):
        if l.endswith("\n"): r = json.loads(l); roll[r["key"]] = r["success"]
S = json.load(open(f"{WD}/sessions.json"))
qa = {q["uid"]: q for v in map(json.loads, open(f"{WD}/video_info.meta.jsonl")) for q in v["qa"]}
def prof(r):
    sl = r["sellog"]; sl = eval(sl) if isinstance(sl, str) else sl
    s, a = np.array(sl["sel_frame"], float), np.array(sl["avail_frame"], float)
    p = s / np.maximum(a, 1)
    return np.repeat(p, 2) if len(p) == 32 else p
def outcome(sid, t, arm):
    r = arms[arm].get(f"{sid}-{t}"); tg = S_by[sid]["targets"][f"{sid}-{t}"]
    cmd = tg["options"][r["pred"]].lower() if r and r["pred"] in tg["options"] else None
    ok = roll.get("|".join([tg["target"], cmd, str(tg["init_idx"])])) if cmd else None
    return dict(pred=r["pred"] if r else None, command=cmd, correct=bool(r and float(r["score"]) > 0.5), success=ok)
S_by = {s["sid"]: s for s in S["sessions"]}
cands, lifts = [], []
for ses in S["sessions"]:
    sid = ses["sid"]; fi = np.array(json.load(open(f"{WD}/frames/{sid}_n64/meta.json"))["frame_idx"])
    for t, sp in [("first", ses["spans"][0]), ("last", ses["spans"][-1])]:
        r = arms["oval_n64"].get(f"{sid}-{t}")
        m = (fi >= sp["start"]) & (fi < sp["end"])
        if not r or not m.sum(): continue
        p = prof(r); lift = p[m].mean() / p.mean(); lifts.append(lift)
        o = outcome(sid, t, "oval_n64")
        if o["correct"] and o["success"]: cands.append((lift, sid, t))
cands.sort(reverse=True)
print("candidates", len(cands), cands[:5]); lift, sid, t = cands[0]
ses = S_by[sid]; meta = json.load(open(f"{WD}/frames/{sid}_n64/meta.json")); fi = np.array(meta["frame_idx"])
p = prof(arms["oval_n64"][f"{sid}-{t}"])
segs = []
for sp in ses["spans"]:
    m = (fi >= sp["start"]) & (fi < sp["end"]); idx = np.where(m)[0]
    segs.append(dict(task=S["tasks"][str(sp["task"])], start=sp["start"], end=sp["end"], frames=idx.tolist(),
                     read_share=float(p[m].mean()), read_share_rel=float(p[m].mean() / p.mean())))
out = dict(rule=__doc__, sid=sid, type=t, lift=float(lift), n_candidates=len(cands), all_lift_mean=float(np.mean(lifts)),
           all_lift_n=len(lifts), question=qa[f"{sid}-{t}"]["question"], answer=qa[f"{sid}-{t}"]["answer"],
           target=ses["targets"][f"{sid}-{t}"], frame_idx=meta["frame_idx"], frame_time=meta["frame_time"],
           oval_frame_share=p.tolist(), segments=segs, mean_share=float(p.mean()),
           arms={a: outcome(sid, t, a) for a in arms})
json.dump(out, open(f"{H}/oval-kv/results/revision/C/fig1b_wd/fig1b_wd.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ["sid", "type", "lift", "all_lift_mean", "answer", "mean_share"]}))
for s in segs: print(s["task"], round(s["read_share"], 3), round(s["read_share_rel"], 2), s["frames"][:3], len(s["frames"]))
print(json.dumps(out["arms"], indent=0)); print(out["target"])
