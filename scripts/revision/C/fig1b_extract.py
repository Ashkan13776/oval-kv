"""Fig. 1(b) data: per-frame selection on one LIBERO-Recall session (Qwen2.5-VL, real runs).
OVAL: per question sel/avail per frame (sellog). SnapKV: its pages are fixed from the last 32
prefill tokens before the question, so its kept share is the same for every question of a session.
Session choice (rule, no hand-picking): sessions where OVAL answers both the 'first' and 'last'
questions correctly; among them, the one where OVAL's selected share inside the gold span
(first task for 'first', last task for 'last'), relative to its mean share, is largest on average.
Output: fig1b.json"""
import glob, json, os
import numpy as np
R = os.path.expanduser("~/oval-kv/results/revision/C/runs/lr_qwen25vl")
def load(a):
    out = {}
    for f in sorted(glob.glob(f"{R}/{a}/shard?of4.jsonl")):
        for l in open(f):
            r = json.loads(l); out[r["question_id"]] = r
    return out
ov, sk = load("oval_n64"), load("snapkv_n64")
S = json.load(open(os.path.expanduser("~/liberorecall/sessions.json")))
def prof(r):
    sl = r["sellog"]; sl = eval(sl) if isinstance(sl, str) else sl
    s, a = np.array(sl["sel_frame"], float), np.array(sl["avail_frame"], float)
    p = s / np.maximum(a, 1)
    return np.repeat(p, 2) if len(p) == 32 else p   # Qwen2.5-VL: one value per temporal group of 2 frames
def score(r): return float(r["score"]) > 0.5
cands = []
for ses in S["sessions"]:
    sid = ses["sid"]
    meta = json.load(open(os.path.expanduser(f"~/liberorecall/frames/{sid}_n64/meta.json")))
    fi = np.array(meta["frame_idx"])
    span_of = lambda sp: (fi >= sp["start"]) & (fi < sp["end"])
    qf, ql = ov.get(f"{sid}-first"), ov.get(f"{sid}-last")
    if not (qf and ql and score(qf) and score(ql)): continue
    pf, pl = prof(qf), prof(ql)
    mf, ml = span_of(ses["spans"][0]), span_of(ses["spans"][-1])
    if mf.sum() == 0 or ml.sum() == 0: continue
    lift = 0.5 * (pf[mf].mean() / pf.mean() + pl[ml].mean() / pl.mean())
    cands.append((lift, sid))
cands.sort(reverse=True)
print("candidates", len(cands), "of", len(S["sessions"]), cands[:5])
sid = cands[0][1]
ses = next(s for s in S["sessions"] if s["sid"] == sid)
meta = json.load(open(os.path.expanduser(f"~/liberorecall/frames/{sid}_n64/meta.json")))
qa = {q["uid"]: q for v in map(json.loads, open(os.path.expanduser("~/liberorecall/video_info.meta.jsonl"))) if v["key"] == sid for q in v["qa"]}
out = {"sid": sid, "rule": __doc__, "n_candidates": len(cands), "frame_idx": meta["frame_idx"], "spans": ses["spans"],
       "tasks": {str(sp["task"]): S["tasks"][str(sp["task"])] for sp in ses["spans"]}, "q": {}}
for t in ["present", "absent", "first", "last", "after", "before"]:
    qid = f"{sid}-{t}"
    out["q"][t] = {"question": qa[qid]["question"], "answer": qa[qid]["answer"],
                   "oval": prof(ov[qid]).tolist(), "oval_correct": score(ov[qid]), "oval_pred": ov[qid]["pred"],
                   "snapkv": prof(sk[qid]).tolist(), "snapkv_correct": score(sk[qid]), "snapkv_pred": sk[qid]["pred"]}
# benchmark-wide context: OVAL lift on first/last over all sessions (not just the chosen one)
allf = []
for ses2 in S["sessions"]:
    s2 = ses2["sid"]; fi = np.array(json.load(open(os.path.expanduser(f"~/liberorecall/frames/{s2}_n64/meta.json")))["frame_idx"])
    for t, sp in [("first", ses2["spans"][0]), ("last", ses2["spans"][-1])]:
        r = ov.get(f"{s2}-{t}"); m = (fi >= sp["start"]) & (fi < sp["end"])
        if r and m.sum(): p = prof(r); allf.append(p[m].mean() / p.mean())
out["all_lift_mean"] = float(np.mean(allf)); out["all_lift_n"] = len(allf)
json.dump(out, open(os.path.expanduser("~/oval-kv/results/revision/C/fig1b/fig1b.json"), "w"), indent=1)
print(sid, "all-sessions lift", out["all_lift_mean"], len(allf))
for t, v in out["q"].items(): print(t, "oval", v["oval_correct"], "snapkv", v["snapkv_correct"])
# reference: the same lift for SnapKV fixed pages (chosen before the question)
sk_l = []
for ses2 in S["sessions"]:
    s2 = ses2["sid"]; fi = np.array(json.load(open(os.path.expanduser(f"~/liberorecall/frames/{s2}_n64/meta.json")))["frame_idx"])
    for t, sp in [("first", ses2["spans"][0]), ("last", ses2["spans"][-1])]:
        r = sk.get(f"{s2}-{t}"); m = (fi >= sp["start"]) & (fi < sp["end"])
        if r and m.sum(): p = prof(r); sk_l.append(p[m].mean() / p.mean())
out["snapkv_all_lift_mean"] = float(np.mean(sk_l))
json.dump(out, open(os.path.expanduser("~/oval-kv/results/revision/C/fig1b/fig1b.json"), "w"), indent=1)
print("snapkv all-sessions lift", out["snapkv_all_lift_mean"], len(sk_l))
q = out["q"]; print("snapkv first==last profile:", np.allclose(q["first"]["snapkv"], q["last"]["snapkv"]))
