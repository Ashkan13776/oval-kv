"""Fig. 1(b): pick a LIBERO-Recall session where two questions with different evidence move OVAL's
selection to different frames (Qwen2.5-VL, real sellog). Gold span = frames of the answer task
(first/last/after/before/present). Lift = mean selected share in gold span / mean share overall.
Rule: among sessions with two correctly answered questions whose gold tasks differ, maximize the smaller
of the two lifts. Also reports the mean lift over all questions (no selection)."""
import json, glob, os, re, numpy as np
R = os.path.expanduser("~/oval-kv/results/revision/C/runs/lr_qwen25vl/oval_n64")
ov = {}
for f in sorted(glob.glob(R + "/shard?of4.jsonl")):
    for l in open(f): r = json.loads(l); ov[r["question_id"]] = r
S = json.load(open(os.path.expanduser("~/liberorecall/sessions.json")))
QA = {}
for l in open(os.path.expanduser("~/liberorecall/video_info.meta.jsonl")):
    v = json.loads(l)
    for q in v["qa"]: QA[q["uid"]] = q
norm = lambda t: re.sub(r"[^a-z ]", "", t.lower()).strip()
def prof(r):
    sl = r["sellog"]; sl = eval(sl) if isinstance(sl, str) else sl
    p = np.array(sl["sel_frame"], float) / np.maximum(np.array(sl["avail_frame"], float), 1)
    return np.repeat(p, 2) if len(p) == 32 else p
best, lifts = [], []
for ses in S["sessions"]:
    sid = ses["sid"]; fi = np.array(json.load(open(os.path.expanduser(f"~/liberorecall/frames/{sid}_n64/meta.json")))["frame_idx"])
    names = [norm(S["tasks"][str(sp["task"])]) for sp in ses["spans"]]
    good = []
    for t in ["first", "last", "after", "before", "present"]:
        qid = f"{sid}-{t}"; q = QA.get(qid); r = ov.get(qid)
        if not q or not r: continue
        opts = dict(re.findall(r"\(([A-D])\) ([^\n]+)", q["question"]))
        ans = norm(opts.get(q["answer"], ""))
        if ans not in names: continue
        k = names.index(ans); sp = ses["spans"][k]; m = (fi >= sp["start"]) & (fi < sp["end"])
        if not m.sum(): continue
        p = prof(r); lift = p[m].mean() / p.mean(); lifts.append(lift)
        if float(r["score"]) > 0.5: good.append((lift, t, k, qid))
    for i in range(len(good)):
        for j in range(i + 1, len(good)):
            if abs(good[i][2] - good[j][2]) >= 3:
                best.append((min(good[i][0], good[j][0]), sid, good[i], good[j]))
best.sort(reverse=True)
print("mean lift over all %d questions: %.2f" % (len(lifts), np.mean(lifts)))
for b in best[:6]: print(b)
sc, sid, g1, g2 = best[0]
fi = json.load(open(os.path.expanduser(f"~/liberorecall/frames/{sid}_n64/meta.json")))["frame_idx"]
ses = next(s for s in S["sessions"] if s["sid"] == sid)
out = {"rule": __doc__, "sid": sid, "frame_idx": fi, "spans": ses["spans"], "mean_lift_all": float(np.mean(lifts)), "n_all": len(lifts),
       "q": [{"qid": g[3], "type": g[1], "task_idx": g[2], "lift": g[0], "question": QA[g[3]]["question"], "answer": QA[g[3]]["answer"],
              "profile": prof(ov[g[3]]).tolist()} for g in (g1, g2)]}
json.dump(out, open(os.path.expanduser("~/oval-kv/results/revision/C/fig1b/fig1b_pair.json"), "w"), indent=1)
