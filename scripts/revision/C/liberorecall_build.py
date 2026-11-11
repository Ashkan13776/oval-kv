"""LIBERO-Recall: robot-episode memory QA built from LIBERO demonstrations (lerobot/libero, 40 tasks,
1,693 episodes, third-person camera, 256x256, 10 fps). No human labels: every answer follows from the
dataset's own task labels and episode order.

Session = K = 8 episodes of 8 DISTINCT tasks (drawn from all 40), one random demonstration each, played back
to back -> one mp4 (H.264, 10 fps, ~2 min). Six four-way questions per session (one per type):
  present  Which task did the robot perform?            1 session task + 3 tasks NOT in the session
  absent   Which task did the robot NOT perform?        1 non-session task + 3 session tasks
  first    Which task did the robot perform first?      4 session tasks (incl. the first)
  last     ... last?                                     4 session tasks (incl. the last)
  after    ... right after "<X>"?                         successor of X + 3 other session tasks (not X)
  before   ... right before "<X>"?                        predecessor of X + 3 other session tasks (not X)
Distractors that are not in the session are drawn from the SAME LIBERO suite as the correct option where
possible (same scene and objects), so they cannot be rejected from the scene alone. Option order is a
random permutation (answer letters ~uniform). Everything is seeded (seed 0).

Output in LVBench's format, so every LVBench runner works unchanged with LVB_ROOT=$LR_ROOT:
  $LR_ROOT/videos/<sid>.mp4, $LR_ROOT/video_info.meta.jsonl, $LR_ROOT/sessions.json (episode spans).

    python scripts/revision/C/liberorecall_build.py --sessions 100
"""
import argparse, glob, json, os, random, subprocess

import numpy as np
import pandas as pd

LR = os.environ.get("LR_ROOT", os.path.expanduser("~/liberorecall"))
SRC = os.path.join(LR, "src")
CAM = "observation.images.image"
SUITE = {**{i: "libero_10" for i in range(10)}, **{i: "libero_goal" for i in range(10, 20)},
         **{i: "libero_object" for i in range(20, 30)}, **{i: "libero_spatial" for i in range(30, 40)}}
POST = "\nAnswer the question with the option letter"


def episodes():
    e = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(SRC, "meta/episodes/**/*.parquet"), recursive=True)))
    d = pd.concat(pd.read_parquet(f, columns=["episode_index", "task_index"])
                  for f in sorted(glob.glob(os.path.join(SRC, "data/**/*.parquet"), recursive=True)))
    t = d.groupby("episode_index").task_index.agg(["min", "max"])
    assert (t["min"] == t["max"]).all()
    e = e.set_index("episode_index").join(t["min"].rename("task_index"))
    tasks = pd.read_parquet(os.path.join(SRC, "meta/tasks.parquet"))
    names = {int(i): s for s, i in zip(tasks.index, tasks.task_index)}
    return e, names


def decode(row):
    f = os.path.join(SRC, "videos", CAM, f"chunk-{int(row[f'videos/{CAM}/chunk_index']):03d}",
                     f"file-{int(row[f'videos/{CAM}/file_index']):03d}.mp4")
    t0 = float(row[f"videos/{CAM}/from_timestamp"])
    n = int(row["length"])
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-i", f, "-frames:v", str(n),
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    a = np.frombuffer(out, np.uint8).reshape(-1, 256, 256, 3)
    assert abs(len(a) - n) <= 1, (len(a), n)
    return a


def encode(frames, path):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "256x256",
                          "-r", "10", "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", path],
                         stdin=subprocess.PIPE)
    p.stdin.write(np.ascontiguousarray(frames).tobytes()); p.stdin.close()
    assert p.wait() == 0


def mcq(rng, correct, wrong):
    opts = [correct] + wrong
    rng.shuffle(opts)
    L = "ABCD"
    return "\n".join(f"({L[i]}) {o[0].upper() + o[1:]}" for i, o in enumerate(opts)), L[opts.index(correct)]


def same_suite_absent(rng, task, absent, k):
    s = [t for t in absent if SUITE[t] == SUITE[task]]
    rest = [t for t in absent if SUITE[t] != SUITE[task]]
    rng.shuffle(s); rng.shuffle(rest)
    return (s + rest)[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=100)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    e, names = episodes()
    by_task = {t: sorted(e.index[e.task_index == t].tolist()) for t in range(40)}
    os.makedirs(os.path.join(LR, "videos"), exist_ok=True)
    meta, sessions = [], []
    for s in range(a.sessions):
        sid = f"lr{s:03d}"
        tasks = rng.sample(range(40), a.k)
        eps = [rng.choice(by_task[t]) for t in tasks]
        absent = [t for t in range(40) if t not in tasks]
        spans, frames, cur = [], [], 0
        for t, ep in zip(tasks, eps):
            fr = decode(e.loc[ep])
            spans.append(dict(task=t, episode=int(ep), start=cur, end=cur + len(fr))); cur += len(fr)
            frames.append(fr)
        path = os.path.join(LR, "videos", sid + ".mp4")
        if not os.path.exists(path):
            encode(np.concatenate(frames), path)
        N = lambda t: names[t]
        qa = []

        def add(qt, q, correct, wrong):
            o, ans = mcq(rng, N(correct), [N(w) for w in wrong])
            qa.append(dict(uid=f"{sid}-{qt}", question=q + "\n" + o, answer=ans, question_type=[qt], time_reference=""))

        x = rng.choice(tasks)
        add("present", "Which of the following tasks did the robot perform in the video?", x,
            same_suite_absent(rng, x, absent, 3))
        y = same_suite_absent(rng, rng.choice(tasks), absent, 1)[0]
        add("absent", "Which of the following tasks did the robot NOT perform in the video?", y, rng.sample(tasks, 3))
        add("first", "Which of the following tasks did the robot perform first?", tasks[0], rng.sample(tasks[1:], 3))
        add("last", "Which of the following tasks did the robot perform last?", tasks[-1], rng.sample(tasks[:-1], 3))
        i = rng.randrange(a.k - 1)
        add("after", f'Which task did the robot perform right after the task "{N(tasks[i])}"?', tasks[i + 1],
            rng.sample([t for j, t in enumerate(tasks) if j not in (i, i + 1)], 3))
        i = rng.randrange(1, a.k)
        add("before", f'Which task did the robot perform right before the task "{N(tasks[i])}"?', tasks[i - 1],
            rng.sample([t for j, t in enumerate(tasks) if j not in (i, i - 1)], 3))
        meta.append(dict(key=sid, type="libero", qa=qa, video_info=dict(duration_minutes=cur / 10 / 60, frames=cur, fps=10)))
        sessions.append(dict(sid=sid, spans=spans))
        print(sid, cur, "frames", flush=True)
    with open(os.path.join(LR, "video_info.meta.jsonl"), "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    json.dump(dict(seed=a.seed, k=a.k, sessions=sessions, tasks=names), open(os.path.join(LR, "sessions.json"), "w"), indent=1)
    ans = [q["answer"] for m in meta for q in m["qa"]]
    print(len(meta), "sessions", len(ans), "questions; answer letters", {L: ans.count(L) for L in "ABCD"})


if __name__ == "__main__":
    main()
