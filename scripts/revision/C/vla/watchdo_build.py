"""LIBERO Watch-and-Do: closed-loop robot memory task for a two-level agent (VLM planner + OpenVLA policy).

Memory phase: a session video of K = 6 DISTINCT LIBERO-Goal tasks (one random demonstration each, from
lerobot/libero; all 10 Goal tasks share one kitchen scene), played back to back (H.264, 10 fps).
Instruction (one per question): repeat a task from the video, identified only by its position in the history:
  first   repeat the task performed first          last    ... performed last
  after   repeat the task performed right after X  before  ... right before X     (X named by its instruction)
The planner (Qwen2.5-VL / LLaVA-Video, cached query-agnostically under each KV method) picks the command from
4 options (the correct task + 3 other session tasks, never X); OpenVLA then executes the chosen command in the
LIBERO simulator from initial state `init_idx` of the TARGET task, and success is the simulator's goal check
of the target task (watchdo_rollout.py).

Output in LVBench format (LVB_ROOT=$WD_ROOT) so every LVBench runner works unchanged:
  $WD_ROOT/videos/<sid>.mp4, video_info.meta.jsonl, sessions.json (episode spans, targets, init states).

    python scripts/revision/C/vla/watchdo_build.py --sessions 60
"""
import argparse, json, os, random, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import liberorecall_build as LR

WD = os.environ.get("WD_ROOT", os.path.expanduser("~/libero_watchdo"))
GOAL = list(range(10, 20))   # lerobot/libero task indices of the LIBERO-Goal suite (names = simulator task languages)
N_INIT = 50                  # LIBERO initial states per task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=60)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    e, names = LR.episodes()
    by_task = {t: sorted(e.index[e.task_index == t].tolist()) for t in GOAL}
    os.makedirs(os.path.join(WD, "videos"), exist_ok=True)
    meta, sessions = [], []
    for s in range(a.sessions):
        sid = f"wd{s:03d}"
        tasks = rng.sample(GOAL, a.k)
        eps = [rng.choice(by_task[t]) for t in tasks]
        spans, frames, cur = [], [], 0
        for t, ep in zip(tasks, eps):
            fr = LR.decode(e.loc[ep])
            spans.append(dict(task=t, episode=int(ep), start=cur, end=cur + len(fr))); cur += len(fr)
            frames.append(fr)
        path = os.path.join(WD, "videos", sid + ".mp4")
        if not os.path.exists(path):
            LR.encode(np.concatenate(frames), path)
        N = lambda t: names[t]
        qa, targets = [], {}
        pre = "The video shows a robot performing several tasks in a kitchen, one after another. "
        post = " Which command should be sent to the robot?"

        def add(qt, q, correct, wrong):
            o, ans = LR.mcq(rng, N(correct), [N(w) for w in wrong])
            uid = f"{sid}-{qt}"
            qa.append(dict(uid=uid, question=pre + q + post + "\n" + o, answer=ans, question_type=[qt], time_reference=""))
            targets[uid] = dict(target=N(correct), options=dict(zip("ABCD", [l[4:] for l in o.split("\n")])),
                                init_idx=rng.randrange(N_INIT))

        add("first", "Now repeat the task the robot performed first.", tasks[0], rng.sample(tasks[1:], 3))
        add("last", "Now repeat the task the robot performed last.", tasks[-1], rng.sample(tasks[:-1], 3))
        i = rng.randrange(a.k - 1)
        add("after", f'Now repeat the task the robot performed right after the task "{N(tasks[i])}".', tasks[i + 1],
            rng.sample([t for j, t in enumerate(tasks) if j not in (i, i + 1)], 3))
        i = rng.randrange(1, a.k)
        add("before", f'Now repeat the task the robot performed right before the task "{N(tasks[i])}".', tasks[i - 1],
            rng.sample([t for j, t in enumerate(tasks) if j not in (i, i - 1)], 3))
        meta.append(dict(key=sid, type="libero_watchdo", qa=qa, video_info=dict(duration_minutes=cur / 600, frames=cur, fps=10)))
        sessions.append(dict(sid=sid, spans=spans, targets=targets))
        print(sid, cur, "frames", flush=True)
    with open(os.path.join(WD, "video_info.meta.jsonl"), "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    json.dump(dict(seed=a.seed, k=a.k, sessions=sessions, tasks={t: names[t] for t in GOAL}),
              open(os.path.join(WD, "sessions.json"), "w"), indent=1)
    ans = [q["answer"] for m in meta for q in m["qa"]]
    print(len(meta), "sessions", len(ans), "instructions; answer letters", {L: ans.count(L) for L in "ABCD"})


if __name__ == "__main__":
    main()
