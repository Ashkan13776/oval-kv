"""LIBERO Watch-and-Do: planner answers -> rollout jobs (and the join used by watchdo_table.py).

For every planner arm (run dir with 240 answered instructions) and two references
  oracle  the correct command (policy ceiling: OpenVLA success on the target task from the same initial state)
  random  a uniformly random option (seed 0; no memory)
each instruction becomes one job (command = chosen option, target, init_idx). Identical jobs are run once
(watchdo_rollout.py de-duplicates by key): OpenVLA is greedy and the rollout re-seeds, so the same
(command, target, initial state) gives the same rollout whichever arm asked for it.
An unparseable planner answer (pred not in A-D) is a failure without a rollout.

    python scripts/revision/C/vla/watchdo_jobs.py  ->  results/revision/C/vla/jobs.jsonl, choices.jsonl
"""
import glob, json, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "..", "..", "..", "results", "revision", "C")
V = os.environ.get("WD_VERSION", "v2")   # v1: ~/libero_watchdo, runs wd_*; v2: ~/libero_watchdo2, runs wd2_*
WD = os.path.expanduser({"v1": "~/libero_watchdo", "v2": "~/libero_watchdo2"}[V])
TAG = {"v1": "wd", "v2": "wd2"}[V]
MODELS = {"Qwen2.5-VL": f"{TAG}_qwen25vl", "LLaVA-Video": f"{TAG}_llava"}
OUT = os.path.join(R, "vla", V)          # choices.jsonl, jobs.jsonl; rollouts are shared in vla/rollouts
TOTAL = 240


def targets():
    """{uid: target spec}; option texts (capitalized for display) mapped back to the exact LIBERO instruction,
    which is what OpenVLA was fine-tuned on."""
    s = json.load(open(os.path.join(WD, "sessions.json")))
    canon = {n.lower(): n for n in s["tasks"].values()}
    T = {u: t for x in s["sessions"] for u, t in x["targets"].items()}
    for t in T.values():
        t["options"] = {L: canon[o.lower()] for L, o in t["options"].items()}
        assert t["target"] in t["options"].values()
    return T


def planner(model_dir):
    """{arm: {uid: pred}} for complete arms (all TOTAL instructions)."""
    out = {}
    for d in sorted(glob.glob(os.path.join(R, "runs", model_dir, "*"))):
        fs = [f for f in glob.glob(os.path.join(d, "shard*.jsonl")) if "_pilot" not in f and "_timing" not in f]
        recs = {}
        for f in fs:
            for l in open(f):
                if l.endswith("\n"):
                    r = json.loads(l); recs[r["question_id"]] = r["pred"]
        if len(recs) >= TOTAL:
            out[os.path.basename(d)] = recs
    return out


def main():
    T = targets()
    assert len(T) == TOTAL
    rng = random.Random(0)
    rand = {u: rng.choice("ABCD") for u in sorted(T)}
    choices = []
    for mlab, md in MODELS.items():
        for arm, preds in planner(md).items():
            for u, t in T.items():
                p = preds.get(u)
                choices.append(dict(model=mlab, arm=arm, uid=u, pred=p, command=t["options"].get(p), target=t["target"],
                                    init_idx=t["init_idx"]))
    for u, t in T.items():
        choices.append(dict(model="-", arm="oracle", uid=u, pred=None, command=t["target"], target=t["target"], init_idx=t["init_idx"]))
        choices.append(dict(model="-", arm="random", uid=u, pred=rand[u], command=t["options"][rand[u]], target=t["target"],
                            init_idx=t["init_idx"]))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "choices.jsonl"), "w") as f:
        for c in choices:
            f.write(json.dumps(c) + "\n")
    jobs = {f'{c["target"]}|{c["command"]}|{c["init_idx"]}': c for c in choices if c["command"]}
    with open(os.path.join(OUT, "jobs.jsonl"), "w") as f:
        for k in sorted(jobs):
            c = jobs[k]; f.write(json.dumps(dict(target=c["target"], command=c["command"], init_idx=c["init_idx"])) + "\n")
    arms = sorted({(c["model"], c["arm"]) for c in choices})
    print(len(choices), "choices over", len(arms), "arms;", len(jobs), "unique rollouts")


if __name__ == "__main__":
    main()
