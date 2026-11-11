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
WD = os.environ.get("WD_ROOT", os.path.expanduser("~/libero_watchdo"))
MODELS = {"Qwen2.5-VL": "wd_qwen25vl", "LLaVA-Video": "wd_llava"}
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
    os.makedirs(os.path.join(R, "vla"), exist_ok=True)
    with open(os.path.join(R, "vla", "choices.jsonl"), "w") as f:
        for c in choices:
            f.write(json.dumps(c) + "\n")
    jobs = {f'{c["target"]}|{c["command"]}|{c["init_idx"]}': c for c in choices if c["command"]}
    with open(os.path.join(R, "vla", "jobs.jsonl"), "w") as f:
        for k in sorted(jobs):
            c = jobs[k]; f.write(json.dumps(dict(target=c["target"], command=c["command"], init_idx=c["init_idx"])) + "\n")
    arms = sorted({(c["model"], c["arm"]) for c in choices})
    print(len(choices), "choices over", len(arms), "arms;", len(jobs), "unique rollouts")


if __name__ == "__main__":
    main()
