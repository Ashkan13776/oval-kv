"""LIBERO Watch-and-Do: execute commands with OpenVLA in the LIBERO simulator (env ~/vla, see ~/vla/env.sh).

Each job is (command, target, init_idx): OpenVLA-7B fine-tuned on LIBERO-Goal receives `command` as its
instruction and is rolled out in the environment of the TARGET task (all LIBERO-Goal tasks share one scene) from
the target's LIBERO initial state `init_idx`; success = the simulator's goal check of the target task.
The loop is the official OpenVLA LIBERO evaluation (experiments/robot/libero/run_libero_eval.py) unchanged:
center crop, 10 no-op steps, at most 300 policy steps, binarized and sign-flipped gripper, seed 7; the seed is
reset before every rollout so a rollout does not depend on which jobs ran before it on the same worker.

    python scripts/revision/C/vla/watchdo_rollout.py --jobs results/revision/C/vla/jobs.jsonl --shard 0 --nshards 4
      -> results/revision/C/vla/rollouts/shard{i}of{n}.jsonl (resumable: keys done by any shard file are skipped; re-shard freely)
"""
import argparse, glob, json, os, sys, time

sys.path.insert(0, os.path.expanduser("~/vla/openvla"))
import numpy as np
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import (get_libero_dummy_action, get_libero_env, get_libero_image,
                                                   quat2axisangle)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (get_action, get_image_resize_size, get_model, invert_gripper_action,
                                           normalize_gripper_action, set_seed_everywhere)

CKPT = "openvla/openvla-7b-finetuned-libero-goal"
MAX_STEPS, WAIT, SEED = 300, 10, 7


class Cfg:  # the fields of run_libero_eval.GenerateConfig that the helpers read
    model_family = "openvla"; pretrained_checkpoint = CKPT; load_in_8bit = False; load_in_4bit = False
    center_crop = True; unnorm_key = "libero_goal"


def key(j):
    return f'{j["target"]}|{j["command"]}|{j["init_idx"]}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default="results/revision/C/vla/rollouts")
    a = ap.parse_args()
    jobs = [json.loads(l) for l in open(a.jobs)]
    os.makedirs(a.out, exist_ok=True)
    outp = os.path.join(a.out, f"shard{a.shard}of{a.nshards}.jsonl")
    done = {json.loads(l)["key"] for f in glob.glob(os.path.join(a.out, "shard*.jsonl")) for l in open(f) if l.endswith("\n")}
    jobs = sorted({key(j): j for j in jobs if key(j) not in done}.values(), key=key)[a.shard::a.nshards]
    print(len(jobs), "jobs to run", flush=True)
    if not jobs:
        return
    cfg = Cfg()
    set_seed_everywhere(SEED)
    model = get_model(cfg)
    if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key += "_no_noops"
    assert cfg.unnorm_key in model.norm_stats
    processor = get_processor(cfg)
    resize = get_image_resize_size(cfg)
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    tid = {suite.get_task(i).language: i for i in range(suite.n_tasks)}
    envs = {}
    for j in jobs:
        t = tid[j["target"]]
        assert j["command"] in tid, j["command"]
        if t not in envs:
            envs[t] = get_libero_env(suite.get_task(t), "openvla", resolution=256)[0]
        env = envs[t]
        set_seed_everywhere(SEED)
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(t)[j["init_idx"]])
        t0, step, done, err = time.time(), 0, False, None
        while step < MAX_STEPS + WAIT:
            try:
                if step < WAIT:
                    obs, _, done, _ = env.step(get_libero_dummy_action("openvla")); step += 1; continue
                img = get_libero_image(obs, resize)
                o = {"full_image": img, "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                                                  obs["robot0_gripper_qpos"]))}
                act = get_action(cfg, model, o, j["command"], processor=processor)
                act = invert_gripper_action(normalize_gripper_action(act, binarize=True))
                obs, _, done, _ = env.step(act.tolist())
                if done:
                    break
                step += 1
            except Exception as e:  # as the official loop: an exception ends the episode as a failure
                err = repr(e); break
        rec = dict(key=key(j), target=j["target"], command=j["command"], init_idx=j["init_idx"], success=bool(done),
                   steps=step, sec=time.time() - t0, error=err)
        with open(outp, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec, flush=True)


if __name__ == "__main__":
    main()
