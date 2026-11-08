"""Job lines for C2 (LVBench, both models, the C1 arms) -> stdout. Usage:
    python scripts/revision/C/gen_jobs_c2.py timing|full --model qwen|llava [--arms ...] [--nshards 4]
Arms = C1 (gen_jobs_c1.ARMS); LLaVA-Video uses its Tab. 5 eta table and --protocol stream.
Timing set: the first 3 questions of each of the first 20 LVBench videos (60 questions, as in C1;
the stream protocol re-prefills per question, so cost per question does not depend on how many
questions a video has). Written as *_timing20.jsonl next to the full-run shards and excluded from
the tables."""
import argparse, json, os, sys
sys.argv, _argv = sys.argv[:1] + ["full"], sys.argv          # reuse gen_jobs_c1's tables without its CLI output
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import gen_jobs_c1 as C1
sys.argv = _argv
LT = "results/vision/eta/tables/llava-video-7b_VCAL_r1.json"
ap = argparse.ArgumentParser(); ap.add_argument("what"); ap.add_argument("--model", required=True)
ap.add_argument("--arms", nargs="+", default=list(C1.ARMS)); ap.add_argument("--nshards", type=int, default=4)
a = ap.parse_args()
if a.model == "qwen":
    PY, R, arms = "LVB_ROOT=$HOME/lvbench python scripts/revision/C/run_qwen25vl.py --bench lvbench", \
        "results/revision/C/runs/c2_lvbench_qwen25vl", dict(C1.ARMS)
else:
    PY, R = "LVB_ROOT=$HOME/lvbench python scripts/revision/C/run_lvbench_llava.py --protocol stream", \
        "results/revision/C/runs/c2_lvbench_llava"
    arms = {k: v.replace(C1.QT, LT) for k, v in C1.ARMS.items()}
tag = f"c2{a.model}"
if a.what == "timing":
    rows = [json.loads(l) for l in open(os.path.expanduser("~/lvbench/video_info.meta.jsonl"))][:20]
    q = " ".join(str(x["uid"]) for r in rows for x in r["qa"][:3])
    for arm in a.arms:
        print(f"{PY} {C1.H} {arms[arm]} --qids {q} --file_tag _timing20 --out {R}/{arm} > logs/revc/jobs/{tag}_{arm}_timing20.log 2>&1")
elif a.what == "full":
    for arm in a.arms:
        for s in range(a.nshards):
            print(f"{PY} {C1.H} {arms[arm]} --shard {s} --nshards {a.nshards} --out {R}/{arm} > logs/revc/jobs/{tag}_{arm}_s{s}.log 2>&1")
