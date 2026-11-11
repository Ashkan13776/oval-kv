"""Job lines for the LIBERO Watch-and-Do planner (vla/watchdo_build.py), both models -> stdout.
Exactly the LVBench (C2 + C3) arms and flags, with LVB_ROOT=$HOME/libero_watchdo and run dirs wd_<model>.
    python scripts/revision/C/vla/gen_jobs_wd.py --model qwen|llava [--arms ...] [--pilot N]
"""
import argparse, json, os, subprocess, sys
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--arms", nargs="+")
ap.add_argument("--pilot", type=int, default=0, help="only the questions of the first N sessions (file tag _pilot)")
a = ap.parse_args()
C2 = ["full_n64", "full_n32", "quest_n64", "freekv_n64", "snapkv_n64", "oval_n64"]
C3 = ["infinipot_2k", "rekv", "infinipot_mem", "vz50_full", "vz25_full", "vz50_oval"]
arms = a.arms or C2 + C3
m = "qwen25vl" if a.model == "qwen" else "llava"
lines = []
c2 = [x for x in arms if x in C2]; c3 = [x for x in arms if x in C3]
if c2:
    lines += subprocess.run([sys.executable, f"{HERE}/gen_jobs_c2.py", "full", "--model", a.model, "--arms", *c2],
                            capture_output=True, text=True, check=True).stdout.splitlines()
if c3:
    lines += subprocess.run([sys.executable, f"{HERE}/gen_jobs_c3.py", "full", "--model", a.model, "--bench", "lvbench", "--arms", *c3],
                            capture_output=True, text=True, check=True).stdout.splitlines()
q = ""
if a.pilot:
    rows = [json.loads(l) for l in open(os.path.expanduser("~/libero_watchdo/video_info.meta.jsonl"))][:a.pilot]
    q = " --qids " + " ".join(x["uid"] for r in rows for x in r["qa"]) + " --file_tag _pilot"
for l in lines:
    l = (l.replace("$HOME/lvbench", "$HOME/libero_watchdo").replace(f"c2_lvbench_{m}", f"wd_{m}").replace(f"c3_lvbench_{m}", f"wd_{m}")
          .replace("logs/revc/jobs/c2", "logs/revc/jobs/wd_c2").replace("logs/revc/jobs/c3", "logs/revc/jobs/wd_c3"))
    if a.pilot:
        if "--shard 0 " not in l:
            continue
        l = l.replace("--shard 0 --nshards 4", "--shard 0 --nshards 1").replace(" > logs", q + " > logs").replace("_s0.log", "_pilot.log")
    print(l)
