"""Done markers for decode cells whose queue wrapper died while the decode kept running
(2026-11-08 scheduler restart). For each logs/revision/running/<id> whose PID is not a child of a
live queue wrapper, wait for the PID to exit, then write logs/revision/done/<id> only if the
cell's output file has every problem. Incomplete cells are left for a later queue pass (the
harness resumes partial cells).   python scripts/revision/orphan_done.py ID [ID ...]"""
import os, sys, time, datetime, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); sys.argv, ids = sys.argv[:1], sys.argv[1:]
import score
R = score.ROOT / "logs/revision"
lines = {}
for jf in glob.glob(str(score.ROOT / "scripts/revision/jobs_*.txt")):
    for l in open(jf):
        if "|" in l and not l.startswith("#"):
            j, a = l.rstrip("\n").split("|", 1); lines[j] = a
pend = {i: int(open(R / "running" / i).read()) for i in ids}
while pend:
    for i, pid in list(pend.items()):
        if os.path.exists(f"/proc/{pid}"):
            continue
        a = lines[i]; arg = lambda n: score.arg(a, n)
        f = score.find(arg("model"), arg("dataset"), int(arg("budget")), int(arg("seed")), a, arg("out_root_dir"))
        n = len(score.outcomes(f)) if f else 0
        ok = n == score.NEXP[arg("dataset")]
        now = datetime.datetime.now().isoformat(timespec="seconds")
        if ok:
            open(R / "done" / i, "w").write(now + "\n")
        (R / "running" / i).unlink(missing_ok=True)
        open(R / "orphans.log", "a").write(f"{now} {i} pid {pid} exited, {n} records -> {'done' if ok else 'INCOMPLETE, needs a queue pass'}\n")
        del pend[i]
    time.sleep(120)
