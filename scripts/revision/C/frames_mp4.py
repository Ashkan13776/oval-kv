"""LLaVA-Video frame cache from kept mp4s: vision/frames.py's extract() unchanged
(64 frames, linspace(0, T-1, 64), 384x384 PIL bicubic, PNG), but reading mp4s from
$VMME_ROOT/videos instead of streaming the zips.
    python scripts/revision/C/frames_mp4.py [--vids ID ...] [--workers 24]
"""
import argparse, os, sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "vision"))
import frames as F

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vids", nargs="*")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    vdir = os.path.join(F.ROOT, "videos")
    os.makedirs(F.CACHE, exist_ok=True)
    vids = a.vids or sorted(f[:-4] for f in os.listdir(vdir) if f.endswith(".mp4"))
    jobs = [(os.path.join(vdir, v + ".mp4"), v, a.n) for v in vids]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(F.extract, *zip(*jobs)))
    print(f"{len(res)} videos, {sum(r[1] == 'ok' for r in res)} new", flush=True)
