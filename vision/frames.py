"""Frame cache for Video-MME: decode each video once, store N frames losslessly.

Sampling follows lmms-eval's llava_vid wrapper with force_sample=True
(np.linspace(0, total-1, N, dtype=int)), which is how LLaVA-Video is evaluated.
Each frame is resized to the SigLIP input size (384x384, PIL bicubic) exactly as
SiglipImageProcessor does, and stored as PNG. The processor's own resize of a
384x384 image is then the identity, so the pixel values every row sees are
bit-identical to the reference pipeline. N/2 frames are frames[::2] of this cache.

    python vision/frames.py --chunks 1 2 ... --n 64     # streams each zip, deletes it after
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

ROOT = "/home/shahbaa/vmme"
CACHE = os.path.join(ROOT, "frames")
SIZE = 384


def extract(path, vid, n):
    import decord
    out = os.path.join(CACHE, f"{vid}_n{n}")
    if os.path.exists(os.path.join(out, "meta.json")):
        return vid, "cached"
    vr = decord.VideoReader(path, ctx=decord.cpu(0), num_threads=1)
    total = len(vr)
    fps = vr.get_avg_fps()
    idx = np.linspace(0, total - 1, n, dtype=int).tolist()
    frames = vr.get_batch(idx).asnumpy()
    tmp = out + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).resize((SIZE, SIZE), resample=Image.BICUBIC).save(
            os.path.join(tmp, f"{i:03d}.png"), compress_level=1)
    meta = dict(video_id=vid, n=n, total_frames=total, fps=fps, video_time=total / fps,
                frame_idx=idx, frame_time=[i / fps for i in idx],
                orig_hw=list(frames.shape[1:3]))
    json.dump(meta, open(os.path.join(tmp, "meta.json"), "w"))
    os.rename(tmp, out)
    return vid, "ok"


def do_chunk(k, n, workers):
    z = os.path.join(ROOT, f"videos_chunked_{k:02d}.zip")
    if not os.path.exists(z):
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "download.py"), str(k)], check=True)
    tmpd = tempfile.mkdtemp(dir=ROOT, prefix=f"unz{k:02d}_")
    with zipfile.ZipFile(z) as zf:
        names = [m for m in zf.namelist() if m.endswith(".mp4")]
        zf.extractall(tmpd, members=names)
    jobs = [(os.path.join(tmpd, m), os.path.basename(m)[:-4], n) for m in names]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(extract, *zip(*jobs)))
    shutil.rmtree(tmpd)
    os.remove(z)
    print(f"chunk {k:02d}: {len(res)} videos, {sum(r[1]=='ok' for r in res)} new", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, nargs="+", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    for k in a.chunks:
        do_chunk(k, a.n, a.workers)
