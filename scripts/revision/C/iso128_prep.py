"""Iso-memory (128 frames), step 2: the vision side for the harness runs, kept just ahead of them.

Disk is full (~11 GB free) and 128-frame prefill embeddings are ~186 MB (full) + ~93 MB (VisionZip
50%) per video, so they live in $QWEN_CACHE_ROOT (tmpfs) and are made on demand: this process
(env ~/qwenvl, as qwen_prep.py) watches the shard files of every arm, embeds (qwen_prep.prep, frames
128, unchanged code path) the next `--ahead` pending videos of every (arm, shard) process, earliest
first, and deletes a video's embedding once no arm that reads it has a pending question on it.
The 128-frame PNG frame cache (iso128_fetch.sh) is kept. run_qwen25vl.py already waits for a
missing embedding. Several workers may run (--worker k --nworkers K: videos with index % K == k).

    QWEN_CACHE_ROOT=/dev/shm/oval_iso128 ~/qwenvl/bin/python scripts/revision/C/iso128_prep.py \
        --out results/revision/C/runs/iso128_qwen25vl --arms full_n128:0:4 oval_n128:0:4 vz50_full_n128:0.5:4
"""
import argparse, json, os, shutil, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen_prep as QP
import torch


def long_docs():
    import pandas as pd
    df = pd.read_parquet(os.path.join(os.path.expanduser("~/vmme"), "videomme/test-00000-of-00001.parquet"))
    df = df[df.duration == "long"].reset_index(drop=True)          # = qwen_docs.load_docs order
    return list(zip(df.question_id, df.videoID))


def done_qids(path):
    s = set()
    if os.path.exists(path):
        for l in open(path):
            if l.endswith("\n"):
                s.add(json.loads(l)["question_id"])
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="name:vz:nshards")
    ap.add_argument("--ahead", type=int, default=3)
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--min_free_gb", type=float, default=15.0, help="tmpfs floor: then only the next video per process")
    a = ap.parse_args()
    assert os.environ.get("QWEN_CACHE_ROOT"), "set QWEN_CACHE_ROOT"
    docs = long_docs()
    vidx = {}
    for _, v in docs:
        vidx.setdefault(v, len(vidx))
    arms = [(n, float(z), int(s)) for n, z, s in (x.split(":") for x in a.arms)]
    proc = model = None
    bad = set()
    while True:
        need = {}          # (vid, vz) -> earliest rank (0 = a process's next video)
        pending = set()
        for name, vz, S in arms:
            for s in range(S):
                done = done_qids(os.path.join(a.out, name, f"shard{s}of{S}.jsonl"))
                pv = []
                for q, v in docs[s::S]:
                    if q not in done:
                        pending.add((v, vz))
                        if v not in pv:
                            pv.append(v)
                for rank, v in enumerate(pv[:a.ahead]):
                    need[v, vz] = min(need.get((v, vz), 99), rank)
        if not pending:
            print("all arms complete", flush=True); break
        # delete consumed embeddings (only this worker's videos)
        for v, i in vidx.items():
            if i % a.nworkers != a.worker:
                continue
            for vz in {z for _, z, _ in arms}:
                p = QP.emb_path(v, "videomme", 128, vz)
                if (v, vz) not in pending and os.path.exists(p):
                    os.remove(p); print(f"deleted {os.path.basename(p)}", flush=True)
        free = shutil.disk_usage(os.environ["QWEN_CACHE_ROOT"]).free / 2**30
        todo = sorted(((r, vidx[v], v, vz) for (v, vz), r in need.items()
                       if vidx[v] % a.nworkers == a.worker and v not in bad
                       and os.path.exists(os.path.join(QP.frame_dir(v, "videomme", 128), "meta.json"))   # fetched
                       and not os.path.exists(QP.emb_path(v, "videomme", 128, vz))
                       and (r == 0 or free > a.min_free_gb)))
        if not todo:
            time.sleep(15); continue
        _, _, v, vz = todo[0]
        if model is None:
            proc, model = QP.load_model()
            # prep() only uses the vision tower, embed_tokens and get_rope_index: park the LM decoder
            # layers and lm_head on the CPU (~14 GB less GPU memory; the tensors prep() computes are unchanged)
            model.model.layers.to("cpu"); model.lm_head.to("cpu"); torch.cuda.empty_cache()
        t0 = time.time()
        try:
            msg = QP.prep(v, "videomme", proc, model, 128, vz)
        except Exception as e:                          # e.g. missing frame cache: log, do not loop on it
            msg = f"ERROR {e!r}"
            bad.add(v)
            print(v, vz, msg, flush=True); continue
        print(v, vz, msg, f"{time.time() - t0:.1f}s free {free:.0f}GB", flush=True)


if __name__ == "__main__":
    main()
