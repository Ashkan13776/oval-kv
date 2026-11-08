"""Job lines for C3 (video baselines), both models, both benchmarks -> stdout.
    python scripts/revision/C/gen_jobs_c3.py timing|full --model qwen|llava --bench videomme|lvbench [--arms ...]
Arms:
  vz50_full / vz25_full   VisionZip 50% / 25%, then FullKV (Qwen: pruned embeddings, harness)
  vz50_oval               OVAL (budget 2048) on the VisionZip-50% cache
  infinipot_mem           InfiniPot-V, cache ~ OVAL's KV bytes (~50% of the visual tokens):
                          Qwen block 16 / compress 12 groups (6,480 tok); LLaVA 32 / 24 frames (6,720)
  infinipot_2k            InfiniPot-V, cache ~2K: Qwen 5 / 4 groups (2,025); LLaVA 10 / 8 frames (2,100)
  rekv                    ReKV, 1792-token retrieval: Qwen 4 groups (1,620), LLaVA 8 frames (1,680) + 128 + 128
  hf_full                 dense reference on the same HF path as infinipot/rekv (LLaVA only; for Qwen
                          the HF stream path matched the harness 100/100)
Timing set: as C1/C2 (Video-MME: 20 Long videos = 60 q; LVBench: first 3 q of the first 20 videos)."""
import argparse, contextlib, io, json, os, sys
sys.argv, _argv = sys.argv[:1] + ["full"], sys.argv
with contextlib.redirect_stdout(io.StringIO()):
    import gen_jobs_c1 as C1
sys.argv = _argv
ap = argparse.ArgumentParser(); ap.add_argument("what"); ap.add_argument("--model", required=True)
ap.add_argument("--bench", required=True); ap.add_argument("--arms", nargs="+"); ap.add_argument("--nshards", type=int, default=4)
a = ap.parse_args()
LT = "results/vision/eta/tables/llava-video-7b_VCAL_r1.json"
if a.model == "qwen":
    HP = f"python scripts/revision/C/run_qwen25vl.py --bench {a.bench} {C1.H}"
    QC = f"~/qwenvl/bin/python scripts/revision/C/qwen_c3.py --bench {a.bench}"
    ARMS = {"vz50_full": f"{HP} {C1.ARMS['full_n64']} --vz 0.5", "vz25_full": f"{HP} {C1.ARMS['full_n64']} --vz 0.25",
            "vz50_oval": f"{HP} {C1.ARMS['oval_n64']} --vz 0.5",
            "infinipot_mem": f"{QC} --method infinipot --block 16 --compress 12",
            "infinipot_2k": f"{QC} --method infinipot --block 5 --compress 4",
            "rekv": f"{QC} --method rekv --topk 4 --sink 128 --recent 128"}
else:
    LC = f"python scripts/revision/C/llava_c3.py --bench {a.bench} {C1.H} --frames 64"
    ARMS = {"infinipot_mem": f"{LC} --c3 infinipot --block 32 --compress 24",
            "infinipot_2k": f"{LC} --c3 infinipot --block 10 --compress 8",
            "rekv": f"{LC} --c3 rekv --topk 8",
            "hf_full": f"{LC} --c3 infinipot --block 64 --compress 48"}
R = f"results/revision/C/runs/c3_{a.bench}_{'qwen25vl' if a.model == 'qwen' else 'llava'}"
pre = "LVB_ROOT=$HOME/lvbench " if a.bench == "lvbench" else ""
dur = " --durations short medium long" if a.bench == "videomme" else ""
if a.what == "timing":
    if a.bench == "videomme":
        q = " ".join(open("logs/revc/repro_qids.txt").read().split()); dur = " --durations long"
    else:
        rows = [json.loads(l) for l in open(os.path.expanduser("~/lvbench/video_info.meta.jsonl"))][:20]
        q = " ".join(str(x["uid"]) for r in rows for x in r["qa"][:3])
for arm in a.arms or ARMS:
    tag = f"c3{a.model}_{a.bench}_{arm}"
    if a.what == "timing":
        print(f"{pre}{ARMS[arm]}{dur} --qids {q} --file_tag _timing20 --out {R}/{arm} > logs/revc/jobs/{tag}_timing20.log 2>&1")
    else:
        for s in range(a.nshards):
            print(f"{pre}{ARMS[arm]}{dur} --shard {s} --nshards {a.nshards} --out {R}/{arm} > logs/revc/jobs/{tag}_s{s}.log 2>&1")
