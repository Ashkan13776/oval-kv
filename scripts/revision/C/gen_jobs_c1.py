"""Job lines for C1 (Qwen2.5-VL-7B on Video-MME) -> stdout. Usage:
    python scripts/revision/C/gen_jobs_c1.py sanity|timing|full|iso128 [--arms ...] [--nshards 4]
Arms and flags = Tab. 5 (vision/runs/row*/args.json); OVAL with the Qwen eta table."""
import sys, argparse
H = ("--budget 1792 --sink 128 --recent 128 --page_size 32 --GQA_policy avgSM --skip_layer 1 --sparsity 1 "
     "--last_layer_budget 0 --corr_group avg --batch_size 1")
QT = "results/revision/C/eta/tables/qwen2.5-vl-7b_VCAL_r1.json"
ARMS = {
    "full_n64":   "--method full --frames 64",
    "full_n32":   "--method full --frames 32",
    "quest_n64":  "--method quest --page_rep quest --frames 64",
    "freekv_n64": "--method spec_ret --page_rep quest --spec_ret_steps 2 --spec_ret_corr 0.8 --frames 64",
    "q4key_n64":  "--method quest --page_rep q4key --frames 64",
    "snapkv_n64": "--method quest --page_rep snapkv --frames 64",
    "oval_n64":   f"--method spec_ret --page_rep oval --oval_rank 8 --oval_recon --oval_vrank 16 --vbasis pca "
                  f"--eta_table {QT} --spec_ret_steps 2 --spec_ret_corr 0.8 --frames 64",
}
# Iso-memory (128 frames, Long split only; REPORT_C.md 'Iso-memory (128 frames)'): run under
# scripts/revision/C/iso128.sh (QWEN_CACHE_ROOT on tmpfs, KVC_MAX_TOKENS=32768, iso128_prep.py alongside).
ISO = {"full_n128": ARMS["full_n64"].replace("--frames 64", "--frames 128"),
       "oval_n128": ARMS["oval_n64"].replace("--frames 64", "--frames 128"),
       "vz50_full_n128": ARMS["full_n64"].replace("--frames 64", "--frames 128") + " --vz 0.5"}
R = "results/revision/C/runs/c1_qwen25vl"
PY = "python scripts/revision/C/run_qwen25vl.py --bench videomme"
ap = argparse.ArgumentParser(); ap.add_argument("what"); ap.add_argument("--arms", nargs="+", default=list(ARMS))
ap.add_argument("--nshards", type=int, default=4)
a = ap.parse_args()
if a.what == "sanity":
    q = open("logs/revc/long100_qids.txt").read().split()
    print(f"{PY} {H} {ARMS['full_n64']} --durations long --limit 100 --out {R}/sanity_stream_full_n64 > logs/revc/jobs/sanity_stream.log 2>&1")
    print(f"~/qwenvl/bin/python scripts/revision/C/qwen_official.py --mode prompt --bench videomme --durations long --limit 100 --out {R}/sanity_prompt_hf_n64 > logs/revc/jobs/sanity_prompt.log 2>&1")
    print(f"~/qwenvl/bin/python scripts/revision/C/qwen_official.py --mode hfstream --bench videomme --durations long --limit 100 --out {R}/sanity_hfstream_n64 > logs/revc/jobs/sanity_hfstream.log 2>&1")
elif a.what == "timing":   # first 20 Long videos = 60 questions; written into the arm's run dir as its own shard file
    q = " ".join(open("logs/revc/repro_qids.txt").read().split())
    for arm in a.arms:
        print(f"{PY} {H} {ARMS[arm]} --durations long --qids {q} --file_tag _timing20 --out {R}/{arm} > logs/revc/jobs/{arm}_timing20.log 2>&1")
elif a.what == "iso128":
    RI = "results/revision/C/runs/iso128_qwen25vl"
    for arm in [x for x in a.arms if x in ISO] or list(ISO):
        for s in range(a.nshards):
            print(f"{PY} {H} {ISO[arm]} --durations long --shard {s} --nshards {a.nshards} --out {RI}/{arm} > logs/revc/iso128/{arm}_s{s}.log 2>&1")
elif a.what == "full":
    for arm in a.arms:
        for s in range(a.nshards):
            print(f"{PY} {H} {ARMS[arm]} --durations short medium long --shard {s} --nshards {a.nshards} --out {R}/{arm} > logs/revc/jobs/{arm}_s{s}.log 2>&1")
