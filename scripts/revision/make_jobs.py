"""Write queue job files for the revision decodes (OVAL Tab. 1 configuration + a given eta table)."""
import sys, os
ROOT = os.path.expanduser("~/oval-kv")
BASE = ("--GQA_policy avgSM --spec_ret_steps 2 --spec_ret_corr 0.9 --temperature 0.6 --top_p 0.95 "
        "--max_gen 16384 --sink 512 --recent 512 --page_size 32 --method spec_ret --page_rep oval "
        "--oval_rank 8 --oval_recon --oval_vrank 16 --vbasis pca")
SEEDS = range(42, 50)
def ovalj(block, model, task, tag, table, seed, budget=2048, out=None):
    out = out or f"{ROOT}/results/revision/{block}/runs"
    return (f"{block}_{model}_{task}_{tag}_b{budget}_s{seed}|--model {model} --dataset {task} --seed {seed} "
            f"--budget {budget} --out_root_dir {out} {BASE} --eta_table {table}")
if __name__ == "__main__":
    which = sys.argv[1]
    lines = []
    if which == "A3":
        models = sys.argv[2:] or ["ds-r1-qwen-7b"]
        for task in ["MATH50", "GPQA50c", "AIME24"]:
            for m in models:
                for v in ["shared", "layer", "argmin01"]:
                    t = f"{ROOT}/results/revision/A/tables/{m}_{task}_r1_{v}.json"
                    assert os.path.exists(t), t
                    for s in SEEDS: lines.append(ovalj("A3", m, task, v, t, s))
    elif which == "A1":      # disjoint tables on their own task
        for m in ["ds-r1-qwen-7b", "ds-r1-llama-8b"]:
            for task in ["MATH50", "AIME24"]:
                t = f"{ROOT}/results/revision/A/calib_vr16/tables/{m}_{task}_r1_disjoint.json"
                if os.path.exists(t):
                    for s in SEEDS: lines.append(ovalj("A1", m, task, "disjoint", t, s))
    elif which == "A2math":  # MATH500-calibrated disjoint table on the other tasks
        for m in ["ds-r1-qwen-7b", "ds-r1-llama-8b"]:
            t = f"{ROOT}/results/revision/A/calib_vr16/tables/{m}_MATH50_r1_disjoint.json"
            if os.path.exists(t):
                for task in ["AIME24", "GPQA50c"]:
                    for s in SEEDS: lines.append(ovalj("A2", m, task, "mathcal", t, s))
    elif which == "B":       # budget sweep, Qwen-7B AIME24; 4096 first (slowest)
        m, task = "ds-r1-qwen-7b", "AIME24"
        tab = f"{ROOT}/results/eta_rules/vr16/tables/{m}_{task}_r1.json"
        out = f"{ROOT}/results/revision/B/runs"
        SR = ("--GQA_policy avgSM --spec_ret_steps 2 --spec_ret_corr 0.9 --temperature 0.6 --top_p 0.95 "
              "--max_gen 16384 --sink 512 --recent 512 --page_size 32 --method spec_ret")
        for b in [4096, 1024, 512]:   # 2048 is Tab. 1 (author decision 2026-11-08)
            for s in SEEDS:
                if b != 2048:
                    lines.append(ovalj("B", m, task, "oval", tab, s, budget=b, out=out))
                lines.append(f"B_{m}_{task}_retv_b{b}_s{s}|--model {m} --dataset {task} --seed {s} --budget {b} "
                             f"--out_root_dir {out} {SR} --page_rep oval --oval_rank 8 --eta_table {tab}")
                lines.append(f"B_{m}_{task}_freekv_b{b}_s{s}|--model {m} --dataset {task} --seed {s} --budget {b} "
                             f"--out_root_dir {out} {SR} --page_rep quest")
    print("\n".join(lines))
