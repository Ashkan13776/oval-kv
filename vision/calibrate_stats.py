"""Run analysis/eta_rules_calibrate.py (stats / rules / regret) unchanged. (The first
calibration, 2026-11-06, swapped the token-domain value PCA in from a helper;
oval/page_basis.py now does the same itself, so nothing is swapped any more.)

    python vision/calibrate_stats.py stats  --model llava-video-7b --task VCAL ...
"""
import os, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
import importlib.util
spec = importlib.util.spec_from_file_location("eta_rules_calibrate", os.path.join(HERE, "..", "analysis", "eta_rules_calibrate.py"))
m = importlib.util.module_from_spec(spec)
sys.argv = [spec.origin] + sys.argv[1:]
spec.loader.exec_module(m)
args = m.parse()
import torch
torch.backends.cuda.matmul.allow_tf32 = False
{"capture": m.stage_capture, "stats": m.stage_stats, "rules": m.stage_rules, "regret": m.stage_regret}[args.stage](args)
