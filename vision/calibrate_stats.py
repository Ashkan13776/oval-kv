"""Run analysis/eta_rules_calibrate.py (stats / rules / regret) unchanged, with the
token-domain value PCA of vision/fast_vpca.py swapped into its page_basis module.

    python vision/calibrate_stats.py stats  --model llava-video-7b --task VCAL ...
"""
import os, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
import fast_vpca
import importlib.util
spec = importlib.util.spec_from_file_location("eta_rules_calibrate", os.path.join(HERE, "..", "analysis", "eta_rules_calibrate.py"))
m = importlib.util.module_from_spec(spec)
sys.argv = [spec.origin] + sys.argv[1:]
spec.loader.exec_module(m)
m.PB.build_value_summary = fast_vpca.build_value_summary_tokdomain(m.PB.build_value_summary)
args = m.parse()
import torch
torch.backends.cuda.matmul.allow_tf32 = False
{"capture": m.stage_capture, "stats": m.stage_stats, "rules": m.stage_rules, "regret": m.stage_regret}[args.stage](args)
