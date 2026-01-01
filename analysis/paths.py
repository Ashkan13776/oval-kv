"""Where the external trees live.

Everything is resolved from environment variables so the repo has no absolute
paths baked in. Defaults assume the layout setup.sh creates:

    $KV_ROOT/
      third_party/FreeKV/          <- upstream, patched by setup.sh
      third_party/InfiniteBench/   <- upstream, for its official scorers

Override any of them if your checkouts live elsewhere.
"""
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get(
    "KV_ROOT", Path(__file__).resolve().parent.parent))
FREEKV = Path(os.environ.get("FREEKV_DIR", ROOT / "third_party/FreeKV"))
INFINITEBENCH = Path(os.environ.get(
    "INFINITEBENCH_DIR", ROOT / "third_party/InfiniteBench"))

FK_ACC = FREEKV / "accuracy"
IB_DIRS = [FK_ACC / "eval/InfiniteBench/results", FK_ACC / "eval/InfiniteBench/res"]
LB2_DIR = FK_ACC / "eval/LongBench2/results"
RESULTS = ROOT / "results"


def use_infinitebench_scorers():
    """Put InfiniteBench's own scorer package on sys.path.

    We score with the reference implementation rather than a reimplementation,
    so the numbers are comparable to published InfiniteBench results.
    """
    src = INFINITEBENCH / "src"
    if not src.is_dir():
        raise SystemExit(
            f"InfiniteBench not found at {src}.\n"
            f"Run ./setup.sh, or set INFINITEBENCH_DIR.")
    sys.path.insert(0, str(src))
