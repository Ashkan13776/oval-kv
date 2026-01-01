"""Seeded record subset selection for InfiniteBench.

WHY THIS EXISTS
---------------
InfiniteBench's .jsonl files are ORDERED, not shuffled, so "the first N
records" is a biased sample. Measured across the shipped data:

  * passkey / number_string / kv_retrieval are SORTED BY NEEDLE POSITION.
    The first 50 kv_retrieval records place the answer at 0.0%, 0.2%, 0.4%,
    ... 9.8% of the context — all 50 inside the leading 10%, which is exactly
    where LOCKS' always-kept sink page sits. Over all 500 records the position
    is uniform on 0-100%.
  * longbook_qa_chn first-50 median context is 233K chars vs 1,139K over the
    full split (5x short).
  * code_debug first-50 median is 667K vs 431K (1.55x long).

Using first-50 scored kv-retr at 60.0 against the paper's 28.0 — above the
paper's own FullKV reference, which is impossible for a method attending ~2%
of tokens. See notes/finding-kv-retr.md.

The paper's phrase is a "seeded subset of 50 records per task"
(tab_infinitebench_main.tex:3). A seeded RANDOM draw is the only reading
consistent with that wording and with the data being pre-sorted. The seed
itself is unpublished, so ours is an explicit choice, not a match.
"""
import json
import random
from pathlib import Path

DEFAULT_SEED = 42


def count_records(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def select_indices(path: Path, n: int, seed: int = DEFAULT_SEED) -> list[int]:
    """Indices of the seeded subset, ascending.

    Uses random.Random(seed).sample over the full record count, so the draw
    depends only on (seed, n, total) — reproducible from the seed alone.
    """
    total = count_records(path)
    if n >= total:
        return list(range(total))
    return sorted(random.Random(seed).sample(range(total), n))


def load_subset(path: Path, n: int, seed: int = DEFAULT_SEED):
    """Yield (index, record) for the seeded subset, streaming the file once."""
    want = set(select_indices(path, n, seed))
    with open(path) as f:
        for i, line in enumerate(f):
            if i in want:
                yield i, json.loads(line)
