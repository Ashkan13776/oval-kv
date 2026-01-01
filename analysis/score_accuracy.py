"""Score the LongBench v2 / LongGenBench cells from raw generations.

Raw generations are not committed (293 MB for LongGenBench alone); this
regenerates results/*_accuracy/oval_eta_sweep.json from a FreeKV checkout that
has been run per scripts/.  LongBench v2 replicates their result.py exactly;
LongGenBench reuses their parse_blocks / calculate_completion_rate, scoring the
LAST 400 records because their append_to_json_file accumulates across runs.
"""
import json, glob, os, re, statistics as st, sys
ACC = sys.argv[1] if len(sys.argv) > 1 else "."     # path to FreeKV/accuracy

rows = []
for f in sorted(glob.glob(f"{ACC}/eval/LongBench2/results/*oval*.jsonl")):
    d = [json.loads(l) for l in open(f)]
    a = lambda s: 100*sum(int(p['judge']) for p in d if s(p))/max(1, len([p for p in d if s(p)]))
    rows.append(dict(cell=os.path.basename(f), n=len(d),
                     overall=round(a(lambda p: True), 2),
                     short=round(a(lambda p: p['length'] == 'short'), 2),
                     medium=round(a(lambda p: p['length'] == 'medium'), 2),
                     long=round(a(lambda p: p['length'] == 'long'), 2),
                     nulls=sum(1 for p in d if p.get('pred') is None)))
print(json.dumps(rows, indent=1))
