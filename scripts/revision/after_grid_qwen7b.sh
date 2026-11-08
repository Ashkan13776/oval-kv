#!/usr/bin/env bash
# After the Qwen-7B r_v=16 grid: free its captures (all outputs are in results/revision/A/grid_vr16)
# and regenerate the A5 figures in the OVAL setting.
source "$HOME/oval-kv/scripts/revision/env.sh"; cd "$KV_ROOT"
until [ "$(ls results/revision/A/grid_vr16/ds-r1-qwen-7b_*.json 2>/dev/null | wc -l)" -ge 3 ] && \
      grep -q "DONE" <(tail -1 logs/revision/grid_vr16.log); do sleep 300; done
ok=1; for t in MATH50 AIME24 GPQA50c; do
  $KV_PY -c "import json;s=json.load(open('results/revision/A/grid_vr16/ds-r1-qwen-7b_$t.json'));assert len(s['grid'])==21 and 'regret' in s" || ok=0; done
if [ $ok = 1 ]; then rm -f logs/revision/capture/ds-r1-qwen-7b_*_p*.pt; echo "[$(date -Iseconds)] freed qwen-7b captures" >> logs/revision/grid_vr16.log; fi
$KV_PY analysis/fig_eta_revision.py >> logs/revision/grid_vr16.log 2>&1
echo "[$(date -Iseconds)] figures regenerated" >> logs/revision/grid_vr16.log
