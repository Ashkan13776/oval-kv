#!/usr/bin/env bash
# LVBench (ungated): videos from lmms-lab/LVBench (14 zips, 61.8 GB), annotations from
# zai-org/LVBench (video_info.meta.jsonl) and lmms-lab's parquet. Raw mp4s kept in
# $LVB_ROOT/videos. Resumable (.done markers); 4 zips in parallel.
set -u
export LVB_ROOT="${LVB_ROOT:-$HOME/lvbench}"; mkdir -p "$LVB_ROOT/videos"
python - <<'PY'
import os
from huggingface_hub import hf_hub_download
R = os.environ["LVB_ROOT"]
print(hf_hub_download("zai-org/LVBench", "video_info.meta.jsonl", repo_type="dataset", local_dir=R))
print(hf_hub_download("lmms-lab/LVBench", "data/train-00000-of-00001.parquet", repo_type="dataset", local_dir=R))
PY
one () {
  k=$1; f=$(printf "videos_chunk_%03d" $k); R="$LVB_ROOT"
  [ -e "$R/videos/.$f.done" ] && return
  python -c "from huggingface_hub import hf_hub_download as h; h('lmms-lab/LVBench', 'video_chunks/$f.zip', repo_type='dataset', local_dir='$R')" >/dev/null 2>&1 \
    || { echo "FAIL download $k"; return; }
  mkdir -p "$R/unz_$f"
  unzip -q -o -j "$R/video_chunks/$f.zip" '*.mp4' -d "$R/unz_$f" && mv "$R/unz_$f"/*.mp4 "$R/videos/" && rmdir "$R/unz_$f" \
    && rm -f "$R/video_chunks/$f.zip" && touch "$R/videos/.$f.done" && echo "chunk $k done"
}
export -f one
seq 1 14 | xargs -P 4 -I{} bash -c 'one {}'
echo "ALL: $(ls "$LVB_ROOT/videos"/*.mp4 | wc -l) mp4"
