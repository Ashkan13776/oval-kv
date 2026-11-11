#!/usr/bin/env bash
# Iso-memory (128 frames), step 1: the 128-frame Qwen2.5-VL frame cache of the Video-MME Long split.
# The raw mp4s (~/vmme/videos) were deleted to free disk and the disk has ~11 GB free, so each
# chunk is downloaded to tmpfs, its Long mp4s are unzipped, the 128 frames are extracted
# (qwen_prep.py --frames_only, linspace(0, T-1, 128), same resize/PNG as the 64-frame cache) into
# $QWEN_CACHE_ROOT/frames_qwen/<vid>_n128, and zip + mp4s are deleted. Resumable (.done per chunk).
#   bash scripts/revision/C/iso128_fetch.sh   (CPU + network only)
set -u
cd "$(dirname "$0")/../../.."
export QWEN_CACHE_ROOT=${QWEN_CACHE_ROOT:-/dev/shm/oval_iso128}
S=$QWEN_CACHE_ROOT/src; mkdir -p $S
~/qwenvl/bin/python -c "
import pandas as pd; df = pd.read_parquet('$HOME/vmme/videomme/test-00000-of-00001.parquet')
print('\n'.join(sorted(set(df[df.duration == 'long'].videoID))))" > $S/long_vids.txt
one () {
  k=$1; f=$(printf "videos_chunked_%02d" $k); D=$S/c$k
  [ -e "$S/.$f.done" ] && return
  mkdir -p $D/videos
  VMME_ROOT=$D ~/miniconda3/envs/ovalkv/bin/python vision/download.py $k >/dev/null 2>&1 || { echo "FAIL download $k"; return; }
  unzip -Z1 "$D/$f.zip" | grep '\.mp4$' | while read p; do
    v=$(basename "$p" .mp4); grep -qx -- "$v" $S/long_vids.txt && unzip -q -o -j "$D/$f.zip" "$p" -d $D/videos
  done
  rm -f "$D/$f.zip"; rm -rf $D/.cache
  vids=$(ls $D/videos | sed 's/\.mp4$//')
  n=$(echo $vids | wc -w)
  if [ $n -gt 0 ]; then
    echo $vids | tr ' ' '\n' > $D/vids.txt      # a file, not --vids: some ids start with '-'
    VMME_ROOT=$D ~/qwenvl/bin/python scripts/revision/C/qwen_prep.py --frames 128 --frames_only 12 --vid_file $D/vids.txt \
      || { echo "FAIL frames $k"; return; }
  fi
  rm -rf $D && touch "$S/.$f.done" && echo "chunk $k done: $n long videos"
}
export -f one; export S
seq 1 20 | xargs -P 3 -I{} bash -c 'one {}'
echo "ALL: $(ls -d $QWEN_CACHE_ROOT/frames_qwen/*_n128 2>/dev/null | wc -l) / 300 Long videos with 128-frame caches"
