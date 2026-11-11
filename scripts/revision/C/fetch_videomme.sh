#!/usr/bin/env bash
# Download the 20 Video-MME video chunks (4 in parallel) and keep the raw mp4s in
# $VMME_ROOT/videos. Resumable: a chunk is skipped once its .done marker exists.
set -u
R="${VMME_ROOT:-$HOME/vmme}"; mkdir -p "$R/videos"
cd "$(dirname "$0")/../../.."
one () {
  k=$1; f=$(printf "videos_chunked_%02d" $k); R="${VMME_ROOT:-$HOME/vmme}"
  [ -e "$R/videos/.$f.done" ] && return
  mkdir -p "$R/unz_$f"
  python vision/download.py $k >/dev/null 2>&1 || { echo "FAIL download $k"; return; }
  unzip -q -o -j "$R/$f.zip" '*.mp4' -d "$R/unz_$f" && mv "$R/unz_$f"/*.mp4 "$R/videos/" && rmdir "$R/unz_$f" \
    && rm -f "$R/$f.zip" && touch "$R/videos/.$f.done" && echo "chunk $k done"
}
export -f one
seq 20 -1 1 | xargs -P 4 -I{} bash -c 'one {}'
echo "ALL: $(ls "$R/videos"/*.mp4 | wc -l) mp4"
