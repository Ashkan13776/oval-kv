#!/usr/bin/env bash
# Fetch the upstream trees and install OVAL into FreeKV.
#
# FreeKV is not vendored here: our change to it is ~500 lines, so we ship a
# patch plus the new modules and apply them to a pinned upstream checkout.
set -euo pipefail
KV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP="$KV_ROOT/third_party"; mkdir -p "$TP"

FREEKV_REPO=https://github.com/sjtu-zhao-lab/FreeKV.git
FREEKV_PIN=2c8a7d25c9f3c7c15ce15b2f84cd03f477bd7469
IB_REPO=https://github.com/OpenBMB/InfiniteBench.git

# FlashInfer pins one of its own submodules (spdlog) by SSH URL, which fails
# for anyone without GitHub SSH keys -- the clone aborts midway. Rewrite SSH to
# HTTPS for the duration of this clone only (-c, not a global config change).
GIT_HTTPS=(-c url."https://github.com/".insteadOf=git@github.com:)

if [ ! -d "$TP/FreeKV/.git" ]; then
  echo "== cloning FreeKV @ $FREEKV_PIN"
  git "${GIT_HTTPS[@]}" clone --recurse-submodules "$FREEKV_REPO" "$TP/FreeKV"
fi
git -C "$TP/FreeKV" checkout -q "$FREEKV_PIN"
git "${GIT_HTTPS[@]}" -C "$TP/FreeKV" submodule update --init --recursive

# FreeKV's own README requires two submodule patches that nothing here applied.
# Without them the CUDA extension either refuses GQA group sizes 5 and 7 (qwen)
# or has no bfloat16 vectorised IO (needed whenever OVAL_QUANT=0).
echo "== patching submodules (flashinfer GQA sizes, raft bf16)"
"${KV_PY:-python}" - "$TP/FreeKV" <<'PYEOF'
import re, sys, pathlib
root = pathlib.Path(sys.argv[1])
p = root / "3rdparty/flashinfer/include/flashinfer/utils.cuh"
s = p.read_text()
if "group_size == 5" in s:
    print("   flashinfer: already patched")
else:
    old = ("  } else if (group_size == 4) {                              \\\n"
           "    constexpr size_t GROUP_SIZE = 4;                         \\\n"
           "    __VA_ARGS__                                              \\\n")
    assert old in s, "flashinfer anchor not found"
    blk = lambda g: (f"  }} else if (group_size == {g}) {{                              \\\n"
                     f"    constexpr size_t GROUP_SIZE = {g};                         \\\n"
                     f"    __VA_ARGS__                                              \\\n")
    p.write_text(s.replace(old, old + blk(5) + blk(7), 1))
    print("   flashinfer: added GQA group sizes 5 and 7")
p = root / "3rdparty/raft/cpp/include/raft/util/vectorized.cuh"
s = p.read_text()
if "__nv_bfloat16" in s:
    print("   raft: already patched")
else:
    s = s.replace("#include <cuda_fp16.h>", "#include <cuda_fp16.h>\n#include <cuda_bf16.h>", 1)
    m = re.search(r"template <>\nstruct IOType<__half2, 4> \{.*?\n\};\n", s, re.S)
    assert m, "raft anchor not found"
    bf = "".join(f"""
template <>
struct IOType<{t}, {n}> {{
  typedef {ty} Type;
}};
""" for t, n, ty in [("__nv_bfloat16",1,"__nv_bfloat16"),("__nv_bfloat16",2,"__nv_bfloat162"),
                     ("__nv_bfloat16",4,"uint2"),("__nv_bfloat16",8,"uint4"),
                     ("__nv_bfloat162",1,"__nv_bfloat162"),("__nv_bfloat162",2,"uint2"),
                     ("__nv_bfloat162",4,"uint4")])
    p.write_text(s[:m.end()] + bf + s[m.end():])
    print("   raft: added bf16 include + IOType specialisations")
PYEOF

# eval/reasoning/pred.py hardcodes eval/o1/... but the directory upstream is
# eval/reasoning/. Upstream bug at this pin; a symlink satisfies both.
ln -sfn reasoning "$TP/FreeKV/accuracy/eval/o1"

echo "== applying patch"
git -C "$TP/FreeKV" apply --3way "$KV_ROOT/patches/freekv.patch"

echo "== installing OVAL modules"
install -Dm644 "$KV_ROOT/oval/page_basis.py"   "$TP/FreeKV/accuracy/kvc/patch/oval.py"
install -Dm644 "$KV_ROOT/oval/page_summary.py" "$TP/FreeKV/source/freekv/oval_summary.py"
install -Dm644 "$KV_ROOT/oval/raas.py"         "$TP/FreeKV/source/freekv/raas.py"
install -Dm644 "$KV_ROOT/oval/oval_estimate.cu" \
               "$TP/FreeKV/source/freekv_cpp/src/oval_estimate.cu"
install -Dm644 "$KV_ROOT/oval/infinitebench_pred.py" \
               "$TP/FreeKV/accuracy/eval/InfiniteBench/pred.py"

if [ ! -d "$TP/InfiniteBench/.git" ]; then
  echo "== cloning InfiniteBench (official scorers)"
  git clone --depth 1 "$IB_REPO" "$TP/InfiniteBench"
fi

cat <<MSG

Done. Next:
  export KV_ROOT="$KV_ROOT"
  export KV_PY=\$(which python)          # a torch>=2.8 / CUDA 12.8 env
  ./scripts/build_freekv.sh              # builds FreeKV's CUDA extension
  ./scripts/ib_freekv.sh                 # InfiniteBench: FreeKV vs OVAL
  python analysis/all_tables.py          # score everything found on disk
MSG
