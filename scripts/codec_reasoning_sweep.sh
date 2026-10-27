#!/usr/bin/env bash
# Long-reasoning sweep for the OVAL key codec: eta x seed grid at rank 8.
#
# Produces avg@k / pass@k over k seeds for
#   dense    FullKV                                     (DENSE=1)
#   codec    OVAL retrieval + compression, --oval_recon (always)
# at every eta requested. Selection-only (retrieval without compression) is
# opt-in via WITH_SEL=1.
#
# Only (dataset, eta) pairs that do NOT already have all SEEDS on disk are
# queued, so the script is safe to re-run, safe to run alongside another
# scheduler, and never lets two processes write the same .jsonl.
#
# Config is the paper's reasoning protocol and is identical in every cell;
# eta, seed and the codec flag are the only things that vary.
#
#   KV_ROOT      repo root            (default: this script's parent)
#   KV_PY        python to use        (default: python)
#   FREEKV_DIR   patched FreeKV tree  (default: $KV_ROOT/third_party/FreeKV)
#   MODEL        ds-r1-qwen-7b | ds-r1-llama-8b | ds-r1-qwen-14b
#   DATASETS     default "MATH50 AIME24 GPQA50c"
#   ETAS         default "0.0 0.25 0.5 0.75 1.0"
#   SEEDS        default "42 43 44 45 46 47 48 49"   (k = number of seeds)
#   GPUS         default "0 1 2 3"
#   VRANK        also compress VALUES at this rank (--oval_vrank); unset = values exact.
#                In this mode ETAS sweeps the VALUE eta (--veta) and the KEY eta is
#                fixed per dataset by ETAMAP (keys stay at rank 8).
#   VBASIS       value basis in VRANK mode: pca (default; plain PCA, no value eta,
#                ETAS is ignored) or oval (output-aware, ETAS sweeps --veta)
#   ETAMAP       key eta per dataset in VRANK mode
#                default "MATH50:0.25 AIME24:1.0 GPQA50c:0.25"
#   DENSE=1      also run the FullKV control
#   WITH_SEL=1   also run selection-only (no compression)
set -u
KV_ROOT="${KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KV_PY="${KV_PY:-python}"
FREEKV_DIR="${FREEKV_DIR:-$KV_ROOT/third_party/FreeKV}"
M="${MODEL:?set MODEL, e.g. MODEL=ds-r1-qwen-14b}"
cd "$FREEKV_DIR/accuracy" 2>/dev/null || { echo "no FreeKV at $FREEKV_DIR -- run ./setup.sh"; exit 1; }

export PYTHONUNBUFFERED=1
export OVAL_QUANT=0                    # bfloat16 records
export KVC_MAX_TOKENS=32768            # short prompt + 16384 generated
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASETS="${DATASETS:-MATH50 AIME24 GPQA50c}"
ETAS="${ETAS:-0.0 0.25 0.5 0.75 1.0}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49}"
GPUS="${GPUS:-0 1 2 3}"
NSEED=$(echo $SEEDS | wc -w)
# value codec: tags the output path _vr<r> and the per-cell logs. The codec glob
# below matches the tag EXACTLY (..._recon1${VTAG}-pQ2), so a values-exact cell
# and a value-codec cell are never counted as each other.
VRANK="${VRANK:-}"
VBASIS="${VBASIS:-pca}"
[ -n "$VRANK" ] && [ "$VBASIS" = "pca" ] && ETAS="pca"   # one cell per dataset
ETAMAP="${ETAMAP:-MATH50:0.25 AIME24:1.0 GPQA50c:0.25}"
keta_of () { local kv; for kv in $ETAMAP; do [ "${kv%%:*}" = "$1" ] && { echo "${kv#*:}"; return; }; done
             echo "no key eta for $1 in ETAMAP" >&2; exit 1; }

LOG="$KV_ROOT/logs"; OUT="$KV_ROOT/results/codec"
RES="$OUT/${M}-spec_ret"; RESF="$OUT/${M}-full"
mkdir -p "$LOG" "$OUT"
Q="$LOG/codec_reasoning_jobs${VRANK:+_vr$VRANK}.txt"; : > "$Q"

# A cell is queued per SEED, and only if its file holds fewer records than the
# dataset has problems. Counting FILES instead skipped a (dataset, eta) as soon
# as every seed had started, so a restart left every partial cell unfinished.
declare -A NPROB=( [MATH50]=50 [AIME24]=30 [GPQA50c]=50 )
# records in the file for seed $2 of glob prefix $1 (seed 42 carries no -seed tag)
seed_lines () {
  local f
  if [ "$2" = "42" ]; then
    f=$(ls $1*.jsonl 2>/dev/null | grep -v -- '-seed[0-9]*\.jsonl$' | head -1)
  else
    f=$(ls $1*-seed$2.jsonl 2>/dev/null | head -1)
  fi
  if [ -n "$f" ]; then grep -c . "$f"; else echo 0; fi
  return 0
}
# queue the incomplete seeds of one cell; $1 glob prefix, $2 job suffix, $3 label
queue_cell () {
  local S n want=${NPROB[$D]:-50} q=0
  for S in $SEEDS; do
    n=$(seed_lines "$1" "$S")
    if [ "$n" -lt "$want" ]; then echo "$D|$S|$2" >> "$Q"; q=$((q+1)); fi
  done
  [ "$q" -eq 0 ] && echo "   skip $D $3 (all $NSEED seeds complete)"
  return 0
}

for D in $DATASETS; do
  if [ "${DENSE:-0}" = "1" ]; then
    queue_cell "$RESF/${D}-" "-|dense|--method full" dense
  fi
  for E in $ETAS; do
    # E is the key eta, or in VRANK mode the VALUE eta with the key eta from ETAMAP
    if [ -n "$VRANK" ]; then
      KE=$(keta_of "$D")
      if [ "$VBASIS" = "pca" ]; then VTAG="_vr${VRANK}_vpca"; VFLAG="--oval_vrank ${VRANK} --vbasis pca"
      else VTAG="_vr${VRANK}_veta${E}"; VFLAG="--oval_vrank ${VRANK} --vbasis oval --veta ${E}"; fi
    else
      KE=$E; VTAG=""; VFLAG=""
    fi
    if [ "${WITH_SEL:-0}" = "1" ]; then
      queue_cell "$RES/${D}-*oval_r8_eta${E}_q0-pQ2" \
        "$E|sel|--method spec_ret --page_rep oval --eta $E --oval_rank 8" "eta=$E sel"
    fi
    queue_cell "$RES/${D}-*oval_r8_eta${KE}_q0_recon1${VTAG}-pQ2" \
      "$KE|codec${VTAG}|--method spec_ret --page_rep oval --eta $KE --oval_rank 8 --oval_recon $VFLAG" \
      "eta=$KE codec${VTAG}"
  done
done
echo "=== $(wc -l < "$Q") cells queued for $M  $(date -Iseconds)"

run_job () {
  local d s e tag flags g
  IFS='|' read -r d s e tag flags <<< "$1"; g=$2
  echo "=== START $M/$d/s$s/eta$e/$tag gpu=$g $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES=$g $KV_PY -u -m eval.reasoning.pred \
      --model "$M" --dataset "$d" \
      --GQA_policy avgSM --spec_ret_steps 2 --spec_ret_corr 0.9 \
      --temperature 0.6 --top_p 0.95 --max_gen 16384 \
      --budget 2048 --sink 512 --recent 512 --page_size 32 \
      --seed "$s" --out_root_dir "$OUT" $flags \
      > "$LOG/codec_${M}_${d}_s${s}_eta${e}_${tag}.log" 2>&1
  echo "=== DONE  $M/$d/s$s/eta$e/$tag exit=$? $(date -Iseconds)"
}

# SLOTS concurrent jobs per GPU, and a GPU is only claimed when it has MINFREE
# MiB free -- so a card held by someone else's job is skipped and picked up
# automatically once it frees, instead of being excluded by hand.
SLOTS="${SLOTS:-1}"
MINFREE="${MINFREE:-40000}"
SETTLE="${SETTLE:-60}"
SLOTLIST=""
for g in $GPUS; do
  for i in $(seq 1 "$SLOTS"); do SLOTLIST="$SLOTLIST $g:$i"; done
done
echo "=== scheduler: $SLOTS slot(s) x GPUs [$GPUS], minfree ${MINFREE}MiB"

declare -A BUSY
while [ -s "$Q" ]; do
  for slot in $SLOTLIST; do
    g="${slot%%:*}"
    [ -n "${BUSY[$slot]:-}" ] && kill -0 "${BUSY[$slot]}" 2>/dev/null && continue
    BUSY[$slot]=""
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g")
    [ "$free" -lt "$MINFREE" ] && continue
    job=$(head -1 "$Q"); [ -z "$job" ] && break
    sed -i '1d' "$Q"
    run_job "$job" "$g" &
    BUSY[$slot]=$!
    sleep "$SETTLE"
  done
  sleep 45
done
wait
echo "=== SWEEP COMPLETE for $M  $(date -Iseconds)"
