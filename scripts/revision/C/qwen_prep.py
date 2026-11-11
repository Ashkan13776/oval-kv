"""Qwen2.5-VL-7B-Instruct: frames + prefill embeddings, computed once per video (env ~/qwenvl,
transformers 4.51.3, the official Qwen2.5-VL processor and vision encoder).

Frames: the same 64 indices as the LLaVA-Video cache (np.linspace(0, T-1, 64, dtype=int),
lmms-eval force_sample), resized as qwen_vl_utils.fetch_video does (torchvision bicubic,
antialias, uint8) to smart_resize(h, w, 28, min_pixels=P, max_pixels=P) with
P = 420 * 28 * 28, i.e. ~420 merged tokens per temporal group -> ~32 * 420 = 13,440 visual
tokens (LLaVA-Video: 13,440); the exact count per video depends on its aspect ratio and is
recorded. Frames are stored as PNG (lossless) under $VMME_ROOT/frames_qwen/<vid>_n64/.

Prefill of the stream protocol (query-agnostic): chat template with the lmms-eval system
prompt, up to and including <|vision_end|>; the question text follows as decode tokens.
Saved per video ($VMME_ROOT/qwen_emb/<vid>.pt): inputs_embeds of the prefill (bf16),
3-D M-RoPE position ids (get_rope_index), the next text position, token ids and hashes.

    ~/qwenvl/bin/python scripts/revision/C/qwen_prep.py --vids ID ... [--src videomme|lvbench]
"""
import argparse, hashlib, json, math, os, sys, time

import numpy as np
import torch
from PIL import Image

ROOT = os.environ.get("VMME_ROOT", os.path.expanduser("~/vmme"))
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
N = 64
FACTOR = 28
PIX = 420 * 28 * 28
SYSTEM = "You are a helpful assistant."


def smart_resize(h, w, factor=FACTOR, min_pixels=PIX, max_pixels=PIX):
    # qwen_vl_utils.vision_process.smart_resize
    hb = max(factor, round(h / factor) * factor)
    wb = max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        hb = math.floor(h / beta / factor) * factor
        wb = math.floor(w / beta / factor) * factor
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return hb, wb


def video_path(vid, src):
    if src == "videomme":
        return os.path.join(ROOT, "videos", vid + ".mp4")
    return os.path.join(os.environ.get("LVB_ROOT", os.path.expanduser("~/lvbench")), "videos", vid + ".mp4")


def _base(src):
    # QWEN_CACHE_ROOT (optional): put frames_qwen/ and qwen_emb/ elsewhere, e.g. the 128-frame
    # iso-memory runs keep them in /dev/shm and delete them once consumed (disk is full).
    if os.environ.get("QWEN_CACHE_ROOT"):
        return os.environ["QWEN_CACHE_ROOT"]
    return ROOT if src == "videomme" else os.environ.get("LVB_ROOT", os.path.expanduser("~/lvbench"))


def frame_dir(vid, src, n=N):
    return os.path.join(_base(src), "frames_qwen", f"{vid}_n{n}")


def emb_path(vid, src, frames=N, vz=0.0):
    base = _base(src)
    name = vid + ("" if frames == N else f"_n{frames}") + (f"_vz{round(vz * 100)}" if vz else "")
    return os.path.join(base, "qwen_emb", name + ".pt")


def extract(vid, src, n=N):
    """n (default 64) frames, qwen_vl_utils-style resize, PNG cache. Returns (uint8 T,C,H,W tensor, meta).
    n = 128 (iso-memory runs) samples np.linspace(0, T-1, 128) directly, not a superset of the 64."""
    import decord
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF
    N = n                                   # local: every N below is this cache's frame count
    d = frame_dir(vid, src, n)
    if not os.path.exists(os.path.join(d, "meta.json")):
        vr = decord.VideoReader(video_path(vid, src), ctx=decord.cpu(0), num_threads=4)
        total, fps = len(vr), vr.get_avg_fps()
        idx = np.linspace(0, total - 1, N, dtype=int).tolist()
        fr = torch.from_numpy(vr.get_batch(idx).asnumpy()).permute(0, 3, 1, 2)       # T,C,H,W uint8
        h, w = fr.shape[2:]
        rh, rw = smart_resize(h, w)
        fr = TF.resize(fr, [rh, rw], interpolation=InterpolationMode.BICUBIC, antialias=True)
        tmp = d + f".tmp{os.getpid()}"; os.makedirs(tmp, exist_ok=True)     # per process: concurrent jobs may race
        for i, f in enumerate(fr):
            Image.fromarray(f.permute(1, 2, 0).numpy()).save(os.path.join(tmp, f"{i:03d}.png"), compress_level=1)
        meta = dict(video_id=vid, n=N, total_frames=total, fps=fps, video_time=total / fps, frame_idx=idx,
                    orig_hw=[int(h), int(w)], resized_hw=[rh, rw],
                    sample_fps=N / max(total, 1e-6) * fps)          # qwen_vl_utils' sample_fps
        json.dump(meta, open(os.path.join(tmp, "meta.json"), "w"))
        try:
            os.rename(tmp, d)
        except OSError:                      # another process finished the same video first
            import shutil; shutil.rmtree(tmp, ignore_errors=True)
    meta = json.load(open(os.path.join(d, "meta.json")))
    raw = [open(os.path.join(d, f"{i:03d}.png"), "rb").read() for i in range(N)]
    fh = hashlib.sha1(b"".join(raw)).hexdigest()[:16]
    fr = torch.stack([torch.from_numpy(np.array(Image.open(os.path.join(d, f"{i:03d}.png")).convert("RGB")))
                      for i in range(N)]).permute(0, 3, 1, 2)
    return fr, meta, fh


def load_model(device="cuda:0"):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                               attn_implementation="sdpa", device_map=device)
    model.eval()
    return proc, model


def messages(text=None):
    content = [{"type": "video"}]
    if text is not None:
        content.append({"type": "text", "text": text})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def processor_inputs(proc, fr, meta, text=None, add_generation_prompt=True):
    prompt = proc.apply_chat_template(messages(text), tokenize=False, add_generation_prompt=add_generation_prompt)
    inp = proc(text=[prompt], videos=[fr.float()], fps=[meta["sample_fps"]], return_tensors="pt")
    return prompt, inp


MAX_T = 32   # temporal groups per vision-encoder call (= the 64-frame videos, which run in one call)


def visual_chunked(visual, pv, thw, fn):
    """fn(visual, pv, thw) -> tuple of per-token tensors (t-major), run on <= MAX_T temporal groups at a
    time. Exact up to kernel shapes: every vision block attends only within a temporal group (window
    blocks within windows, full blocks within cu_seqlens = one group), the vision rotary embedding
    depends on (h, w) only, and patches/merged tokens are stored t-major. Needed at 128 frames, where
    HF 4.51's SDPA vision path materialises a dense mask over all patches of the video (OOM)."""
    t, h, w = (int(x) for x in thw[0])
    if t <= MAX_T:
        return fn(visual, pv, thw)
    outs = []
    for a in range(0, t, MAX_T):
        b = min(t, a + MAX_T)
        outs.append(fn(visual, pv[a * h * w:b * h * w], torch.tensor([[b - a, h, w]], device=thw.device, dtype=thw.dtype)))
    return tuple(torch.cat(x, 0) for x in zip(*outs))


@torch.no_grad()
def prep(vid, src, proc, model, frames=N, vz=0.0):
    out = emb_path(vid, src, frames, vz)
    if os.path.exists(out):
        return "cached"
    fr, meta, fh = extract(vid, src, max(frames, N))
    if frames < N:          # N/2 (Tab. 5 row 2): every other cached frame; fps metadata scaled to match
        step = N // frames
        assert N % frames == 0
        fr = fr[::step]
        meta = dict(meta, sample_fps=meta["sample_fps"] / step)
        raw = [open(os.path.join(frame_dir(vid, src), f"{i:03d}.png"), "rb").read() for i in range(0, N, step)]
        fh = hashlib.sha1(b"".join(raw)).hexdigest()[:16]
    prompt, inp = processor_inputs(proc, fr, meta, text="", add_generation_prompt=False)
    ids = inp["input_ids"][0]
    vend = int((ids == model.config.vision_end_token_id).nonzero()[0])
    ids = ids[:vend + 1][None].to(model.device)                           # prefill: ... <|vision_end|>
    pv = inp["pixel_values_videos"].to(model.device, torch.bfloat16)
    thw = inp["video_grid_thw"].to(model.device)
    spg = inp["second_per_grid_ts"]
    ph = hashlib.sha1(inp["pixel_values_videos"].float().numpy().tobytes()).hexdigest()[:16]
    emb = model.model.embed_tokens(ids)
    if vz:                                  # C3 VisionZip (scripts/revision/C/visionzip.py)
        import visionzip as VZ
        vis, vz_score, vz_key = visual_chunked(model.visual, pv, thw, VZ.qwen_visual_with_scores)
    else:
        vis, = visual_chunked(model.visual, pv, thw, lambda m, p, g: (m(p, grid_thw=g),))
    mask = ids[0] == model.config.video_token_id
    assert int(mask.sum()) == vis.shape[0], (int(mask.sum()), vis.shape)
    emb[0, mask] = vis.to(emb.dtype)
    pos, delta = model.get_rope_index(ids, None, thw, torch.tensor(spg), torch.ones_like(ids))
    pos = pos[:, 0].to(torch.int32).cpu()                                 # (3, L)
    nxt = int(pos.max()) + 1                 # unpruned next position, as upstream VisionZip (rope_deltas)
    n_full = int(mask.sum())
    if vz:
        keep, kv = VZ.select_merge(vz_score, vz_key, vis.to(emb.dtype), vz)
        vpos = mask.nonzero(as_tuple=True)[0]
        emb[0, vpos[keep]] = kv
        sk = torch.ones(ids.shape[1], dtype=torch.bool, device=ids.device)
        sk[vpos] = False; sk[vpos[keep]] = True
        emb, ids, pos = emb[:, sk], ids[:, sk], pos[:, sk.cpu()]
        mask = ids[0] == model.config.video_token_id
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(dict(emb=emb[0].cpu(), pos=pos, next_pos=nxt, ids=ids[0].cpu(), n_visual=int(mask.sum()),
                    n_visual_unpruned=n_full, vz=vz,
                    grid_thw=thw[0].tolist(), second_per_grid_ts=float(spg[0]), resized_hw=meta["resized_hw"],
                    frames_sha1=fh, pixels_sha1=ph,
                    ids_sha1=hashlib.sha1(ids.cpu().numpy().tobytes()).hexdigest()[:16],
                    created=time.strftime("%Y-%m-%d %H:%M:%S")), out + f".tmp{os.getpid()}")
    os.replace(out + f".tmp{os.getpid()}", out)
    return f"ok n_visual={int(mask.sum())} hw={meta['resized_hw']} L={ids.shape[1]}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vids", nargs="*")
    ap.add_argument("--vid_file")
    ap.add_argument("--src", default="videomme", choices=["videomme", "lvbench"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--frames", type=int, default=N)
    ap.add_argument("--vz", type=float, default=0.0, help="VisionZip retention (0.5, 0.25); 0 = off")
    ap.add_argument("--frames_only", type=int, default=0, help="CPU only: extract the frame cache with this many workers")
    a = ap.parse_args()
    vids = a.vids or open(a.vid_file).read().split()
    vids = vids[a.shard::a.nshards]
    if a.frames_only:
        from concurrent.futures import ProcessPoolExecutor
        def _one(v):
            try:
                extract(v, a.src, max(a.frames, N)); return v, "ok"
            except Exception as e:
                return v, f"ERR {e!r}"
        with ProcessPoolExecutor(a.frames_only) as ex:
            res = list(ex.map(_one, vids))
        print(f"{len(res)} videos, errors: {[r for r in res if r[1] != 'ok']}", flush=True)
        sys.exit(0)
    proc, model = load_model()
    for v in vids:
        t0 = time.time()
        try:
            print(v, prep(v, a.src, proc, model, a.frames, a.vz), f"{time.time() - t0:.1f}s", flush=True)
        except FileNotFoundError as e:
            print(v, "MISSING", e, flush=True)
