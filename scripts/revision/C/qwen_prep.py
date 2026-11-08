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


def frame_dir(vid, src):
    base = ROOT if src == "videomme" else os.environ.get("LVB_ROOT", os.path.expanduser("~/lvbench"))
    return os.path.join(base, "frames_qwen", f"{vid}_n{N}")


def emb_path(vid, src, frames=N):
    base = ROOT if src == "videomme" else os.environ.get("LVB_ROOT", os.path.expanduser("~/lvbench"))
    return os.path.join(base, "qwen_emb", f"{vid}.pt" if frames == N else f"{vid}_n{frames}.pt")


def extract(vid, src):
    """64 frames, qwen_vl_utils-style resize, PNG cache. Returns (uint8 T,C,H,W tensor, meta)."""
    import decord
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF
    d = frame_dir(vid, src)
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


@torch.no_grad()
def prep(vid, src, proc, model, frames=N):
    out = emb_path(vid, src, frames)
    if os.path.exists(out):
        return "cached"
    fr, meta, fh = extract(vid, src)
    if frames != N:          # N/2 (Tab. 5 row 2): every other cached frame; fps metadata scaled to match
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
    vis = model.visual(pv, grid_thw=thw)
    mask = ids[0] == model.config.video_token_id
    assert int(mask.sum()) == vis.shape[0], (int(mask.sum()), vis.shape)
    emb[0, mask] = vis.to(emb.dtype)
    pos, delta = model.get_rope_index(ids, None, thw, torch.tensor(spg), torch.ones_like(ids))
    pos = pos[:, 0].to(torch.int32).cpu()                                 # (3, L)
    nxt = int(pos.max()) + 1
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(dict(emb=emb[0].cpu(), pos=pos, next_pos=nxt, ids=ids[0].cpu(), n_visual=int(mask.sum()),
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
    ap.add_argument("--frames_only", type=int, default=0, help="CPU only: extract the frame cache with this many workers")
    a = ap.parse_args()
    vids = a.vids or open(a.vid_file).read().split()
    vids = vids[a.shard::a.nshards]
    if a.frames_only:
        from concurrent.futures import ProcessPoolExecutor
        def _one(v):
            try:
                extract(v, a.src); return v, "ok"
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
            print(v, prep(v, a.src, proc, model, a.frames), f"{time.time() - t0:.1f}s", flush=True)
        except FileNotFoundError as e:
            print(v, "MISSING", e, flush=True)
