"""Capture stage of the eta_h calibration for Qwen2.5-VL-7B-Instruct (env ~/qwenvl, transformers 4.51.3).

Same design as vision/calibrate_capture.py (LLaVA-Video, Sec. S4.4) and the same six
LLaVA-Video-178K clips (2_3_m_youtube_v0_1, archive 98; identified by
scripts/revision/C/identify_calib.py, list in $CLIPS), in the same order (p0..p5; p0-p3
calibrate, p4-p5 test). Each clip is the stream protocol's prefill for this model:
system prompt + 64 frames (qwen_prep.py: same 64 indices, ~13K visual tokens) up to
<|vision_end|>; no question, no label. Queries: 3 windows of 96 prefill positions ending at
the page-aligned positions T, T-1024, T-2048 (sink/recent 128).

M-RoPE: the keys are HF's cached keys, i.e. after the 3-D rotation; the captured queries
(q_proj outputs at the window positions) are rotated with the SAME 3-D positions of those
tokens (get_rope_index), via HF's own rotary module and apply_multimodal_rotary_pos_emb.
analysis/eta_rules_calibrate.py `stats` then takes post-rotation q and k as it does for
1-D RoPE models; nothing there depends on how the rotation was made.

Writes CAPTURE_DIR/qwen2.5-vl-7b_VCAL_p<i>.pt and CAPTURE_DIR/qwen2.5-vl-7b_oproj.pt in the
format of analysis/eta_rules_calibrate.py capture, plus $VMME_ROOT/calib_qwen/calib.jsonl.

    ~/qwenvl/bin/python scripts/revision/C/qwen_calibrate_capture.py --clips ytb_a.mp4 ... (6, in order)
"""
import argparse, glob, json, os, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qwen_prep as QP

REPO = os.path.join(HERE, "..", "..", "..")
CAP_DIR = os.path.join(REPO, "logs", "eta_rules", "capture")
SRC = os.path.expanduser("~/vmme/calib_src")
MODEL, TASK = "qwen2.5-vl-7b", "VCAL"
SINK, RECENT, PAGE, NQ, NWIN, STRIDE = 128, 128, 32, 96, 3, 1024


@torch.no_grad()
def main(clips):
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as M
    os.makedirs(CAP_DIR, exist_ok=True)
    cal = os.path.join(os.environ.get("VMME_ROOT", os.path.expanduser("~/vmme")), "calib_qwen")
    os.makedirs(cal, exist_ok=True)
    paths = []
    for c in clips:
        hit = glob.glob(os.path.join(SRC, "**", c), recursive=True)
        assert len(hit) == 1, (c, hit)
        paths.append(hit[0])
    with open(os.path.join(cal, "calib.jsonl"), "w") as f:
        for p in paths:
            f.write(json.dumps({"video": os.path.basename(p)}) + "\n")
    proc, model = QP.load_model()
    lm = model.model
    cfg = model.config
    nl, nh, nkv = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
    hd = cfg.hidden_size // nh
    oproj = os.path.join(CAP_DIR, f"{MODEL}_oproj.pt")
    if not os.path.exists(oproj):
        torch.save({"o_proj": [l.self_attn.o_proj.weight.detach().cpu() for l in lm.layers],
                    "n_layers": nl, "n_heads": nh, "n_kv_heads": nkv, "head_dim": hd,
                    "hidden": cfg.hidden_size, "model": MODEL, "path": QP.MODEL}, oproj)
    for pidx, v in enumerate(paths):
        out = os.path.join(CAP_DIR, f"{MODEL}_{TASK}_p{pidx}.pt")
        if os.path.exists(out):
            continue
        t0 = time.time()
        # frames exactly as qwen_prep.extract, from the clip file
        vid = os.path.basename(v)[:-4]
        QP_video_path = QP.video_path
        QP.video_path = lambda _v, _s: v
        fd = QP.frame_dir
        QP.frame_dir = lambda _v, _s: os.path.join(cal, "frames", f"{vid}_n{QP.N}")
        fr, meta, fh = QP.extract(vid, "videomme")
        QP.video_path, QP.frame_dir = QP_video_path, fd
        _, inp = QP.processor_inputs(proc, fr, meta, text="", add_generation_prompt=False)
        ids = inp["input_ids"][0]
        vend = int((ids == cfg.vision_end_token_id).nonzero()[0])
        ids = ids[:vend + 1][None].to(model.device)
        thw = inp["video_grid_thw"].to(model.device)
        emb = lm.embed_tokens(ids)
        vis = model.visual(inp["pixel_values_videos"].to(model.device, torch.bfloat16), grid_thw=thw)
        mask = ids[0] == cfg.video_token_id
        emb[0, mask] = vis.to(emb.dtype)
        pos, _ = model.get_rope_index(ids, None, thw, torch.tensor(inp["second_per_grid_ts"]), torch.ones_like(ids))
        T = ids.shape[1]
        wins = []
        for w in range(NWIN):
            L = SINK + ((T - w * STRIDE - SINK) // PAGE) * PAGE
            if L - SINK - RECENT >= PAGE and L - NQ >= 0:
                wins.append(L)
        wins = sorted(set(wins))
        qpos = torch.cat([torch.arange(L - NQ, L) for L in wins]).to(model.device)
        qcap = {}
        hooks = [layer.self_attn.q_proj.register_forward_hook(
                    lambda m, i, o, li=li: qcap.__setitem__(li, o[0, qpos].detach()))
                 for li, layer in enumerate(lm.layers)]
        o = lm(inputs_embeds=emb, position_ids=pos, use_cache=True)
        for h in hooks:
            h.remove()
        pkv = o.past_key_values
        cos, sin = lm.rotary_emb(emb, pos[:, :, qpos])                     # 3-D positions of the query tokens
        K = torch.empty(nl, nkv, T, hd, dtype=torch.bfloat16)
        V = torch.empty(nl, nkv, T, hd, dtype=torch.bfloat16)
        Q = torch.empty(nl, len(wins), NQ, nh, hd, dtype=torch.bfloat16)
        for li in range(nl):
            K[li], V[li] = pkv.key_cache[li][0].to(torch.bfloat16).cpu(), pkv.value_cache[li][0].to(torch.bfloat16).cpu()
            q = qcap[li].view(1, -1, nh, hd).transpose(1, 2)               # bf16, as HF rotates it
            q, _ = M.apply_multimodal_rotary_pos_emb(q, q[:, :nkv], cos, sin, cfg.rope_scaling["mrope_section"])
            Q[li] = q[0].transpose(0, 1).reshape(len(wins), NQ, nh, hd).to(torch.bfloat16).cpu()
        del o, pkv, qcap
        torch.save({"K": K, "V": V, "Q": Q, "windows": wins, "nq": NQ, "T": T, "prompt_len": T,
                    "problem_idx": pidx, "gen_seed": None, "model": MODEL, "task": TASK,
                    "text": os.path.basename(v), "frames_sha1": fh, "n_visual": int(mask.sum()),
                    "resized_hw": meta["resized_hw"], "created": time.strftime("%Y-%m-%d %H:%M:%S")}, out)
        print(f"capture p{pidx} {os.path.basename(v)}: T={T} n_visual={int(mask.sum())} windows={wins} "
              f"{time.time() - t0:.0f}s", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs=6, required=True)
    main(ap.parse_args().clips)
