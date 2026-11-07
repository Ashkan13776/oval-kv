"""Capture stage of the eta_h calibration for LLaVA-Video (prefill windows).

Writes files in exactly the format of analysis/eta_rules_calibrate.py capture,
so its `stats`, `rules` and `regret` stages run unchanged on them:
  CAPTURE_DIR/llava-video-7b_VCAL_p<i>.pt   K, V (post-RoPE, every layer), Q windows
  CAPTURE_DIR/llava-video-7b_oproj.pt       o_proj weights

Calibration data: 6 YouTube clips of 2-3 minutes from LLaVA-Video-178K
(2_3_m_youtube_v0_1, archive 98), none of them in Video-MME. Each is fed as the
stream protocol's prefill (system prompt + 64 frames + time instruction; no
question, no label). Queries: 3 windows of 96 prefill positions ending at the
page-aligned positions T, T-1024, T-2048 (the reasoning calibration's layout,
with the long-context sink/recent of 128), i.e. video-token and time-instruction
queries attending to the cached video.

    python vision/calibrate_capture.py            # then eta_rules_calibrate.py stats/rules/regret
"""
import glob, json, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_videomme as RV

CAL_DIR = os.path.join(os.environ.get("VMME_ROOT", os.path.expanduser("~/vmme")), "calib")
CAP_DIR = os.path.join(HERE, "..", "logs", "eta_rules", "capture")
MODEL, TASK = "llava-video-7b", "VCAL"
SINK, RECENT, PAGE, NQ, NWIN, STRIDE, N = 128, 128, 32, 96, 3, 1024, 64


def frames(path):
    import decord
    from PIL import Image
    vr = decord.VideoReader(path, ctx=decord.cpu(0), num_threads=1)
    T, fps = len(vr), vr.get_avg_fps()
    idx = np.linspace(0, T - 1, N, dtype=int).tolist()
    arr = np.stack([np.array(Image.fromarray(f).resize((384, 384), resample=Image.BICUBIC))
                    for f in vr.get_batch(idx).asnumpy()])
    return arr, ",".join(f"{i / fps:.2f}s" for i in idx), T / fps


def main():
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    os.makedirs(CAP_DIR, exist_ok=True)
    vids = sorted(glob.glob(os.path.join(CAL_DIR, "videos", "**", "*.mp4"), recursive=True))
    assert len(vids) == 6, vids
    with open(os.path.join(CAL_DIR, "calib.jsonl"), "w") as f:      # row list for the stats stage
        for v in vids:
            f.write(json.dumps({"video": os.path.basename(v)}) + "\n")

    class A: pass
    tok, model, proc = RV.load_llava(A())
    lm = model.model
    nl, nh, nkv = len(lm.layers), model.config.num_attention_heads, model.config.num_key_value_heads
    hd = model.config.hidden_size // nh
    oproj = os.path.join(CAP_DIR, f"{MODEL}_oproj.pt")
    if not os.path.exists(oproj):
        torch.save({"o_proj": [l.self_attn.o_proj.weight.detach().cpu() for l in lm.layers],
                    "n_layers": nl, "n_heads": nh, "n_kv_heads": nkv, "head_dim": hd,
                    "hidden": model.config.hidden_size, "model": MODEL, "path": RV.MODEL}, oproj)
    for pidx, v in enumerate(vids):
        out = os.path.join(CAP_DIR, f"{MODEL}_{TASK}_p{pidx}.pt")
        if os.path.exists(out):
            continue
        arr, ftime, vtime = frames(v)
        pix = proc.preprocess(list(arr), return_tensors="pt")["pixel_values"].to("cuda:0", torch.bfloat16)
        ti = (f"The video lasts for {vtime:.2f} seconds, and {len(arr)} frames are uniformly sampled from it. "
              f"These frames are located at {ftime}.Please answer the following questions related to this video.")
        conv = conv_templates["qwen_1_5"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + ti)
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), tok, IMAGE_TOKEN_INDEX, return_tensors="pt")[None].cuda()
        # prefill of the stream protocol: everything up to the end of the time instruction
        p = int((ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0])
        pre = tok("\n" + ti).input_ids
        k = len(pre)
        while k > 0 and ids[0, p + 1:p + 1 + k].tolist() != pre[:k]:
            k -= 1
        ids = ids[:, :p + 1 + k]
        with torch.no_grad():
            _, _, _, _, emb, _ = model.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, [pix],
                                                                             modalities=["video"])
        T = emb.shape[1]
        wins = []
        for w in range(NWIN):
            L = SINK + ((T - w * STRIDE - SINK) // PAGE) * PAGE
            if L - SINK - RECENT >= PAGE and L - NQ >= 0:
                wins.append(L)
        wins = sorted(set(wins))
        qpos = torch.cat([torch.arange(L - NQ, L) for L in wins]).cuda()
        qcap = {}
        hooks = [layer.self_attn.q_proj.register_forward_hook(
                    lambda m, i, o, li=li: qcap.__setitem__(li, o[0, qpos].detach()))
                 for li, layer in enumerate(lm.layers)]
        t0 = time.time()
        with torch.no_grad():
            from transformers import Qwen2ForCausalLM
            fw = Qwen2ForCausalLM.forward(model, inputs_embeds=emb, use_cache=True, num_logits_to_keep=1)
        for h in hooks:
            h.remove()
        pkv = fw.past_key_values
        cos, sin = lm.rotary_emb(qcap[0][None].float(), qpos[None])
        K = torch.empty(nl, nkv, T, hd, dtype=torch.bfloat16)
        V = torch.empty(nl, nkv, T, hd, dtype=torch.bfloat16)
        Q = torch.empty(nl, len(wins), NQ, nh, hd, dtype=torch.bfloat16)
        for li in range(nl):
            K[li], V[li] = pkv[li][0][0].to(torch.bfloat16).cpu(), pkv[li][1][0].to(torch.bfloat16).cpu()
            q = qcap[li].float().view(1, -1, nh, hd).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, q, cos, sin)
            Q[li] = q[0].transpose(0, 1).reshape(len(wins), NQ, nh, hd).to(torch.bfloat16).cpu()
        del fw, pkv, qcap
        torch.save({"K": K, "V": V, "Q": Q, "windows": wins, "nq": NQ, "T": T, "prompt_len": T,
                    "problem_idx": pidx, "gen_seed": None, "model": MODEL, "task": TASK,
                    "text": os.path.basename(v), "created": time.strftime("%Y-%m-%d %H:%M:%S")}, out)
        print(f"capture p{pidx} {os.path.basename(v)}: T={T} windows={wins} {time.time() - t0:.0f}s", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
