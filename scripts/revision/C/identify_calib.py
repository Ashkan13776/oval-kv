"""Identify the six LLaVA-Video-178K calibration clips behind results/vision/eta/llava-video-7b_VCAL.json.

The clip names were not recorded (only "2_3_m_youtube_v0_1, archive 98": 34 clips). The stats
file stores, per split, L[layer][head][eta] = sum_clips d2 / sum_clips u2 (pooled relative
output error). For every candidate clip this script reproduces vision/calibrate_capture.py's
prefill capture (LLaVA-Video, same prompt, same 64 frames, same 3 windows of 96 queries) and
analysis/eta_rules_calibrate.py's window_stats at eta = 0 and 1 on a few layers (same config as
the committed run), and stores the per-clip sums d2 and u2. `--match` then searches the
subsets consistent with the committed per-clip window signature and reports the one whose
pooled ratios reproduce the committed numbers.

    python scripts/revision/C/identify_calib.py --run            # GPU, per-clip sums -> logs/revc/calib_id/
    python scripts/revision/C/identify_calib.py --match
"""
import argparse, glob, importlib.util, itertools, json, os, sys, time
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, os.path.join(REPO, "vision"))
sys.path.insert(0, os.path.join(REPO, "analysis"))
REF = os.path.join(REPO, "results", "vision", "eta", "llava-video-7b_VCAL.json")
OUT = os.path.join(REPO, "logs", "revc", "calib_id")
SRC = os.path.expanduser("~/vmme/calib_src")
LAYERS = [2, 14, 26]


def run():
    import calibrate_capture as CC
    import run_videomme as RV
    spec = importlib.util.spec_from_file_location("erc", os.path.join(REPO, "analysis", "eta_rules_calibrate.py"))
    ERC = importlib.util.module_from_spec(spec); sys.argv = sys.argv[:1]; spec.loader.exec_module(ERC)
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from transformers import Qwen2ForCausalLM
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    ref = json.load(open(REF))
    args = SimpleNamespace(**ref["config"])
    os.makedirs(OUT, exist_ok=True)
    tok, model, proc = RV.load_llava(SimpleNamespace())
    lm = model.model
    nl, nh, nkv = len(lm.layers), model.config.num_attention_heads, model.config.num_key_value_heads
    hd = model.config.hidden_size // nh; g = nh // nkv; D = model.config.hidden_size
    dev = torch.device("cuda:0")
    Wh = {li: lm.layers[li].self_attn.o_proj.weight.float().view(D, nkv, g, hd) for li in LAYERS}
    ERC.PB = ERC.PB if hasattr(ERC, "PB") else None
    Gh = {li: ERC.PB.publish_metric(lm.layers[li].self_attn.o_proj.weight.float(), nkv, hd, g) for li in LAYERS}
    for v in sorted(glob.glob(os.path.join(SRC, "**", "*.mp4"), recursive=True)):
        name = os.path.basename(v)
        fo = os.path.join(OUT, name + ".json")
        if os.path.exists(fo):
            continue
        t0 = time.time()
        arr, ftime, vtime = CC.frames(v)
        pix = proc.preprocess(list(arr), return_tensors="pt")["pixel_values"].to("cuda:0", torch.bfloat16)
        ti = (f"The video lasts for {vtime:.2f} seconds, and {len(arr)} frames are uniformly sampled from it. "
              f"These frames are located at {ftime}.Please answer the following questions related to this video.")
        conv = conv_templates["qwen_1_5"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + ti)
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), tok, IMAGE_TOKEN_INDEX, return_tensors="pt")[None].cuda()
        p = int((ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0])
        pre = tok("\n" + ti).input_ids
        k = len(pre)
        while k > 0 and ids[0, p + 1:p + 1 + k].tolist() != pre[:k]:
            k -= 1
        ids = ids[:, :p + 1 + k]
        with torch.no_grad():
            model.get_vision_tower().to("cuda:0")
            _, _, _, _, emb, _ = model.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, [pix],
                                                                             modalities=["video"])
        T = emb.shape[1]
        wins = []
        for w in range(CC.NWIN):
            L = CC.SINK + ((T - w * CC.STRIDE - CC.SINK) // CC.PAGE) * CC.PAGE
            if L - CC.SINK - CC.RECENT >= CC.PAGE and L - CC.NQ >= 0:
                wins.append(L)
        wins = sorted(set(wins))
        qpos_all = torch.cat([torch.arange(L - CC.NQ, L) for L in wins]).cuda()
        qcap = {}
        hooks = [lm.layers[li].self_attn.q_proj.register_forward_hook(
                    lambda m, i, o, li=li: qcap.__setitem__(li, o[0, qpos_all].detach())) for li in LAYERS]
        with torch.no_grad():
            fw = Qwen2ForCausalLM.forward(model, inputs_embeds=emb, use_cache=True, num_logits_to_keep=1)
        for h in hooks:
            h.remove()
        pkv = fw.past_key_values
        cos, sin = lm.rotary_emb(qcap[LAYERS[0]][None].float(), qpos_all[None])
        acc = ERC.Acc(nl, nkv, 2)
        for li in LAYERS:
            Kf = pkv[li][0][0].to(torch.bfloat16).float(); Vf = pkv[li][1][0].to(torch.bfloat16).float()
            q = qcap[li].float().view(1, -1, nh, hd).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, q, cos, sin)
            Q = q[0].transpose(0, 1).reshape(len(wins), CC.NQ, nh, hd).to(torch.bfloat16)
            for wi, L in enumerate(wins):
                qpos = torch.arange(L - CC.NQ, L, device=dev)
                qq = Q[wi].float().permute(1, 0, 2).reshape(nkv, g, CC.NQ, hd)
                ERC.window_stats(acc, "test", li, qq, Kf[:, :L], Vf[:, :L], L, qpos, Wh[li], Gh[li], [0.0, 1.0], args)
        res = dict(clip=name, T=T, windows=wins,
                   u2={li: acc.u2["test"][li].tolist() for li in LAYERS},
                   d2={li: acc.d2["test"][li].tolist() for li in LAYERS})
        json.dump(res, open(fo, "w"))
        del fw, pkv, qcap
        torch.cuda.empty_cache()
        print(f"{name}: T={T} windows={wins} {time.time() - t0:.0f}s", flush=True)


def match():
    ref = json.load(open(REF))
    Lref = {sp: np.array(ref["loss_grid"][sp]) for sp in ("calib", "test")}          # [layer][head][21]
    sig = {int(p): max(w) for p, w in ref["windows"].items()}
    C = {}
    for f in glob.glob(os.path.join(OUT, "*.json")):
        r = json.load(open(f))
        C[r["clip"]] = dict(top=max(r["windows"]), u2={int(k): np.array(v) for k, v in r["u2"].items()},
                            d2={int(k): np.array(v) for k, v in r["d2"].items()})
    print(f"{len(C)} candidate clips with per-clip sums")
    for sp, pids in (("test", [p for p, s in ref["splits"].items() if s == "test"]),
                     ("calib", [p for p, s in ref["splits"].items() if s == "calib"])):
        need = sorted(sig[int(p)] for p in pids)
        best = []
        for combo in itertools.combinations(sorted(C), len(pids)):
            if sorted(C[c]["top"] for c in combo) != need:
                continue
            err = 0.0
            for li in LAYERS:
                d2 = sum(C[c]["d2"][li] for c in combo); u2 = sum(C[c]["u2"][li] for c in combo)
                L = d2 / np.maximum(u2[:, None], 1e-300)
                err = max(err, float(np.abs(L - Lref[sp][li][:, [0, -1]]).max() / np.abs(Lref[sp][li][:, [0, -1]]).max()))
            best.append((err, combo))
        best.sort()
        print(f"{sp}: {len(best)} signature-consistent subsets; best 3:")
        for e, c in best[:3]:
            print(f"   max rel. deviation {e:.2e}  {c}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--match", action="store_true")
    a = ap.parse_args()
    if a.run:
        run()
    if a.match:
        match()
