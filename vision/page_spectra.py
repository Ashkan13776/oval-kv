"""Per-page explained variance: visual-token pages vs text pages, LLaVA-Video-7B.

For every 32-token page (layers >= 1, every KV head) of the dense prefill cache:
  key energy captured at rank 8 of the centered post-RoPE keys,
  value energy captured at rank 16 of the centered values,
i.e. sum_{i<=r} s_i^2 / sum_i s_i^2 of the page minus its mean (the quantity the
OVAL bases keep). Visual pages: the video span of 20 Video-MME videos (64
frames; pages fully inside the span). Text pages: one ~10K-token LongBench v2
document fed to the same model with the same chat template, no video.

    python vision/page_spectra.py         # -> results/vision/page_spectra.{json,npz}; plot with vision/plot_spectra.py
"""
import json, os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_videomme as RV

PAGE, RK, RV_ = 32, 8, 16
OUT_JSON = os.path.join(HERE, "..", "results", "vision", "page_spectra.json")
OUT_PDF = os.path.join(HERE, "..", "figures", "page_spectra.pdf")


def energy(pages, r):
    """pages [N, page, d] float32 -> fraction of centered energy in the top r directions [N]."""
    x = pages - pages.mean(1, keepdim=True)
    # squared singular values = eigenvalues of the page x page Gram (batched svd of
    # 32 x 128 has no fast CUDA path; 32 x 32 eigvalsh does)
    s2 = torch.linalg.eigvalsh(x @ x.transpose(-1, -2)).flip(-1).clamp_min(0)
    return (s2[:, :r].sum(-1) / s2.sum(-1).clamp_min(1e-30)).cpu().numpy()


def collect(pkv, lo, hi):
    ek, ev = [], []
    for li in range(1, len(pkv)):                 # layer 0 stays dense in the harness
        K, V = pkv[li][0][0].float(), pkv[li][1][0].float()      # [H, T, d]
        n = (hi - lo) // PAGE
        for h in range(K.shape[0]):
            kp = K[h, lo:lo + n * PAGE].reshape(n, PAGE, -1)
            vp = V[h, lo:lo + n * PAGE].reshape(n, PAGE, -1)
            ek.append(energy(kp, RK)); ev.append(energy(vp, RV_))
    return np.concatenate(ek), np.concatenate(ev)


@torch.no_grad()
def main():
    import pandas as pd
    from transformers import Qwen2ForCausalLM
    from llava.constants import IMAGE_TOKEN_INDEX
    class A: frames = 64
    tok, model, proc = RV.load_llava(A())
    fwd = lambda **k: Qwen2ForCausalLM.forward(model, use_cache=True, num_logits_to_keep=1, **k)
    df = pd.read_parquet(RV.ANN)
    vids = df.drop_duplicates("videoID")
    vids = vids.iloc[np.linspace(0, len(vids) - 1, 20, dtype=int)]   # spread over short/medium/long
    vis_k, vis_v = [], []
    for _, r in vids.iterrows():
        doc = r.to_dict(); doc["options"] = list(doc["options"])
        ids, pix, _ = RV.build_inputs(doc, A, tok, model, proc)
        _, _, _, _, emb, _ = model.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, [pix],
                                                                         modalities=["video"])
        p = int((ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0])
        n_vis = emb.shape[1] - (ids.shape[1] - 1)
        pkv = fwd(inputs_embeds=emb).past_key_values
        lo = -(-p // PAGE) * PAGE                       # pages fully inside the video span
        k, v = collect(pkv, lo, p + n_vis)
        vis_k.append(k); vis_v.append(v)
        del pkv; torch.cuda.empty_cache()
        print(doc["videoID"], doc["duration"], f"key r8 {k.mean():.3f} value r16 {v.mean():.3f}", flush=True)
    # text: one LongBench v2 document of >= 10K tokens, truncated to 10K
    from datasets import load_dataset
    lb = load_dataset("THUDM/LongBench-v2", split="train")
    from llava.conversation import conv_templates
    for ex in lb:
        t = tok(ex["context"]).input_ids
        if len(t) >= 10240:
            ctx = tok.decode(t[:10240]); src = ex["_id"]; break
    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], ctx); conv.append_message(conv.roles[1], None)
    ids = tok(conv.get_prompt(), return_tensors="pt").input_ids.cuda()
    pkv = fwd(input_ids=ids).past_key_values
    tk, tv = collect(pkv, 128, ids.shape[1] - 128)       # skip the sink-sized head and the tail
    vk, vv = np.concatenate(vis_k), np.concatenate(vis_v)
    q = lambda a: dict(mean=float(a.mean()), q25=float(np.quantile(a, .25)), median=float(np.median(a)),
                       q75=float(np.quantile(a, .75)), n_pages=int(a.size))
    res = {"model": RV.MODEL, "page": PAGE, "key_rank": RK, "value_rank": RV_,
           "videos": vids.videoID.tolist(), "text_source": f"THUDM/LongBench-v2 {src}, first 10240 tokens",
           "visual": {"key": q(vk), "value": q(vv)}, "text": {"key": q(tk), "value": q(tv)}}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(res, open(OUT_JSON, "w"), indent=1)
    np.savez_compressed(OUT_JSON.replace(".json", ".npz"), vis_key=vk, vis_value=vv, text_key=tk, text_value=tv)
    print(json.dumps(res, indent=1))
    print("plot: python3 vision/plot_spectra.py")

if __name__ == "__main__":
    main()
