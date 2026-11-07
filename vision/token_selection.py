"""Which visual tokens does OVAL actually keep? -> results/vision/token_selection.{json,npz}

For a few Video-MME Long questions, under the query-agnostic protocol:
  prefill = system prompt + 64 frames + time instruction (dense), then the
  question is streamed. At every streamed position each layer scores all
  video-span pages with its rank-8 OVAL key records (per-head eta from the
  calibrated table) and keeps the top `budget/page` pages per KV head.

We record, per (layer, KV head, streamed position), which pages were selected,
and map each 32-token page onto the 64 frames it covers (210 tokens per frame).
The result is a per-frame retention rate: of all (frame token x streamed
position x KV head) opportunities, the fraction where that token sat inside a
kept page. A uniform selection would retain page_budget/pages ~ 13% of every
frame, so values above that mark frames the scorer prefers.

    python vision/token_selection.py            # then: python3 vision/plot_token_selection.py

DOCUMENTED SIMPLIFICATIONS (same spirit as the eta calibration):
  - Pages are scored with the CURRENT query, not spec_ret's previous query plus
    the cosine correction.
  - The streamed queries are taken from one dense forward over the whole prompt,
    so the hidden states behind them never attended to a compressed cache. This
    measures where OVAL's scoring points, not a re-simulation of the full run.
  - Layer 0 is dense in the harness (`skip_layer 1`) and is excluded.
"""
import json, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_videomme as RV

PAGE, RANK, BUDGET, SINK, RECENT = 32, 8, 1792, 128, 128
N_VIDEOS = 6
ETA_TABLE = os.path.join(HERE, "..", "results", "vision", "eta", "tables", "llava-video-7b_VCAL_r1.json")
OUT_JSON = os.path.join(HERE, "..", "results", "vision", "token_selection.json")


def _load_oval():
    import importlib.util
    s = importlib.util.spec_from_file_location("page_basis", os.path.join(HERE, "..", "oval", "page_basis.py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


@torch.no_grad()
def main():
    import pandas as pd
    from transformers import Qwen2ForCausalLM
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    from llava.constants import IMAGE_TOKEN_INDEX
    PB = _load_oval()

    class A: frames = 64
    tok, model, proc = RV.load_llava(A())
    lm = model.model
    nl = len(lm.layers)
    nh, nkv = model.config.num_attention_heads, model.config.num_key_value_heads
    hd = model.config.hidden_size // nh
    g = nh // nkv
    page_budget = BUDGET // PAGE
    eta_tab = json.load(open(ETA_TABLE))
    eta_all = torch.tensor(eta_tab["eta"], dtype=torch.float64)
    Gh = [PB.publish_metric(l.self_attn.o_proj.weight.float().cuda(), nkv, hd, g) for l in lm.layers]

    df = pd.read_parquet(RV.ANN)
    longs = df[df.duration == "long"].drop_duplicates("videoID")
    picks = longs.iloc[np.linspace(0, len(longs) - 1, N_VIDEOS, dtype=int)]

    ret = np.zeros((nl, A.frames))      # frame tokens kept, per (layer, frame)
    den_f = np.zeros(A.frames)          # frame tokens offered, summed over decisions
    per_video = []                      # [n_videos, frames] retention, pooled over layers
    meta = []
    for vi, (_, r) in enumerate(picks.iterrows()):
        doc = r.to_dict(); doc["options"] = list(doc["options"])
        ids, pix, _ = RV.build_inputs(doc, A, tok, model, proc)
        n_ctx_text = RV.build_inputs.n_ctx_text
        vt = model.get_vision_tower(); vt.to("cuda:0")
        _, _, _, _, emb, _ = model.prepare_inputs_labels_for_multimodal(
            ids, None, None, None, None, [pix], modalities=["video"])
        vt.to("cpu"); torch.cuda.empty_cache()
        p = int((ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0])
        T = emb.shape[1]
        n_vis = T - (ids.shape[1] - 1)
        vis_lo, vis_hi = p, p + n_vis                       # video span in cache positions
        n_prefill = p + n_vis + n_ctx_text                  # stream protocol cut
        qpos = torch.arange(n_prefill, T).cuda()            # the streamed question positions
        if len(qpos) < 4:
            print(f"skip {doc['videoID']}: nothing streamed"); continue

        qcap = {}
        hooks = [l.self_attn.q_proj.register_forward_hook(
                    lambda m, i, o, li=li: qcap.__setitem__(li, o[0, qpos].detach()))
                 for li, l in enumerate(lm.layers)]
        t0 = time.time()
        fw = Qwen2ForCausalLM.forward(model, inputs_embeds=emb, use_cache=True, num_logits_to_keep=1)
        for h in hooks:
            h.remove()
        pkv = fw.past_key_values
        cos, sin = lm.rotary_emb(qcap[0][None].float(), qpos[None])

        # pages fully inside the video span, and before the recent window
        lo = -(-vis_lo // PAGE) * PAGE
        hi = min(vis_hi, T - RECENT)
        hi = lo + ((hi - lo) // PAGE) * PAGE
        P = (hi - lo) // PAGE
        nq = len(qpos)
        # token -> frame index (210 tokens per frame inside the video span)
        per_frame = n_vis / A.frames
        tok_frame = np.clip(((np.arange(lo, hi) - vis_lo) / per_frame).astype(int), 0, A.frames - 1)
        # W[page, frame] = how many of the page's 32 tokens fall in that frame;
        # tf[frame] = how many of the frame's tokens are inside the selectable span
        W = np.zeros((P, A.frames))
        for i in range(P):
            f, c = np.unique(tok_frame[i * PAGE:(i + 1) * PAGE], return_counts=True)
            W[i, f] = c
        tf = W.sum(0)

        vret = np.zeros(A.frames); vden = np.zeros(A.frames)
        for li in range(1, nl):
            K = pkv[li][0][0].float(); V = pkv[li][1][0].float()        # [H, T, d]
            Kp = K[:, lo:hi].reshape(nkv, P, PAGE, hd).permute(1, 2, 0, 3)[None].contiguous()
            Vp = V[:, lo:hi].reshape(nkv, P, PAGE, hd).permute(1, 2, 0, 3)[None].contiguous()
            mu, bs, cf = PB.build_summary(Kp, Vp, Gh[li], eta_all[li].cuda(), RANK)
            q = qcap[li].float().view(1, nq, nh, hd).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, q, cos, sin)
            qw = q[0].transpose(0, 1).reshape(nq, 1, nh, hd)            # [nq, 1, nh, d]
            sc = PB.oval_sel(qw, bs.expand(nq, -1, -1, -1, -1), cf.expand(nq, -1, -1, -1, -1),
                             mu.expand(nq, -1, -1, -1), "avgSM", nh, nkv)        # [nq, H, P]
            k = min(page_budget, P)
            sel = torch.zeros_like(sc, dtype=torch.bool).scatter_(-1, sc.topk(k, -1).indices, True)
            cnt = sel.cpu().numpy().sum(axis=(0, 1)).astype(float)      # per page, over nq x H
            # tokens of each frame that were inside a kept page, over all decisions
            ret[li] += cnt @ W
            vret += cnt @ W
            vden += nq * nkv * tf
            if li == 1: den_f += nq * nkv * tf
            del K, V, Kp, Vp, mu, bs, cf, sc, sel
        per_video.append(vret / np.maximum(vden, 1))
        meta.append(dict(videoID=doc["videoID"], question_id=doc["question_id"], T=int(T),
                         n_visual=int(n_vis), n_streamed=int(nq), pages=int(P)))
        print(f"[{vi+1}/{len(picks)}] {doc['videoID']} T={T} pages={P} streamed={nq} "
              f"{time.time()-t0:.0f}s", flush=True)
        del fw, pkv, qcap
        torch.cuda.empty_cache()

    frac = ret[1:] / np.maximum(den_f[None, :], 1)     # layers >= 1; fraction of each frame's tokens kept
    res = dict(model=RV.MODEL, page=PAGE, rank=RANK, budget=BUDGET, page_budget=page_budget,
               sink=SINK, recent=RECENT, frames=A.frames, eta_table_hash=eta_tab["hash"],
               n_videos=len(meta), videos=meta,
               mean_retention=float(frac.mean()),
               note="per-frame retention of OVAL page selection, layers >= 1; "
                    "scored with the current query on dense states (see module docstring)")
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(res, open(OUT_JSON, "w"), indent=1)
    np.savez_compressed(OUT_JSON.replace(".json", ".npz"), retention=frac,
                        per_video=np.array(per_video),
                        video_ids=np.array([m["videoID"] for m in meta]))
    print(json.dumps({k: v for k, v in res.items() if k != "videos"}, indent=1))
    print("plot: python3 vision/plot_token_selection.py")


if __name__ == "__main__":
    main()
