"""VisionZip (C3), ported from dvlab-research/VisionZip (Qwen2_5_VL/qwen2_5vl_visionzip.py), applied
to video tokens. Upstream's video branch does not run (it unpacks a 3-tuple from self.visual at
:1865); its image path is ported as is, with one change for video:

  score   attention RECEIVED at the last ViT block (a full-attention block): softmax(q k^T / sqrt d)
          with post-rotary q, k, mean over heads, sum over queries, mean over each 2x2 merge unit,
          window order undone (upstream :595-613). Upstream (FA2 path) computes it without the
          block mask; on a single image that is the same thing. For video we keep the model's own
          mask, i.e. attention within each temporal group (cu_seqlens), = upstream's eager path (:297).
  dominant  global top-k over all visual units, k = int(d * N)            (upstream :1916-1918)
  contextual  c = max(int(0.05 * N), 1) targets at a uniform stride over the non-dominant units
          (in sequence order); every other non-dominant unit joins its most cosine-similar target
          by key (keys: post-rotary, mean over heads and over the merge unit); merged token =
          target embedding + mean of the assigned embeddings              (upstream :1919-1960)
  positions  kept tokens keep their M-RoPE positions (upstream :1964); the text after the video
          continues from the unpruned sequence's next position (upstream rope_deltas, :1899).

Retention r = d + 0.05: r = 0.50 -> d = 0.45 (upstream README), r = 0.25 -> d = 0.20.
"""
import torch
import torch.nn.functional as F

CTX = 0.05


@torch.no_grad()
def qwen_visual_with_scores(visual, pv, thw):
    """Run model.visual; also return per-merged-unit score [N] and key [N, d] in output order."""
    blk = visual.blocks[-1].attn
    cap = {}

    def hook(mod, args, kwargs):
        hs = args[0] if args else kwargs["hidden_states"]
        cap["hs"] = hs
        cap["cu"] = kwargs.get("cu_seqlens", args[1] if len(args) > 1 else None)
        cap["pe"] = kwargs.get("position_embeddings")
    h = blk.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        out = visual(pv, grid_thw=thw)
    finally:
        h.remove()
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision
    hs, cu, (cos, sin) = cap["hs"], cap["cu"], cap["pe"]
    T = hs.shape[0]
    q, k, _ = blk.qkv(hs).reshape(T, 3, blk.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)                    # [T, H, d]
    score = torch.zeros(T, device=hs.device, dtype=torch.float32)
    for i in range(1, len(cu)):
        a, b = int(cu[i - 1]), int(cu[i])
        qq, kk = q[a:b].transpose(0, 1).float(), k[a:b].transpose(0, 1).float()   # [H, n, d]
        w = torch.softmax(qq @ kk.transpose(-1, -2) / qq.shape[-1] ** 0.5, -1)    # [H, n, n]
        score[a:b] = w.mean(0).sum(0)
        del w
    key = k.float().mean(1)                                               # [T, d] mean over heads
    win, _ = visual.get_window_index(thw)
    rev = torch.argsort(win)
    score = score.view(-1, 4).mean(1)[rev]
    key = key.view(-1, 4, key.shape[-1]).mean(1)[rev]
    return out, score, key


@torch.no_grad()
def select_merge(score, key, emb, retain):
    """VisionZip selection on N visual units. Returns (keep_idx sorted [M], new embeddings [M, D])."""
    N = score.shape[0]
    dom = int((retain - CTX) * N)
    ctx = max(int(CTX * N), 1)
    top = torch.topk(score, dom).indices
    mask = torch.zeros(N, dtype=torch.bool, device=score.device); mask[top] = True
    rest = (~mask).nonzero(as_tuple=True)[0]                                 # non-dominant, in order
    kn = key[rest] / key[rest].norm(dim=-1, keepdim=True)
    step = max(1, len(rest) // ctx)
    tgt = torch.arange(0, len(rest), step, device=score.device)[:ctx]
    is_t = torch.zeros(len(rest), dtype=torch.bool, device=score.device); is_t[tgt] = True
    others = (~is_t).nonzero(as_tuple=True)[0]
    assign = (kn[others] @ kn[tgt].T).argmax(1)                              # [n_others]
    D = emb.shape[-1]
    agg = torch.zeros(len(tgt), D, device=emb.device, dtype=torch.float32)
    agg.index_add_(0, assign, emb[rest[others]].float())
    cnt = torch.bincount(assign, minlength=len(tgt)).clamp(min=1).unsqueeze(-1)
    merged = (emb[rest[tgt]].float() + agg / cnt).to(emb.dtype)
    new = emb.clone()
    new[rest[tgt]] = merged
    mask[rest[tgt]] = True
    keep = mask.nonzero(as_tuple=True)[0]
    return keep, new[keep]
