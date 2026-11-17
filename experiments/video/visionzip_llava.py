"""VisionZip (C3) for LLaVA-Video-7B-Qwen2 -- OUR ADAPTATION: upstream VisionZip covers CLIP
(scores from the CLS row) and Qwen2.5-VL images, not SigLIP (no CLS) or LLaVA-Video. We use the
CLS-free rule of upstream's Qwen2.5-VL port (attention RECEIVED: head-mean, sum over queries) at the
SigLIP layer LLaVA-Video reads (its tower drops the last layer, so this is the last remaining layer
= upstream CLIP's attentions[-2] choice), per frame (SigLIP encodes frames independently).
Scores and keys (k_proj output, head-mean) on the 27 x 27 patch grid are pooled to the 14 x 14 grid
with LLaVA-Video's own get_2dPool (bilinear), the same pooling that makes the visual tokens.

Selection over the 64 x 196 patch tokens (global top-k, as upstream's Qwen port); the 896 newline
tokens (one per grid row) are structural and always kept. Budgets as fractions of ALL 13,440 visual
tokens: contextual = 5% (672 tokens), dominant = retain * 13,440 - 896 - 672 (50%: 5,152; 25%: 1,792),
so the kept count is exactly 50% / 25% (6,720 / 3,360). Contextual merge = upstream (uniform-stride
targets over the non-dominant tokens, cosine on keys, target + mean of assigned). As in upstream
LLaVA, the pruned sequence is fed as is (contiguous positions).

    install(model, retain)   # wraps model.prepare_inputs_labels_for_multimodal
"""
import torch

from visionzip import CTX


@torch.no_grad()
def frame_scores(model, pix):
    vt = model.get_vision_tower()
    enc = vt.vision_tower.vision_model.encoder
    last = enc.layers[-1].self_attn
    H = last.num_heads
    scores, keys = [], []
    for i in range(0, pix.shape[0], 16):
        cap = {}
        h = last.k_proj.register_forward_hook(lambda m, a, o: cap.__setitem__("k", o))
        try:
            out = vt.vision_tower(pix[i:i + 16].to(device=vt.device, dtype=vt.dtype), output_attentions=True)
        finally:
            h.remove()
        att = out.attentions[-1].float()                       # [f, H, 729, 729]
        scores.append(att.mean(1).sum(1))                      # [f, 729] attention received
        k = cap["k"].float()                                   # [f, 729, H*d]
        keys.append(k.view(k.shape[0], k.shape[1], H, -1).mean(2))
        del out, att
    return torch.cat(scores), torch.cat(keys)                  # [F, 729], [F, 729, d]


def install(model, retain):
    orig = model.prepare_inputs_labels_for_multimodal
    hw = model.get_vision_tower().num_patches_per_side          # 27

    def wrapped(ids, pos, am, pkv, labels, images, **kw):
        out = list(orig(ids, pos, am, pkv, labels, images, **kw))
        emb = out[4]
        from llava.constants import IMAGE_TOKEN_INDEX
        pix = images[0]
        F = pix.shape[0]
        sc, ky = frame_scores(model, pix)
        sc = model.get_2dPool(sc[..., None]).squeeze(-1)        # [F, 196]
        ky = model.get_2dPool(ky)                               # [F, 196, d]
        g = int(round(sc.shape[1] ** 0.5))                      # 14
        p = int((ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0])
        n_vis = emb.shape[1] - (ids.shape[1] - 1)
        assert n_vis == F * g * (g + 1), (n_vis, F, g)
        f_, r_, c_ = torch.meshgrid(torch.arange(F), torch.arange(g), torch.arange(g), indexing="ij")
        tok = (p + f_ * g * (g + 1) + r_ * (g + 1) + c_).flatten().to(emb.device)      # patch token positions
        score, key = sc.flatten(), ky.reshape(-1, ky.shape[-1])
        N = score.shape[0]
        ctx = max(int(CTX * n_vis), 1)
        dom = int(retain * n_vis) - (n_vis - N) - ctx
        top = torch.topk(score, dom).indices
        mask = torch.zeros(N, dtype=torch.bool, device=score.device); mask[top] = True
        rest = (~mask).nonzero(as_tuple=True)[0]
        kn = key[rest] / key[rest].norm(dim=-1, keepdim=True)
        step = max(1, len(rest) // ctx)
        tgt = torch.arange(0, len(rest), step, device=score.device)[:ctx]
        is_t = torch.zeros(len(rest), dtype=torch.bool, device=score.device); is_t[tgt] = True
        others = (~is_t).nonzero(as_tuple=True)[0]
        assign = (kn[others] @ kn[tgt].T).argmax(1)
        e = emb[0]
        agg = torch.zeros(len(tgt), e.shape[-1], device=e.device, dtype=torch.float32)
        agg.index_add_(0, assign, e[tok[rest[others]]].float())
        cnt = torch.bincount(assign, minlength=len(tgt)).clamp(min=1).unsqueeze(-1)
        e = e.clone()
        e[tok[rest[tgt]]] = (e[tok[rest[tgt]]].float() + agg / cnt).to(e.dtype)
        mask[rest[tgt]] = True
        drop = torch.zeros(e.shape[0], dtype=torch.bool, device=e.device)
        drop[tok[~mask]] = True
        out[4] = e[~drop][None]
        return tuple(out)
    model.prepare_inputs_labels_for_multimodal = wrapped
