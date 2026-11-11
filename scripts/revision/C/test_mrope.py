"""Unit tests of the Qwen2.5-VL M-RoPE integration (env ~/qwenvl; GPU).

  1. mrope_cos_sin + rotate-half (run_qwen25vl.py) == HF apply_multimodal_rotary_pos_emb, bitwise,
     on the real 3-D positions of a Video-MME prefill.
  2. Decode tokens (t = h = w = p): M-RoPE == 1-D RoPE at p (HF), bitwise; and flashinfer's
     apply_rope_inplace at offset p (what the harness runs) agrees to bf16 rounding.
  3. The harness's cached prefill keys (run_qwen25vl.py --method full with QWEN_DUMP_KEYS) ==
     HF Qwen2.5-VL's post-M-RoPE keys of the same prefill (layers 1, 14, 27).
  4. Brief's check: page logits rebuilt from OVAL page bases at r = 31 on those post-M-RoPE
     keys match the exact logits (query: the first decode token, rotated at its text position).

    ~/qwenvl/bin/python scripts/revision/C/test_mrope.py VIDEO_ID HARNESS_KEYS.pt
"""
import importlib.util, os, sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qwen_prep as QP
from run_qwen25vl import mrope_cos_sin, _rotate_half

spec = importlib.util.spec_from_file_location("page_basis", os.path.join(HERE, "..", "..", "..", "oval", "page_basis.py"))
PB = importlib.util.module_from_spec(spec); spec.loader.exec_module(PB)


def main(vid, harness_keys):
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as M
    ok = True
    c = torch.load(QP.emb_path(vid, "videomme"), weights_only=True)
    proc, model = QP.load_model()
    dev = model.device
    pos = c["pos"].long().to(dev)                                         # (3, L)
    L = pos.shape[1]
    # 1. prefill M-RoPE vs HF
    q = torch.randn(1, 28, L, 128, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, 4, L, 128, device=dev, dtype=torch.bfloat16)
    rot = model.model.rotary_emb
    cos_hf, sin_hf = rot(q, pos[:, None])
    qh, kh = M.apply_multimodal_rotary_pos_emb(q, k, cos_hf, sin_hf, model.config.rope_scaling["mrope_section"])
    cos, sin = mrope_cos_sin(pos.cpu(), theta=model.config.rope_theta, device=dev)
    qm = q * cos[None, None] + _rotate_half(q) * sin[None, None]
    km = k * cos[None, None] + _rotate_half(k) * sin[None, None]
    t1 = torch.equal(qh, qm) and torch.equal(kh, km)
    print(f"1. prefill M-RoPE == HF apply_multimodal_rotary_pos_emb (bitwise, L={L}): {t1}")
    ok &= t1
    # 2. decode: t = h = w = p
    p = c["next_pos"] + 37
    p3 = torch.full((3, 1, 1), p, device=dev)
    q1 = torch.randn(1, 28, 1, 128, device=dev, dtype=torch.bfloat16)
    k1 = torch.randn(1, 4, 1, 128, device=dev, dtype=torch.bfloat16)
    c3, s3 = rot(q1, p3)
    qa, ka = M.apply_multimodal_rotary_pos_emb(q1, k1, c3, s3, model.config.rope_scaling["mrope_section"])
    from transformers.models.qwen2 import modeling_qwen2 as Q2
    inv = 1.0 / (model.config.rope_theta ** (torch.arange(0, 128, 2, device=dev).float() / 128))
    fr = p * inv; e = torch.cat((fr, fr)); c1, s1 = e.cos().to(torch.bfloat16), e.sin().to(torch.bfloat16)
    qb = q1 * c1 + _rotate_half(q1) * s1; kb = k1 * c1 + _rotate_half(k1) * s1
    t2 = torch.equal(qa, qb) and torch.equal(ka, kb)
    import flashinfer
    qf = q1.transpose(1, 2).contiguous().view(1, 28, 128).clone(); kf = k1.transpose(1, 2).contiguous().view(1, 4, 128).clone()
    flashinfer.apply_rope_inplace(qf, kf, torch.tensor([0, 1], device=dev, dtype=torch.int32),
                                  torch.tensor([p], device=dev, dtype=torch.int32), rope_scale=1.0,
                                  rope_theta=model.config.rope_theta)
    dq = (qf.float() - qa[0, :, 0].float()).abs().max().item(); sc = qa.float().abs().max().item()
    print(f"2. decode M-RoPE (t=h=w={p}) == 1-D RoPE (bitwise): {t2}; flashinfer max |diff| {dq:.4f} (|q| max {sc:.2f}, bf16 ulp ~{sc/128:.3f})")
    ok &= t2 and dq <= sc / 64
    # 3. harness cached keys vs HF post-M-RoPE keys
    with torch.no_grad():
        out = model.model(inputs_embeds=c["emb"][None].to(dev), position_ids=pos[:, None], use_cache=True)
    pkv = out.past_key_values
    H = torch.load(harness_keys, weights_only=True)
    for l, kh_ in H.items():
        hf = pkv.key_cache[l][0].transpose(0, 1).float().cpu()             # (L, 4, 128)
        hk = kh_[0].float()
        rel = ((hf - hk).norm() / hf.norm()).item()
        cosm = torch.nn.functional.cosine_similarity(hf.flatten(1), hk.flatten(1), dim=1).min().item()
        good = rel < 0.02
        print(f"3. layer {l}: harness cached keys vs HF post-M-RoPE keys: rel. error {rel:.2e}, min per-token cosine {cosm:.5f} -> {'OK' if good else 'FAIL'}")
        ok &= good
    # 4. r = 31 page logits on the HF post-M-RoPE keys
    l = 14
    K = pkv.key_cache[l][0].transpose(0, 1).double().cpu()                # (L, 4, 128)
    V = pkv.value_cache[l][0].transpose(0, 1).double().cpu()
    npg = L // 32
    Kp = K[:npg * 32].view(1, npg, 32, 4, 128); Vp = V[:npg * 32].view(1, npg, 32, 4, 128)
    W = model.model.layers[l].self_attn.o_proj.weight.double().cpu()
    Gh = PB.publish_metric(W, 4, 128, 7)
    with torch.no_grad():
        qv = model.model.layers[l].self_attn.q_proj(out.last_hidden_state[:, -1:]).view(1, 1, 28, 128)
    cq, sq = rot(qv.transpose(1, 2), torch.full((3, 1, 1), c["next_pos"], device=dev))
    qr, _ = M.apply_multimodal_rotary_pos_emb(qv.transpose(1, 2), qv.transpose(1, 2)[:, :4], cq, sq,
                                              model.config.rope_scaling["mrope_section"])
    qr = qr[0, :, 0].double().cpu()                                       # (28, 128)
    for eta in (0.0, 0.5):           # eta = 1 alone is not full-rank (pure output metric)
        mu, basis, coef = PB.build_summary(Kp, Vp, Gh, eta, 31)           # [B,P,H,d] [B,P,H,d,r] [B,P,H,page,r]
        Kr = mu.unsqueeze(3) + coef @ basis.transpose(-1, -2)              # [B,P,H,page,d]
        Kr = Kr.permute(0, 1, 3, 2, 4).reshape(npg * 32, 4, 128).double()
        g = 28 // 4
        ex = torch.einsum("hd,thd->ht", qr, K[:npg * 32].repeat_interleave(g, 1)) / 128 ** 0.5
        rb = torch.einsum("hd,thd->ht", qr, Kr.repeat_interleave(g, 1)) / 128 ** 0.5
        err = (ex - rb).abs().max().item(); scale = ex.abs().max().item()
        good = err < 1e-3 * max(scale, 1)                                # build_summary runs in fp32
        print(f"4. eta={eta}: r=31 rebuilt page logits vs exact, layer {l}, {npg} pages: max |diff| {err:.2e} (max |logit| {scale:.1f}) -> {'OK' if good else 'FAIL'}")
        ok &= good
    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
