"""Capture real decode-time K/V/Q from a reasoning trace, all layers.

The codec analyses need keys as they actually are during long generation, not
prefill keys from a summarisation prompt: reasoning decode keys are markedly
more low-rank, and measuring on the wrong ones misstates the codec badly.

One layer is not enough either -- end-to-end accuracy is gated by the WORST
layers, and the per-layer spread is wide (at rank 8, 9% on the best layer and
37% on the worst).
"""
import argparse, json, os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from paths import ROOT, FREEKV

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
ap.add_argument("--ngen", type=int, default=4200)
ap.add_argument("--out", default=str(ROOT / "logs/kv_capture.pt"))
ap.add_argument("--dataset", default=str(FREEKV / "accuracy/eval/reasoning/datasets/math50.jsonl"))
A = ap.parse_args()

if os.path.exists(A.out):
    print(f"{A.out} exists; delete it to recapture"); raise SystemExit

tok = AutoTokenizer.from_pretrained(A.model, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    A.model, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()

prob = json.loads(open(A.dataset).readline())
msgs = [{"role": "user", "content": prob["problem"] +
         " Please reason step by step, and put your final answer within \\boxed{}.\n"}]
ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
          return_tensors="pt").input_ids.cuda()
print(f"prompt {ids.shape[-1]} tokens; generating {A.ngen}", flush=True)
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=A.ngen, min_new_tokens=A.ngen,
                         do_sample=True, temperature=0.6, top_p=0.95,
                         pad_token_id=tok.eos_token_id)
print(f"sequence {out.shape[-1]} tokens; capturing all layers", flush=True)

cap = {}
for li, layer in enumerate(model.model.layers):
    lay = layer.self_attn; lay._li = li; lay._orig = lay.forward
    def mk(lay):
        def fwd(hidden_states, *a, **kw):
            nh = lay.config.num_attention_heads
            nkv = lay.config.num_key_value_heads
            hd = lay.config.hidden_size // nh
            b, t, _ = hidden_states.shape
            cap[lay._li] = (lay.k_proj(hidden_states).view(b, t, nkv, hd).float().cpu(),
                            lay.v_proj(hidden_states).view(b, t, nkv, hd).float().cpu(),
                            lay.q_proj(hidden_states).view(b, t, nh, hd)[:, -1].float().cpu())
            return lay._orig(hidden_states, *a, **kw)
        return fwd
    lay.forward = mk(lay)
with torch.no_grad():
    model(out)
Path(A.out).parent.mkdir(parents=True, exist_ok=True)
torch.save(cap, A.out)
print(f"saved {len(cap)} layers to {A.out}")
