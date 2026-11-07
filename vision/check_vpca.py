"""Check oval/page_basis.py's token-domain value PCA against the previous channel-domain
code (git HEAD~ version of the file) on text pages from LongBench v2, CPU only.

Text: the first 128 + 20*32 tokens of a LongBench v2 document through LLaVA-Video-7B's
language model (same chat template as the runs); 20 pages after a 128-token sink,
every layer and KV head. Compares, for r_v in {8, 16}: the page mean, the rebuilt
values (T D^T), and the value reconstruction error.

    python vision/check_vpca.py [git-rev-of-old-file]
"""
import importlib.util, os, subprocess, sys, tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def load_module(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def main():
    rev = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    old_src = subprocess.check_output(["git", "-C", ROOT, "show", f"{rev}:oval/page_basis.py"])
    f = tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False); f.write(old_src); f.close()
    OLD = load_module(f.name, "pb_old")
    NEW = load_module(os.path.join(ROOT, "oval", "page_basis.py"), "pb_new")
    assert "token domain" not in old_src.decode(), "old file already has the token-domain path"

    from datasets import load_dataset
    from transformers import AutoTokenizer, Qwen2ForCausalLM
    from huggingface_hub import snapshot_download
    path = snapshot_download("lmms-lab/LLaVA-Video-7B-Qwen2")
    tok = AutoTokenizer.from_pretrained(path, use_fast=False)
    lm = Qwen2ForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)   # LM weights only
    lm.eval()
    lb = load_dataset("THUDM/LongBench-v2", split="train")
    ctx = lb[0]["context"]
    msgs = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{ctx}"
    ids = tok(msgs, return_tensors="pt").input_ids[:, :128 + 20 * 32]
    with torch.no_grad():
        pkv = lm(ids, use_cache=True).past_key_values
    worst = {}
    for rv in (8, 16):
        errs_o, errs_n, diffs, mu_eq = [], [], [], True
        for li in range(len(pkv)):
            K, V = pkv[li][0][0], pkv[li][1][0]                      # [H, T, d] bf16
            H, T, d = K.shape
            Kp = K[:, 128:128 + 640].reshape(H, 20, 32, d).permute(1, 2, 0, 3)[None]
            Vp = V[:, 128:128 + 640].reshape(H, 20, 32, d).permute(1, 2, 0, 3)[None]
            a = OLD.build_value_summary(Kp, Vp, None, None, 0.0, rv)
            b = NEW.build_value_summary(Kp, Vp, None, None, 0.0, rv)
            mu_eq &= torch.equal(a[0], b[0])
            ra = a[2].float() @ a[1].float().transpose(-1, -2)
            rb = b[2].float() @ b[1].float().transpose(-1, -2)
            ex = Vp.permute(0, 1, 3, 2, 4).float() - a[0].float().unsqueeze(3)
            diffs.append(((ra - rb).norm() / ra.norm()).item())
            errs_o.append(((ra - ex).norm() / ex.norm()).item()); errs_n.append(((rb - ex).norm() / ex.norm()).item())
        worst[rv] = max(diffs)
        print(f"r_v={rv}: 28 layers x 4 heads x 20 pages | page means identical: {mu_eq} | "
              f"rebuilt values, max relative difference old vs new: {max(diffs):.2e} | "
              f"value error old {sum(errs_o)/len(errs_o):.4%} new {sum(errs_n)/len(errs_n):.4%} (mean over layers)")
    os.unlink(f.name)
    ok = all(v < 2e-3 for v in worst.values())
    print("PASS" if ok else "FAIL", "(threshold 2e-3 = half a bf16 ulp)")


if __name__ == "__main__":
    main()
