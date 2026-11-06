"""Profile OVAL decode steps on one Video-MME question (diagnostic only)."""
import sys, os, time, torch
sys.argv = [sys.argv[0]] + sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_videomme as RV
import pandas as pd
args = RV.build_parser().parse_args()
tok, model, proc = RV.load_llava(args)
model, su, eos = RV.patch_harness(args, tok, model)
df = pd.read_parquet(RV.ANN); doc = df.iloc[0].to_dict(); doc["options"] = list(doc["options"])
from llava.constants import IMAGE_TOKEN_INDEX
ids, pix, _ = RV.build_inputs(doc, args, tok, model, proc)
with torch.no_grad():
    _, _, _, _, emb, _ = model.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, [pix], modalities=["video"])
    p = int((ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0]); n_vis = emb.shape[1] - (ids.shape[1] - 1)
    k = RV.build_inputs.n_ctx_text; split = p + n_vis + k; stream = ids[0, p + 1 + k:].tolist()
    su.reset(ids)
    torch.cuda.synchronize(); t = time.time()
    out = model(inputs_embeds=emb[:, :split], past_key_values=None, use_cache=True); pkv = out.past_key_values
    torch.cuda.synchronize(); print("prefill", time.time() - t, flush=True)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
        out = model(input_ids=torch.tensor([[stream[0]]], device="cuda:0"), past_key_values=pkv, use_cache=True)
        torch.cuda.synchronize()
    print(prof.key_averages(group_by_stack_n=6).table(sort_by="cpu_time_total", row_limit=12, max_name_column_width=60))
