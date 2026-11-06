"""Fetch LLaVA-Video-7B-Qwen2, the Video-MME annotations, and chunk(s) of videos."""
import sys
from huggingface_hub import snapshot_download, hf_hub_download
what = sys.argv[1]
if what == "model":
    p = snapshot_download("lmms-lab/LLaVA-Video-7B-Qwen2")
elif what == "ann":
    p = snapshot_download("lmms-lab/Video-MME", repo_type="dataset", allow_patterns=["videomme/*", "README.md"],
                          local_dir="/home/shahbaa/vmme")
else:  # chunk number
    p = hf_hub_download("lmms-lab/Video-MME", f"videos_chunked_{int(what):02d}.zip", repo_type="dataset",
                        local_dir="/home/shahbaa/vmme")
print(p)
