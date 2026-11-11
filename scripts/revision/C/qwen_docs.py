"""Questions for the Qwen2.5-VL runs (shared by run_qwen25vl.py and qwen_official.py, so both
envs build the identical prompt text). Video-MME: lmms-eval videomme_doc_to_text with the
default lmms_eval_specific_kwargs (the ones its qwen2_5_vl model uses)."""
import os

VMME_ROOT = os.environ.get("VMME_ROOT", os.path.expanduser("~/vmme"))
ANN = os.path.join(VMME_ROOT, "videomme/test-00000-of-00001.parquet")
OPT = ("Select the best answer to the following multiple-choice question based on the video "
       "and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.")
POST = "\nAnswer with the option's letter from the given choices directly."


def load_docs(args):
    import pandas as pd
    if args.bench == "videomme":
        df = pd.read_parquet(ANN)
        df = df[df.duration.isin(args.durations)].reset_index(drop=True)
        df["text"] = [OPT + "\n" + q + "\n" + "\n".join(list(o)) + "\n" + POST
                      for q, o in zip(df.question, df.options)]
        df["group"] = df.duration
        return df
    import lvbench
    return lvbench.load_docs(args)
