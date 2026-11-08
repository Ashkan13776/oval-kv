"""LVBench (C2): 103 videos (~1 h each), 1,549 four-way MCQs. Annotations: zai-org/LVBench
video_info.meta.jsonl (identical uids/questions/answers to lmms-lab/LVBench's parquet).
Prompt = lmms-eval tasks/lvbench lvbench_doc_to_text with the default kwargs (options are inline
in `question`): question + "\\nAnswer the question with the option letter"."""
import json, os

LVB_ROOT = os.environ.get("LVB_ROOT", os.path.expanduser("~/lvbench"))
POST = "\nAnswer the question with the option letter"


def load_docs(args=None):
    import pandas as pd
    rows = [json.loads(l) for l in open(os.path.join(LVB_ROOT, "video_info.meta.jsonl"))]
    recs = []
    for r in rows:
        for q in r["qa"]:
            recs.append(dict(question_id=str(q["uid"]), videoID=r["key"], answer=q["answer"],
                             question=q["question"], text=q["question"] + POST,
                             task_type="|".join(q["question_type"]), question_type=list(q["question_type"]),
                             time_reference=q["time_reference"], group=r["type"], vtype=r["type"],
                             duration_minutes=r["video_info"]["duration_minutes"]))
    return pd.DataFrame(recs)
