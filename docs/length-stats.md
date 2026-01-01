# Token lengths of the first-50 subsets

Measured with `harness/measure_lengths.py` (GLM-4-9B-Chat-1M tokenizer,
InfiniteBench reference prompt templates, `chatglm3` -> yarn_mistral mapping).
Lengths are of the full prompt, i.e. what the engine actually prefills.

```
task                     n      min      p50      max  >128K  >262K       total
-------------------------------------------------------------------------------
passkey                 50  125,340  125,342  125,342      0      0   6,267,380
number_string           50  125,345  125,349  125,351      0      0   6,268,026
kv_retrieval            50  137,079  137,377  137,639     50      0   6,870,602
longdialogue_qa_eng     50   92,701  105,184  122,715      0      0   5,308,103
longbook_sum_eng        50   72,645  176,916  743,825     34     10  10,326,631
longbook_choice_eng     50   72,687  156,009  452,649     32     12   9,334,556
longbook_qa_eng         50   87,878  134,980  255,376     28      0   7,848,146
longbook_qa_chn         50   21,308  178,985  3,612,926    25     20  22,871,617
math_find               50   90,033   90,033   90,033      0      0   4,501,800
code_run                50   76,179   76,305   76,450      0      0   3,815,625
code_debug              50  130,877  179,994  199,443     46      0   8,571,349
-------------------------------------------------------------------------------
TOTAL prefill+gen tokens                                               91,983,835
```

## What this settles

**Truncation length is not a free choice — it changes 215 of 550 records.**
Six tasks have records above 128K: `kv_retrieval` (all 50), `code_debug` (46),
`longbook_sum_eng` (34), `longbook_choice_eng` (32), `longbook_qa_eng` (28),
`longbook_qa_chn` (25).

**Truncation is mandatory at any setting.** `longbook_qa_chn` reaches 3.6M
tokens — beyond even the model's native 1M. No configuration runs this suite
untruncated.

**Our sm_120 ceiling binds only if we go above 128K.** 42 records exceed
262,144 tokens (longbook_qa_chn 20, longbook_choice_eng 12, longbook_sum_eng
10). At the reference impl's 128K they all fit with room to spare, and
deviation D2 never fires.

**Cost.** ~92M prefill tokens per run uncapped; ~62M with a 128K cap. Two runs
(b=512, b=2048). At a rough 3-8K tok/s for a 9B model at this context, that is
roughly 4-7 h per run.

## Note on the reference default

`eval_chatglm.py:22-23` sets `MAX_POSITION_ID = TRUNCATE_LEN = 128 * 1024` with
the comment "Determined by the model" — written for ChatGLM3's 128K window, not
GLM-4-9B-Chat-1M's 1M. So the reference default is not self-evidently the
paper's choice. Under a 128K cap every record lands at ~100-131K, which does
match the paper's repeated description of the suite as "100K+ context"
(sec5_eval.tex:32, appC_extres.tex:8).
