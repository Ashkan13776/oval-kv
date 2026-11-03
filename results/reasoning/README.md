# Long-reasoning results (FreeKV harness, `eval.reasoning.pred`)

Every OVAL long-reasoning cell lands here: one `.jsonl` per (model, dataset,
configuration, seed), one record per problem (`qid`, `pred`, `answer`,
`output_len`, ...). The configuration is encoded in the filename, so the file
name alone says which pieces of OVAL were on.

## Reading a filename

```
results/reasoning/ds-r1-qwen-7b-spec_ret/
  AIME24-s512-r512-1.00-manual-p32-b2048-avgSM-oval_r8_etab71b9668a_q0_recon1_vr16_vpca-pQ2-llb0-pQcs0.9-skl1-t0.6-topp0.95-Mg16384-seed43.jsonl
```

| field | meaning |
|---|---|
| `<model>-spec_ret/` | model, and the FreeKV speculative-retrieval method every cell uses |
| `AIME24` | dataset: `MATH50`, `AIME24` (30 problems), `GPQA50c` |
| `s512-r512` | 512 sink and 512 recent tokens, always exact |
| `1.00-manual` | harness sparsity / pattern fields (fixed) |
| `p32-b2048` | pages of 32 tokens, top-k budget 2048 tokens |
| `avgSM` | GQA combine of page scores |
| `oval_r8` | **retrieval**: pages scored with rank-8 OVAL records |
| `eta0.25` | one shared eta for every head |
| `etab<hash>` | **per-head eta** from a table (`--eta_table`); `<hash>` = 8 hex of the table's content, see `results/eta_rules/*/tables/*.json` (`"hash"`) |
| `q0` | bf16 records (`OVAL_QUANT=0`) |
| `recon1` | **key compression**: selected pages' keys rebuilt from the records |
| `cr<R>` | key reconstruction at rank R while scoring stays at `oval_r` |
| `vr<R>_vpca` | **value compression**: selected pages' values rebuilt from rank-R PCA records |
| `pQ2-llb0-pQcs0.9` | spec_ret: 2 steps, correction threshold 0.9 |
| `skl1` | layer 0 dense |
| `t0.6-topp0.95-Mg16384` | sampling temperature, top-p, max generated tokens |
| `-seed<S>` | sampling seed; no suffix = seed 42 |

No `recon1` = retrieval only (exact keys and values attended). No `vr` =
exact values.

## Which files belong to which experiment

| experiment | tag pattern | models | summary |
|---|---|---|---|
| eta sweep, retrieval + key compression | `oval_r8_eta{0.0,0.25,0.5,0.75,1.0}_q0_recon1-` | llama-8b, qwen-14b (qwen-7b cells not in the repo) | `eta_sweep_*.md`, [docs/1-retrieval.md](../../docs/1-retrieval.md) |
| key reconstruction rank, scoring fixed at 8 | `_recon1_cr{1,4,8,16,32}-` | qwen-7b | `key_rank_decoupled_qwen7b.md`, [docs/2-key-compression.md](../../docs/2-key-compression.md) |
| value PCA rank | `_recon1_vr{1,4,8,16,32}_vpca-` (fixed eta) | qwen-7b | `value_rank_qwen7b.md`, [docs/3-value-compression.md](../../docs/3-value-compression.md) |
| per-head eta, rule 1, values rank 8 | `_etab<hash>_q0_recon1_vr8_vpca-` | qwen-7b, llama-8b | `results/eta_rules/table.md`, [docs/4-per-head-eta.md](../../docs/4-per-head-eta.md) |
| **full pipeline**, values rank 16 | `_etab<hash>_q0_recon1_vr16_vpca-` | to be run | `results/eta_rules/vr16/table_<model>.md`, [docs/5-full-pipeline.md](../../docs/5-full-pipeline.md) |

`latency_memory_probe.json`: decode latency and record-slab bytes vs key rank
(qwen-7b, 16K context), used in docs/2-key-compression.md.

## Scoring

```bash
python analysis/score_avg_pass_at_k.py --models ds-r1-llama-8b             # eta sweep
python analysis/value_rank_table.py                                          # value rank
python analysis/eta_rules_table.py                                           # rule 1, values rank 8
python analysis/eta_rules_table.py --vrank 16 --eta_dir results/eta_rules/vr16  # full pipeline
```

A seed counts only if its file holds every problem; partial files are never
mixed in.
