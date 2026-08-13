# Results Claim–Source Checklist

| Claim family | Authoritative source | Check |
|---|---|---|
| Benchmark cardinality and frozen lineage | `canonical_v2/freeze_manifest.json` | 140 questions, 3 strategies, 420 answers; PASS |
| Shared corpus, embedding, retrieval protocol, and index lineage | canonical retrieval `manifest.json`; prepared-input `manifest.json` | Fingerprints match freeze; candidate depth 50; 2,048-token protocol |
| Shared context construction | prepared-input `manifest.json`; canonical generation inputs | Same context fingerprint/configuration and 4,096 rendered-context budget |
| Shared generation | `configs/generation_gpt5mini_v2.json`; generation manifests | `openai/gpt-5-mini`; generation fingerprint matches freeze |
| Chunk counts and token distributions | each canonical chunk `manifest.json`, `stats.json`, and `chunks.jsonl` | Matches derived summary |
| Fixed-size overlap and expansion | fixed-size `stats.json` and stored spans/overlap metadata | 522,558 source; 581,181 occurrences; 58,623 overlap; 11.22% |
| Evidence preservation and fragmentation | compatibility `evidence_chunk_mappings.jsonl`; processed corpus; canonical chunks | Exact source spans projected through canonical provenance; 233 items/strategy |
| Context utilization and overhead | canonical prepared generation inputs | 420 records; raw sums and rendered counts reproduced |
| Retrieval metrics | canonical production v2 `per_query.jsonl`; final report | Same-token-budget records only |
| Token F1 and difficulty/type results | canonical evaluation artifacts; `FINAL_BENCHMARK_REPORT.md` | 420 rows joined by query and strategy |
| Strategy confidence intervals | statistical `primary_metrics.json` | 50,000 paired bootstrap resamples, seed 2026 |
| Paired deltas and tests | `paired_comparisons.json`; `significance_tests.json` | 100,000 sign-flip resamples; Holm correction; no rejection |
| Wilcoxon robustness | `paired_comparisons.json`; `significance_tests.json` | All Holm-adjusted p = 0.6346 |
| Provider input/output/reasoning coverage | canonical answers and provider diagnostics | Reasoning coverage fixed 136/140; others 140/140 |
| Prompt artifact version | prompt chunk `manifest.json`; current `prompt_prompts.py`; README | Frozen artifact `prompt_based_v1`; current default `prompt_based_v2` |

## Language audit

- No claim states that one strategy significantly outperformed another.
- Observed rankings are labeled observed or descriptive.
- Correlations and proposed mechanisms are labeled exploratory, non-causal, or possible.
- The practical recommendation is explicitly limited to the tested Angular benchmark.
- No composite score or invented redundancy value is used.
