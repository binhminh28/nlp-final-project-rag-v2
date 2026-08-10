# Dense retrieval and evaluation

This phase measures the existing pure dense-vector baseline. It deliberately has no generation, reranking, hybrid search, query rewriting, threshold, or strategy-specific heuristic.

## Architecture

```text
query -> trim-only normalization -> query embedding cache -> embedding provider on miss
      -> manifest-validated local cosine index -> canonical RetrievalHit values
      -> relative-path relevance evaluation -> deterministic benchmark artifacts
```

`RetrievalRequest` validates a non-empty normalized query, one of the three known strategies, a positive `top_k`, and exact string filters for `strategy`, `doc_id`, or `source`. `RetrievalHit` contains rank, finite cosine score, canonical chunk/document identity, corpus source, relative path, strategy, full text, metadata, token/character lengths, and chunk configuration identity. `RetrievalResult` traces the query to retrieval, embedding, and index fingerprints. `RetrievalConfig` v1 identifies cosine search only; request depth is intentionally not configuration identity.

The query cache is separate from document embeddings at `data/query-embedding-cache/<embedding-fingerprint>/`. Its key covers the normalized query, embedding configuration fingerprint, `input_type=query`, and cache schema. Cache records validate all identity fields, dimensions, numeric types, and finite values before reuse.

Indexes are passed into `RetrievalService`, loaded through `manifest.json`, and checked for corpus, strategy, schema/backend, dimension, recomputed index fingerprint, and embedding lineage. The referenced embedding artifact is then hash-checked and used for canonical hit resolution. Missing or duplicated vector/chunk IDs are fatal. Ranking is descending cosine score with `chunk_id` as the exact-tie breaker.

## Dataset and metrics

The JSONL dataset uses this schema:

```json
{"query_id":"stable-id","query":"natural question","category":"conceptual","relevant_sources":["guide/path.md"]}
```

Ground truth is binary at canonical `relative_path` level, shared across strategies. The loader rejects malformed JSON, duplicate keys/IDs/labels, empty queries, unknown categories, absolute or backslash paths, missing labels, and targets absent from the common corpus.

Evaluation retrieves once to depth 10 per query/strategy and computes Hit@1/3/5/10, MRR from the first relevant source through rank 10, and distinct-source Recall@5/10. Aggregates are reported overall and per category. Pairwise wins use the lower first-relevant rank; two misses tie. Changing query text, labels, schema, metrics, depth, retrieval configuration, embedding configuration, corpus, or any index identity changes the benchmark fingerprint.

## Commands

```bash
retrieve --corpus angular --strategy fixed_size --query "How does dependency injection work?" --top-k 5
evaluate-retrieval --corpus angular --dataset data/evaluation/angular/baseline_v1.jsonl --plan-only
evaluate-retrieval --corpus angular --dataset data/evaluation/angular/baseline_v1.jsonl
```

Use repeatable `--filter FIELD=VALUE` options with `retrieve`. `evaluate-retrieval` embeds each normalized query once and reuses that vector for every strategy. `--plan-only` validates all lineage and labels and reports cache/call workload without provider calls.

Artifacts are published transactionally under `data/retrieval/<corpus>/<baseline-name>/<benchmark-fingerprint>/`; `manifest.json` is the last commit marker. `per_query.jsonl`, `aggregate.json`, `comparison.json`, `failures.json`, `baseline_report.md`, and `manifest.json` are deterministic. `stats.json` contains runtime/cache telemetry and is intentionally excluded from deterministic comparisons. A failure writes `failure.json` and cannot publish a valid manifest.

`baseline_v1` is frozen. Any later ranking, query transformation, metric, label, depth, or retrieval change must create a new identity and must not overwrite this baseline as though it were the same experiment.
