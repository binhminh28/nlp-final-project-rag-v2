# Production Dense Retrieval Baseline Report

## A. Architecture implemented

The implemented path is query → conservative trim-only normalization → content-addressed query embedding cache → existing embedding provider on a miss → validated manifest-selected cosine index → canonical hits → source-level evaluator → deterministic artifacts. The implementation adds `src/rag_chunking/retrieval/`, `src/rag_chunking/evaluation/`, two CLIs, a shared embedding-config loader, two datasets, tests, and retrieval documentation. The only upstream extension changes exact cosine ties from backend `index_id` ordering to canonical `chunk_id` ordering; chunking, document embeddings, and production indexes were not regenerated or mutated.

## B. Retrieval contracts

- `RetrievalRequest`: normalized `query`, known `strategy`, positive `top_k`, optional validated exact-match `filters`.
- `RetrievalHit`: contiguous `rank`, finite `score`, `chunk_id`, `doc_id`, `source`, `relative_path`, `strategy`, `text`, deterministic `metadata`, length diagnostics, and chunk-config identity.
- `RetrievalResult`: request values, canonical hits, cache-hit flag, retrieval fingerprint, embedding fingerprint, and index fingerprint.
- `RetrievalConfig`: schema `dense_retrieval_v1` and `similarity_metric=cosine`; no threshold or tuning parameter.

## C. Identity and fingerprinting

- Embedding configuration: `7a71fffabd36fbf5fca9018a3a0cb50c7839af6534a4bf6df7898b335d295353`
- Retrieval configuration: `24b9d7812f5c58d771c474489d6c71019fc55cb09af630dcac49eeb1a64c1058`
- Dataset: `bb215464a60a55b40193a325d1337be424e15eb85cc61a9ac4a7d0894180fe5d`
- Benchmark: `a98bca563d7147144f4cf200fbfa66dca7b24ff241bb2eac89f84b1a4317d4f3`
- Fixed-size index: `32c133ee50b4941999546a854a3a3e85a900c81080ffcbb836c614291a0ea6d3`
- Structure-aware index: `edc6569a9ac43bfd2ab0faa95b86549656f21e4b8cfe67d6cb1f757287bac729`
- Prompt-based index: `77a7a0e5b5c613e7f053c0dc92971efaebcd6cf68a51953bd34dc8068f08330b`

Embedding identity covers output-affecting model/configuration fields. Retrieval identity covers schema and cosine metric. Dataset identity covers sorted canonical records and relative-path relevance. Benchmark identity combines corpus, dataset, retrieval, embedding and all index identities, metric version, ground-truth level, and depth; it excludes timestamps, paths, latency, and cache telemetry.

## D. Evaluation dataset

The manually inspectable dataset contains 64 natural user-style questions and 79 binary relative-path labels. It has eight queries in each of conceptual, how-to, API lookup, configuration, code-related, terminology, paraphrase, and cross-document categories. Validation found zero duplicate IDs/targets, empty queries, unknown categories, malformed paths, or absent corpus targets. The same labels are used for all strategies.

## E. Query workload and cost

There are 64 unique normalized queries and an estimated 828 `cl100k_base` input tokens. The smoke populated 8 entries (117 tokens); the baseline populated the remaining 56 (711 estimated tokens). One-query provider batching implies 64 total production calls across smoke plus baseline, after which the cache contains exactly 64 entries. OpenRouter lists this model at $0.02 per million input tokens, making estimated total cost $0.00001656 and incremental full-baseline cost after smoke $0.00001422. Provider-reported actual token usage from the producing process was not retained after `stats.json` was intentionally replaced by the required cache-only rerun.

## F. Production retrieval smoke

Eight queries covered all categories. All query vectors were 1,536-dimensional and finite; every hit resolved, filters remained operational, and every strategy reached Hit@10 = 1.0. Hit@1 was 0.7500 fixed-size, 0.6250 structure-aware, and 0.3750 prompt-based. The immediate credential-free rerun had 8 hits, zero misses, and zero provider calls.

## G. Baseline aggregate metrics

| Strategy | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_size | 0.6563 | 0.8281 | 0.9063 | 0.9844 | 0.7536 | 0.8568 | 0.9492 |
| structure_aware | 0.6563 | 0.8438 | 0.8906 | 1.0000 | 0.7719 | 0.8346 | 0.9714 |
| prompt_based | 0.6094 | 0.8750 | 0.9219 | 0.9844 | 0.7499 | 0.8685 | 0.9648 |

Fixed-size and structure-aware tie on Hit@1. Prompt-based has the highest Hit@5 and Recall@5. Structure-aware has the highest Hit@10, MRR, and Recall@10. The differences are marginal overall and do not establish a single dominant strategy.

## H. Category breakdown

Values are Hit@5 / MRR.

| Category | fixed_size | structure_aware | prompt_based |
|---|---:|---:|---:|
| api_lookup | 1.0000 / 0.9000 | 1.0000 / 0.9375 | 1.0000 / 0.9375 |
| code_related | 1.0000 / 0.8542 | 1.0000 / 0.8063 | 1.0000 / 0.8229 |
| conceptual | 1.0000 / 0.8375 | 0.8750 / 0.8906 | 1.0000 / 0.8438 |
| configuration | 0.8750 / 0.8281 | 0.8750 / 0.7639 | 0.8750 / 0.7679 |
| cross_document | 0.8750 / 0.5781 | 1.0000 / 0.9375 | 1.0000 / 0.7500 |
| how_to | 1.0000 / 0.7438 | 0.7500 / 0.6240 | 0.7500 / 0.6198 |
| paraphrase | 0.5000 / 0.3807 | 0.7500 / 0.5972 | 0.8750 / 0.5521 |
| terminology | 1.0000 / 0.9063 | 0.8750 / 0.6181 | 0.8750 / 0.7054 |

Category behavior varies substantially: structure-aware is strongest on cross-document MRR, prompt-based on paraphrase Hit@5, and fixed-size on how-to and terminology Hit@5.

## I. Pairwise comparison

Using first relevant rank (A wins / ties / B wins): fixed vs structure is 15 / 35 / 14; fixed vs prompt is 14 / 38 / 12; structure vs prompt is 9 / 44 / 11. Ties dominate every pairing, reinforcing that aggregate differences are small.

## J. Failure analysis

Across 192 query-strategy records there are 69 Miss@1, 18 Miss@5, and 2 Miss@10 records. Per strategy, Miss@1 / Miss@5 / Miss@10 is 22 / 6 / 1 for fixed-size, 22 / 7 / 0 for structure-aware, and 25 / 5 / 1 for prompt-based.

Only `config-007` and `para-002` miss at five for all strategies. `config-007` retrieves closely related SSR material instead of the specifically labeled route-rendering page, indicating sibling-document ranking/terminology mismatch. `para-002` retrieves output/event tutorials but not the one labeled component-output guide; this is evidence of both paraphrase difficulty and likely incomplete benchmark labeling, not an upstream chunking defect. Other misses similarly favor close sibling documents (library creation vs use, harness usage vs environments, signal-form models vs typed forms). Fixed-size additionally shows several broad paraphrase misses, while the smaller-grained strategies sometimes return multiple chunks from a close but unlabeled document, crowding the labeled source. No evidence justifies changing chunk boundaries or labels inside this frozen baseline.

## K. Determinism

The explicit credential-free full rerun recorded 64 cache hits, 0 misses, 0 provider calls, and 192 retrieval calls. SHA-256 values for `per_query.jsonl`, `aggregate.json`, `comparison.json`, `failures.json`, `baseline_report.md`, and `manifest.json` were identical before and after rerun. Runtime-only `stats.json` was correctly allowed to differ.

## L. Integrity

Wrong dimensions: 0. NaN/Infinity: 0. Unresolved chunk IDs: 0. Duplicate result records/ranks/chunks: 0. Missing query-strategy evaluations: 0. Index fingerprint mismatches: 0. Artifact fingerprint mismatches: 0. Active failure markers in production benchmark/index/embedding paths: 0. Three historical failed `data/embeddings-smoke/.../openai-te3s-d1536-v1` directories predate this phase, have no complete manifests, and were not treated as production artifacts.

## M. Tests

Full pytest: 177 passed. Fourteen new retrieval/evaluation tests cover contracts, cache, ranking/ties, filters, mapping, lineage rejection, metrics, dataset validation, offline end-to-end evaluation, and deterministic publication. `git diff --check` passes.

## N. Generated artifacts

- Dataset and manifest: `data/evaluation/angular/baseline_v1.jsonl`, `data/evaluation/angular/manifest.json`
- Query cache: `data/query-embedding-cache/7a71fffabd36fbf5fca9018a3a0cb50c7839af6534a4bf6df7898b335d295353/`
- Frozen results: `data/retrieval/angular/baseline_v1/a98bca563d7147144f4cf200fbfa66dca7b24ff241bb2eac89f84b1a4317d4f3/`
- That directory contains per-query results, aggregates, pairwise comparison, failure analysis, runtime stats, human summary, and benchmark manifest.

## O. Findings

No strategy is uniformly strongest. Structure-aware has the best early-rank quality by MRR and the only perfect Hit@10; prompt-based retrieves a relevant source most often by rank five and has the best Recall@5; fixed-size remains competitive and leads several exact/terminology-heavy categories. The largest variation is category-specific, especially paraphrase, cross-document, how-to, and terminology. Dominant failures are close-document competition and terminology mismatch; some expose narrow labels where multiple corpus pages are plausibly useful. Nothing in the results demonstrates an upstream chunking correctness defect.

## P. Final verdict

SAFE TO PROCEED TO RETRIEVAL TUNING PHASE
