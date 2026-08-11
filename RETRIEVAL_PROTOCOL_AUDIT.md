# Retrieval Protocol Audit and Freeze Gate

This phase keeps chunking strategy as the research variable. The controlled
base retriever is unchanged: trim-only query normalization, the configured
query embedding, dense cosine similarity against the selected strategy index,
then deterministic ordering by `(score DESC, chunk_id ASC)`. No hierarchical,
hybrid, BM25, reranking, rewriting, context construction, generation, or answer
evaluation is part of this path.

## A. Codebase audit

| Component | Status | Existing location | Action |
|---|---|---|---|
| Dense retriever | EXISTS_AND_REUSABLE | `retrieval/service.py` | Reused unchanged |
| Query embedding/cache | EXISTS_AND_REUSABLE | `retrieval/service.py`, `retrieval/cache.py` | Reused unchanged |
| Vector index loader | EXISTS_AND_REUSABLE | `embedding/index.py` | Reused unchanged |
| Retrieval result schema | EXISTS_BUT_NEEDS_EXTENSION | `retrieval/models.py` | Existing hits reused; protocol selection adds budget audit fields |
| Top-K retrieval | EXISTS_AND_REUSABLE | `RetrievalRequest`, `LocalCosineIndex.search` | Preserved as historical baseline |
| Token counting | EXISTS_AND_REUSABLE | `chunking/tokenizer.py`, unified `Chunk.token_count` | Reused; text only |
| Token-budget retrieval | MISSING | — | Added in `retrieval/protocols.py` |
| Historical dataset loader | EXISTS_AND_REUSABLE | `evaluation/dataset.py` | Preserved without semantic changes |
| Historical relevance mapping | EXISTS_AND_REUSABLE | `EvaluationQuery.relevant_sources` | Still source-path labels |
| External QA adapter | MISSING | — | Added in `evaluation/qa_dataset.py` |
| Evidence mapping | MISSING | — | Added in `evaluation/evidence.py` |
| Hit@K / Recall@K / MRR | EXISTS_AND_REUSABLE | `evaluation/metrics.py` | Historical definitions preserved |
| Evidence coverage | MISSING | — | Added as separate metrics |
| Config fingerprints | EXISTS_BUT_NEEDS_EXTENSION | embedding/retrieval models | Added protocol-specific semantic fingerprints |
| Result manifest | EXISTS_AND_REUSABLE | `evaluation/runner.py` | Manifest-last writer reused by new runner |
| Tests | EXISTS_BUT_NEEDS_EXTENSION | `tests/` | Protocol/evidence/QA/integration cases added |

Audit answers:

1. Base retrieval is query embedding followed by local dense cosine ranking.
2. It is identical for all three strategies; only validated chunk, embedding,
   and index artifacts differ.
3. Before this phase, `top_k` was the only base retrieval budget.
4. Historical relevance is binary membership in a set of source
   `relative_path` labels.
5. Historical labels are not chunk-ID dependent.
6. Unified chunks persist `token_count` for chunk text and identify
   `tiktoken:cl100k_base`.
7. The current metadata is reusable: validators re-encode it, and observed
   production ranges are fixed 32–512, structure 1–512, prompt 3–512 tokens.
8. Retrieval is deterministic for unchanged vectors and artifacts.
9. Ties are deterministic: `chunk_id ASC`.
10. The old baseline is reproducible from checked-in artifacts and cached query
    vectors. The freeze rerun reproduced all deterministic files byte-for-byte.

## B. Retrieval and budget contracts

`same_top_k` returns the first K dense-ranked whole chunks (or all candidates
when fewer exist). This is the conventional baseline.

`same_token_budget` first requests `candidate_k` dense candidates, then scans
them in order. It selects unique whole chunks that fit, skips later chunks that
do not fit, continues scanning, and never truncates. Token accounting includes
chunk text only; metadata and separators are excluded. If the first candidate
alone exceeds the budget, it is returned as the sole selection with
`budget_overflow=true`. Current production chunks are at most 512 tokens, so a
2048-token budget cannot trigger this exception.

Per-query output records requested budget, actual selected tokens, utilization,
selected chunk count, overflow, candidates, dense ranks, scores, chunk IDs,
index/config fingerprints, evidence mappings, covered/uncovered evidence, and
metric contributions. Aggregate token-budget output reports mean, median, p95,
mean chunk count, and mean utilization separately by strategy and protocol.

The default frozen protocol identities are:

- `same_top_k` (K=5):
  `ba5a38b9151c26c41d8d2a9a62dd1fa77873d564fd75a265dec2d1baf4a2292a`
- `same_token_budget` (candidate_k=50, B=2048):
  `b3119959956f707f488de71ea677a8d88ea137c09c7108a5f6655a4a63dd1e85`

## C. External QA and evidence mapping

The QA loader accepts JSONL or a JSON array with the team-owned fields `id`,
`doc_id`, `question`, `answer`, `evidence_sentences`, `evidence_sections`,
`question_type`, `difficulty`, and optional `notes`. It validates structure,
uniqueness, required answers, and corpus document identity without constraining
team-defined question or difficulty vocabularies. Semantic validation is a
separate report and never repairs input.

Evidence is mapped independently for every strategy. Resolution priority is:

1. supplied block/character provenance;
2. exact canonical source-text locations intersected with chunk source spans;
3. deterministic Unicode NFKC, case-folded, whitespace-normalized text;
4. section paths when sentence evidence is absent.

Fixed chunks use canonical token spans. Structure and prompt chunks use their
persisted source block fragments. Intersections retain fractional offsets, so
evidence crossing a chunk boundary can require the union of multiple retrieved
chunks. Derived chunk IDs are output artifacts only and are never written into
the canonical QA dataset.

Historical Hit@K, source Recall@K, and MRR are untouched. The evidence runner
adds macro evidence coverage and all-evidence-retrieved rate. Multi-evidence
questions retain each evidence unit rather than collapsing to one binary hit.

## D. Identity and artifact freeze

The evidence benchmark identity includes corpus, dataset, chunk artifact,
embedding config/artifact, index, base retrieval config, protocol configs,
tie-breaking, metric versions, and evaluator schema. Runtime duration, cache
hits, and provider call counts are statistics and do not affect fingerprints.

Pinned artifact hashes:

| Artifact | SHA-256 / semantic fingerprint |
|---|---|
| Corpus `documents.jsonl` | `cc3ccf3401e0005004525466d6424517719dbc879d0c7b0ed9489fe08d33f32c` |
| Fixed chunks | `f3aa6da318275c474a3e8e1123e6612f2a70f35054e720356ef79f9ac49b50c5` |
| Structure chunks | `87c0d22940e5d2ea206f0c5b7d9e878935e93b65c77dbaef4d921094b36e066c` |
| Prompt chunks | `1c809a3bdfb86d255e0cf90426bc2c4f9a3a2693ca42679e52023aadbc24dff4` |
| Embedding config | `7a71fffabd36fbf5fca9018a3a0cb50c7839af6534a4bf6df7898b335d295353` |
| Historical QA/source dataset | `bb215464a60a55b40193a325d1337be424e15eb85cc61a9ac4a7d0894180fe5d` |
| Historical benchmark | `a98bca563d7147144f4cf200fbfa66dca7b24ff241bb2eac89f84b1a4317d4f3` |

The compatibility artifact is under
`data/retrieval/angular/retrieval_protocol_freeze_v1/a98bca.../`. Aggregate,
per-query, comparison, failures, and Markdown report files are byte-identical
to `baseline_v1`; only run-label/runtime-bearing files differ. The rerun made
zero provider calls.

| Strategy | Hit@1 | Hit@5 | MRR | Recall@10 |
|---|---:|---:|---:|---:|
| fixed_size | 0.65625 | 0.90625 | 0.753577628968254 | 0.94921875 |
| structure_aware | 0.65625 | 0.890625 | 0.771875 | 0.9713541666666666 |
| prompt_based | 0.609375 | 0.921875 | 0.7499069940476191 | 0.96484375 |

Publication uses the existing staged artifact-set writer: data and diagnostics
are installed before `manifest.json`, which is the commit marker. Failed new
runs have no complete manifest; previously committed files are rolled back.

## E. Commands and gate

Run the frozen source-label baseline:

```bash
python -m rag_chunking.cli.evaluate_retrieval \
  --corpus angular --dataset data/evaluation/angular/baseline_v1.jsonl \
  --baseline-name retrieval_protocol_freeze_v1
```

Validate a team QA dataset without retrieval/provider work:

```bash
python -m rag_chunking.cli.evaluate_evidence_retrieval \
  --corpus angular --dataset PATH_TO_TEAM_QA.jsonl --plan-only
```

Run both controlled protocols after validation:

```bash
python -m rag_chunking.cli.evaluate_evidence_retrieval \
  --corpus angular --dataset PATH_TO_TEAM_QA.jsonl \
  --top-k 5 --candidate-k 50 --token-budget 2048
```

The five-record fixture in `tests/fixtures/qa_development_only.jsonl` is
explicitly development-only and must not be used for research conclusions.

Limitations/future work: the canonical evidence QA dataset is externally owned
and is not present, so no final evidence benchmark is claimed. Ambiguous
repeated evidence strings map to all exact occurrences and remain auditable.
Hierarchical retrieval, reranking, hybrid/BM25 retrieval, query rewriting,
context building, generation, and answer evaluation remain future work and are
not implemented by this phase.

Freeze verdict: the retrieval/evidence layer is ready for the separately owned
context-construction phase, subject to supplying and validating the canonical
QA dataset.
