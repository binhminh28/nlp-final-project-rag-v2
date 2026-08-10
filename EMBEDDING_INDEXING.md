# Embedding and indexing phase

The pipeline consumes every chunking strategy through the same Unified Chunk
Format reader. It does not contain strategy dispatch logic. Canonical identity
includes the chunk-manifest hash and chunk config fingerprint plus provider,
model, dimension, tokenizer/input limits, input type, encoding, and schema
version. Runtime retry/batch statistics do not affect identity.

The production adapter calls OpenRouter's `/embeddings` endpoint. Default tests
and dry runs use the deterministic fake provider and cannot make network calls.
No text is silently truncated: batch planning enforces per-input and aggregate
token limits before provider calls.

Use `--limit N` for an explicit smoke artifact. Its manifest is labeled
`build_scope: sample` and records the full `source_chunk_count`, so it cannot be
mistaken for a completed corpus build.

Recommended artifact layout:

```text
data/embeddings/<corpus>/<strategy>/<embedding-config-fingerprint>/
  embeddings.jsonl
  stats.json
  manifest.json
data/embedding-cache/<embedding-config-fingerprint>/
data/indexes/<corpus>/<strategy>/<index-fingerprint>/
  index.jsonl
  stats.json
  manifest.json
```

`manifest.json` is published last and is the completion marker. Successful
batches enter the atomic, content-addressed cache immediately, so an interrupted
run resumes without repeating them. A failed run writes `failure.json` and does
not publish a complete manifest. Existing output directories reject a different
experiment identity.

Example production build (load `OPENROUTER_API_KEY` into the environment first):

```powershell
embed-chunks --input data/chunks/angular/fixed_size --output data/embeddings/angular/fixed_size/<fingerprint> --cache data/embedding-cache/<fingerprint> --corpus angular
build-vector-index --input data/embeddings/angular/fixed_size/<fingerprint> --output data/indexes/angular/fixed_size/<index-fingerprint>
```

The local backend performs persistent cosine similarity search and exact
payload filtering for `strategy`, `doc_id`, `source`, or any other canonical
payload field. It is intentionally an integrity/smoke-test backend; retrieval
orchestration, hybrid search, reranking, and answer generation remain out of
scope.
