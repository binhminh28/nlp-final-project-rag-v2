# Final Benchmark Handoff

## Readiness versus completion

The implementation pipeline is complete through deterministic paired answer
evaluation. The production experiment is **not complete**: the teammate-owned
QA dataset is present, but its strict evidence-to-corpus/chunk compatibility gate
currently fails. Synthetic fixtures in `tests/` are plumbing checks only.

Expected handoff path:

```text
data/evaluation/angular/qa_dataset.jsonl
```

This immutable upstream path is explicitly unignored. The repository owns its
adapter, validation, fingerprinting, compatibility checks, orchestration, and
artifact publication. Validation never repairs or rewrites authored content.

Run `audit-dataset-compatibility` before any retrieval or generation. The report
is committed manifest-last under `data/evaluation/angular/compatibility/qa_dataset`.
`prepare-answer-inputs`, evidence retrieval, and `benchmark-preflight` apply the
same gate and exit non-zero when it fails.

## Authoritative QA contract

The real external schema is adapted as `team_evidence_qa_adapter_v1`. JSONL row
order and JSON key order do not affect identity because validated records are
sorted by `question_id` before fingerprinting. File location and timestamps are
excluded. Nested evidence is never flattened.

| Field | Type | Required | Meaning and validation | Pipeline use |
|---|---|---:|---|---|
| `question_id` | string | yes | Query ID; non-empty, whitespace-free, globally unique | Canonical `id`; retrieval/generation identity |
| `question` | string | yes | Non-empty; trim-only normalization at load | Exact retrieval and generation question; evaluation provenance |
| `reference_answer` | string | yes | Non-empty gold reference | Canonical `answer`; evaluation only |
| `evidence` | object array | yes | `evidence_id`, `doc_id`, `section_path`, and non-empty `evidence_sentences` | Structured provenance and relevance labels |
| `question_type` | string | yes | Non-empty open-ended canonical category; no separate `category` field | Evaluation grouping/paired metadata only |
| `difficulty` | string | yes | Non-empty teammate-assigned label | Evaluation/paired metadata only |
| `reasoning_type` | string | yes | Non-empty authored reasoning label | Stratified reporting only |
| `metadata` | object | yes | Scope, evidence count, competition/content/answerability fields | Validation and stratified reporting only |

An evidence item has exactly:

```json
{
  "evidence_id": "q1_e01",
  "doc_id": "angular:guide/example.md",
  "section_path": ["Parent", "Child"],
  "evidence_sentences": ["Exact authored evidence text."]
}
```

The audit derives block and zero-based half-open character provenance without
changing this object. It permits exact text, NFKC/case/whitespace normalization,
and delimiter-only rendered-Markdown normalization. It never uses embeddings,
LLMs, semantic similarity, retrieval results, or filename-only document matches.

Every accepted row field—including `notes`, category, difficulty, answer, and
evidence—contributes to the dataset fingerprint. The external file does not carry
a competing per-row schema tag; adding `schema_version` to a row is rejected as
an unknown field. The validator reports its selected schema version explicitly.

Placeholder shape only—**not production QA content**:

```json
{"id":"DEMO_ONLY","doc_id":"angular:path/from-processed-manifest.md","question":"DEMO QUESTION","answer":"DEMO GOLD ANSWER","evidence_sentences":["EXACT DEMO SOURCE TEXT"],"evidence_sections":[],"question_type":"demo_only","difficulty":"demo_only","notes":"SYNTHETIC NON-PRODUCTION EXAMPLE"}
```

## Gold boundary

The legal projection is enforced in code:

```text
QA record
  ├─ retrieval: id + question
  ├─ generation: id + exact question + authoritative ContextResult
  └─ evaluation: generated result + answer/category/difficulty/evidence metadata
```

`prepare-answer-inputs` projects `BenchmarkQuery` values before retrieval. It
never receives answer/evidence in its core preparation function. Retrieval does
not use categories or difficulty. Generation does not receive a QA record and has
no category/difficulty prompt branch.

## Validation and preflight

Validate the schema, then run the required compatibility gate:

```bash
validate-qa-dataset \
  --dataset data/evaluation/angular/qa_dataset.jsonl \
  --documents data/processed/angular/documents.jsonl

audit-dataset-compatibility \
  --dataset data/evaluation/angular/qa_dataset.jsonl
```

The compatibility command adds document/section/sentence resolution, exact
provenance, all-strategy chunk coverage, detailed unresolved cases, and the final
PASS/FAIL decision. It is offline and does not run benchmark metrics.

The read-only three-strategy preflight validates the processed corpus, complete
chunk artifacts, chunk coverage, full embedding artifact hashes, index entries,
all manifest lineage, shared embedding configuration, protocol/context/
generation/evaluation configurations, credentials when requested, and output
conflict markers:

```bash
export ANSWER_GENERATION_MODEL='PINNED_PRODUCTION_MODEL_ID'

benchmark-preflight \
  --dataset data/evaluation/angular/qa_dataset.jsonl \
  --generation-provider openrouter \
  --generation-model "$ANSWER_GENERATION_MODEL" \
  --require-live-credentials \
  --output-path data/benchmark/angular/canonical_v1/inputs \
  --output-path data/benchmark/angular/canonical_v1/generation/fixed_size \
  --output-path data/benchmark/angular/canonical_v1/generation/structure_aware \
  --output-path data/benchmark/angular/canonical_v1/generation/prompt_based \
  --output-path data/benchmark/angular/canonical_v1/evaluation
```

Pin the answer model deliberately before production. The repository does not
choose a research model implicitly.

Current dataset compatibility is `FAIL`; therefore do not run the execution
sequence below. Exact current counts are in the compatibility artifact.

```text
PASS    processed corpus (384 documents)
WARNING retained processed-corpus audit warnings (non-blocking)
PASS    fixed_size chunks / embeddings / index
PASS    structure_aware chunks / embeddings / index
PASS    prompt_based chunks / embeddings / index
PASS    protocol / context / fake-generation / evaluation configs
BLOCKED canonical production QA dataset not yet available
```

This is code/artifact readiness, not an executed production experiment.

## Final execution sequence

Run only after strict validation and live preflight pass.

### 1. Retrieve, select protocol, and build contexts

```bash
prepare-answer-inputs \
  --dataset data/evaluation/angular/qa_dataset.jsonl \
  --corpus angular \
  --embedding-mode openrouter \
  --protocol same_token_budget \
  --candidate-k 50 \
  --token-budget 2048 \
  --context-token-budget 4096 \
  --output data/benchmark/angular/canonical_v1/inputs
```

This embeds each unique question once, retrieves all three strategies, applies
one protocol/config, builds authoritative `ContextResult` objects, and atomically
publishes:

```text
fixed_size.generation_inputs.jsonl
structure_aware.generation_inputs.jsonl
prompt_based.generation_inputs.jsonl
stats.json
manifest.json
```

Use `--embedding-mode cache_only` for a no-network verification; it fails on the
first missing query embedding rather than polluting the production cache with a
fake vector. A matching committed input manifest is reused without retrieval.

### 2. Generate answers with one pinned configuration

Use identical flags for every strategy:

```bash
generate-answers \
  --input data/benchmark/angular/canonical_v1/inputs/fixed_size.generation_inputs.jsonl \
  --output data/benchmark/angular/canonical_v1/generation/fixed_size \
  --cache data/generation-cache \
  --provider openrouter --model "$ANSWER_GENERATION_MODEL" \
  --temperature 0 --max-output-tokens 512 --context-window-tokens 8192 \
  --timeout-seconds 60 --max-retries 3 --retry-backoff-seconds 0.5

generate-answers \
  --input data/benchmark/angular/canonical_v1/inputs/structure_aware.generation_inputs.jsonl \
  --output data/benchmark/angular/canonical_v1/generation/structure_aware \
  --cache data/generation-cache \
  --provider openrouter --model "$ANSWER_GENERATION_MODEL" \
  --temperature 0 --max-output-tokens 512 --context-window-tokens 8192 \
  --timeout-seconds 60 --max-retries 3 --retry-backoff-seconds 0.5

generate-answers \
  --input data/benchmark/angular/canonical_v1/inputs/prompt_based.generation_inputs.jsonl \
  --output data/benchmark/angular/canonical_v1/generation/prompt_based \
  --cache data/generation-cache \
  --provider openrouter --model "$ANSWER_GENERATION_MODEL" \
  --temperature 0 --max-output-tokens 512 --context-window-tokens 8192 \
  --timeout-seconds 60 --max-retries 3 --retry-backoff-seconds 0.5
```

### 3. Evaluate and pair

```bash
evaluate-answers \
  --dataset data/evaluation/angular/qa_dataset.jsonl \
  --documents data/processed/angular/documents.jsonl \
  --prepared-inputs data/benchmark/angular/canonical_v1/inputs \
  --generation fixed_size=data/benchmark/angular/canonical_v1/generation/fixed_size \
  --generation structure_aware=data/benchmark/angular/canonical_v1/generation/structure_aware \
  --generation prompt_based=data/benchmark/angular/canonical_v1/generation/prompt_based \
  --output data/benchmark/angular/canonical_v1/evaluation
```

Evaluation publishes `evaluations.jsonl`, `summary.json`, `paired.jsonl`,
`stats.json`, and a final `manifest.json`. Comparative interpretation begins only
after these canonical artifacts validate.

## Cost and resume safety

For `N` validated questions, input preparation requests at most `N` query
embeddings before cache reuse and generation schedules `3N` strategy/query
requests before generation-cache reuse. Exact expected counts come from validator
and preflight output; no monetary estimate is made here.

Generation is intentionally sequential—there is no hidden concurrency setting.
Retries are bounded by the pinned CLI flags. Each successful answer enters the
content-addressed cache immediately. Provider failure or interruption leaves
explicit partial files without a manifest; rerunning the same command reuses valid
cache entries and retries remaining work. Evaluation rejects partial generation
runs. Committed identities prevent output collisions, and the shared generation
cache is safe because its key binds prompt and generation configuration.

The retained generation-input files include authoritative context pieces and
chunk IDs. They provide a safe future join route for evidence diagnostics by
matching query/context fingerprints; no generation schema redesign or fuzzy
provenance inference is required. Evidence diagnostics remain deferred and are
not a benchmark blocker.
