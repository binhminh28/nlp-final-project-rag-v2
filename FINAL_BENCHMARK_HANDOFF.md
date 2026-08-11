# Final Benchmark Handoff

## Readiness versus completion

The implementation pipeline is complete through deterministic paired answer
evaluation. The production experiment is **not complete**: the teammate-owned
canonical QA dataset has not been delivered. Synthetic fixtures in `tests/` are
plumbing checks only and must never be used for research results.

Expected handoff path:

```text
data/evaluation/angular/canonical_qa_v1.jsonl
```

This path is explicitly unignored while other generated `data/` artifacts remain
ignored. The repository owns validation, fingerprinting, compatibility checks,
orchestration, and artifact publication. The dataset teammate owns authored
questions, gold answers, evidence, `question_type`, difficulty, notes, and content
review. Validation never repairs or rewrites their semantic content.

## Authoritative QA contract

The selected loader fixes the external schema to `evidence_qa_dataset_v1`. JSONL
is preferred; a UTF-8 JSON array is also accepted. JSONL row order and JSON key
order do not affect identity because validated records are sorted by `id` before
fingerprinting. File name, location, timestamps, and runtime paths are excluded.

| Field | Type | Required | Meaning and validation | Pipeline use |
|---|---|---:|---|---|
| `id` | string | yes | Query ID; non-empty, whitespace-free, globally unique | Retrieval/generation identity; evaluation alignment |
| `doc_id` | string | yes | Exact canonical processed-document ID, e.g. `angular:guide/signals.md`; must exist | Evidence mapping/evaluation only |
| `question` | string | yes | Non-empty; trim-only normalization at load | Exact retrieval and generation question; evaluation provenance |
| `answer` | string | yes | Non-empty single gold reference; no aliases/multiple-answer encoding | Evaluation only |
| `evidence_sentences` | array | yes | Strings or provenance objects; may be empty only when sections provide evidence | Evidence validation/diagnostics only |
| `evidence_sections` | string array | yes | Canonical heading or heading path; may be empty only when sentence evidence exists | Evidence validation/diagnostics only |
| `question_type` | string | yes | Non-empty open-ended canonical category; no separate `category` field | Evaluation grouping/paired metadata only |
| `difficulty` | string | yes | Non-empty teammate-assigned label | Evaluation/paired metadata only |
| `notes` | string or null | no | Curation/audit metadata | Evaluation metadata only |

An evidence provenance object has exactly:

```json
{
  "text": "Exact evidence text",
  "block_index": 12,
  "char_start": 5,
  "char_end": 24
}
```

`text` is required. `block_index` is optional and zero-based. Character offsets
must be supplied together, require `block_index`, are zero-based half-open source
offsets, and must select exactly `text` from that normalized document block. If
only `block_index` is supplied, the text must occur in that block. Plain evidence
text must occur in the declared document. Section matching uses Unicode NFKC,
case folding, and whitespace collapse for validation while fingerprinting keeps
the original semantic strings.

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

After the teammate places the file at the expected path:

```bash
validate-qa-dataset \
  --dataset data/evaluation/angular/canonical_qa_v1.jsonl \
  --documents data/processed/angular/documents.jsonl
```

This rejects malformed UTF-8/JSON/JSONL, duplicate JSON keys or IDs, unknown or
missing fields, empty IDs/questions/answers/categories/difficulty, unknown
documents, malformed/out-of-range spans, source-text/span disagreement, missing
evidence, and unsupported schema usage. It prints the canonical fingerprint and
query count without modifying the file.

The read-only three-strategy preflight validates the processed corpus, complete
chunk artifacts, chunk coverage, full embedding artifact hashes, index entries,
all manifest lineage, shared embedding configuration, protocol/context/
generation/evaluation configurations, credentials when requested, and output
conflict markers:

```bash
export ANSWER_GENERATION_MODEL='PINNED_PRODUCTION_MODEL_ID'

benchmark-preflight \
  --dataset data/evaluation/angular/canonical_qa_v1.jsonl \
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

Current checkout result from the offline default preflight:

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
  --dataset data/evaluation/angular/canonical_qa_v1.jsonl \
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
  --dataset data/evaluation/angular/canonical_qa_v1.jsonl \
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
