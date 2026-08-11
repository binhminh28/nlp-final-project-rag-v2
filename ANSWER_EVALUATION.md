# Deterministic Answer Evaluation

Answer Evaluation is a terminal, offline consumer of the canonical QA dataset
and frozen answer-generation artifacts. Its invariant is:

> Evaluation measures frozen generated answers against frozen canonical
> references. It may observe upstream lineage for diagnostics, but it never
> alters or regenerates the experiment being measured.

## Input and commitment contract

Canonical evaluation accepts only a `generation_run_v1` directory with a valid,
complete `manifest.json`. `answers.jsonl`, `failures.jsonl`, and `stats.json`
without that final commit marker are partial state and are rejected. The loader
validates the manifest schema and run fingerprint; exact ordered query/question/
context identity; answer and failure counts; unique and known query IDs; question
identity; strategy; generation config; context lineage; dataset fingerprint on
every answer; and every `AnswerResult.result_fingerprint`. Extra query IDs fail
validation. Expected QA queries without a result remain explicit
`missing_generation_result` rows.

The evaluator imports no provider or retrieval service and does not retrieve,
rerank, rebuild context, invoke an LLM, fill missing answers, or edit an upstream
artifact. Partial-run evaluation is intentionally unsupported.

## Frozen evaluation configuration

`answer_evaluation_config_v1` fingerprints every score-affecting policy:

- normalization: `answer_normalization_v1`;
- tokenization: `unicode_word_or_symbol_v1`;
- enabled metrics;
- multiple-reference best-match and tie policy;
- failed/missing generation policies;
- quality and success-aware aggregation policy;
- exact `question_type` category policy;
- deferred evidence-diagnostic policy.

Output paths, runtimes, and other operational values are excluded.

## Normalization and deterministic metrics

`answer_normalization_v1` applies Unicode NFKC, Unicode `casefold`, surrounding
whitespace trimming through whitespace collapse, and no other transformation.
It does not remove punctuation, articles, numbers, units, negation, code symbols,
mathematical symbols, versions, or named entities.

`unicode_word_or_symbol_v1` tokenizes the normalized string into Unicode word
runs or individual non-whitespace symbols. It is an evaluation lexical tokenizer,
not the generator model tokenizer. Duplicate tokens use multiset overlap.

The implemented metrics are:

- `normalized_exact_match`: equality after `answer_normalization_v1`;
- `token_precision`, `token_recall`, and `token_f1`: multiset token overlap;
- `normalized_containment`: secondary diagnostic equal to one only when the
  normalized reference token sequence occurs contiguously in the prediction.

Both-empty strings score one for EM and token precision/recall/F1. If exactly one
side is empty those metrics score zero. Containment is not a primary score and
can produce false positives when a verbose answer repeats a reference without
using it correctly.

The current `evidence_qa_dataset_v1` has one required, non-empty string answer per
query. There are no alias, multiple-answer, unanswerable, or answer-normalization
metadata fields. The metric implementation nevertheless supports multiple
references without concatenation: each metric takes its best reference, and
ties select the lowest reference index deterministically.

## Failure policy and denominators

Per-query statuses are `evaluated`, `generation_failed`, and
`missing_generation_result`. Failed or missing generations keep a null generated
answer and null metric values; no text is fabricated.

Every overall and exact-`question_type` category aggregate reports:

- total expected, successful, failed, evaluated, and missing counts;
- generation success rate and evaluation coverage;
- successful-answer metric mean and its non-null denominator (`metric_counts`);
- end-to-end zero-filled metric mean and the full expected denominator
  (`success_aware_metric_counts`).

This prevents failures from disappearing while retaining a quality-only view of
successful answers.

## Lineage, artifacts, and fairness

Each per-query fingerprint binds the query/category/question, gold reference,
generated answer and generation result identity, status, metrics, prompt/context/
generation lineage, dataset fingerprint, and evaluation config fingerprint. Each
aggregate fingerprint binds its ordered per-query fingerprints, strategy,
dataset, config, generation-run identity, counts, denominators, and category
aggregates. The benchmark fingerprint binds all strategies and paired rows.

Publication is transactional and writes `manifest.json` last:

```text
evaluations.jsonl
summary.json
paired.jsonl
stats.json
manifest.json
```

The committed-artifact validator recomputes row, aggregate, paired, and benchmark
fingerprints. A conflicting committed identity is never overwritten.

All strategies use the same normalization, metric, missing/failure, aggregate,
category, and diagnostic code. Strategy is only provenance and input selection.
`paired.jsonl` aligns every QA query with each supplied strategy's statuses,
metrics, result/evaluation fingerprints, gold answer, and category. It performs
no ranking, winner selection, significance test, or causal attribution.

Gold evidence can be mapped reliably to chunks by the retrieval evidence layer,
but committed `AnswerResult` rows do not contain selected context piece IDs.
Therefore answer evaluation reports `evidence_diagnostics: null`; joining an
authoritative context artifact is deferred rather than using fuzzy inference.

No LLM-as-a-judge, embedding similarity, learned metric, faithfulness model,
hallucination classifier, retry, or answer regeneration is implemented.

## CLI

```bash
evaluate-answers \
  --dataset path/to/canonical-qa.jsonl \
  --documents data/processed/angular/documents.jsonl \
  --generation fixed_size=path/to/fixed-run \
  --generation structure_aware=path/to/structure-run \
  --generation prompt_based=path/to/prompt-run \
  --output data/answer-evaluation/angular/run-v1
```

The CLI is entirely offline. All supplied runs must use the same dataset and
generation configuration and must identify distinct strategies.
