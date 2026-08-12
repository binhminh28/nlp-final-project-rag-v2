# GPT-5 Mini generation budget and reasoning audit

## Decision

**Outcome A — keep `openai/gpt-5-mini`.** The production failures are explained by a shared completion budget that was entirely consumed by reasoning. The smallest tested configuration that completed the full diagnostic set and a fresh replication is:

- config: `configs/generation_gpt5mini_v2.json`
- generation config fingerprint: `c4f4768ec9b80361dfd0a1e252f74ff348aa4e4c953bcca02761ba345f38b301`
- completion allowance: 1024 via OpenRouter `max_tokens`
- reasoning: `{"effort": "low"}`
- completion integrity: accept only `finish_reason="stop"`

This audit did not run or mix a new 420-answer production benchmark. The existing 136 v1 answers, partial artifacts, prepared contexts, and caches remain untouched.

## Phase 0: preserved state

The worktree was clean at audit start. The frozen production state is `generation config v1 — failed production qualification`:

- dataset fingerprint: `9866799ea8f87c6a7c118cbaf0d8c298524757bd753db5d640e1a32234206d74`
- generation fingerprint: `1045c3382003284e5fc02ebe1b86834d45423dc08313e773c4409acf4bad6cb6`
- fixed-size results: 136 answers and 4 failures; no commit manifest exists for the partial run
- failures: `q_hard_019`, `q_hard_031`, `q_hard_032`, `q_hard_040`
- the 146 existing generation-cache records all belong to v1

No QA, chunking, embedding, indexing, retrieval, context, gold-answer, or evaluator-scoring code/data was changed.

## Phase 1: actual v1 OpenRouter request

The serialized request is `POST https://openrouter.ai/api/v1/chat/completions` with:

```json
{
  "model": "openai/gpt-5-mini",
  "messages": [
    {"role": "system", "content": "<answer_system_v1>"},
    {"role": "user", "content": "Question:\n<exact question>\n\nContext:\n<exact prepared context>\n\nAnswer:"}
  ],
  "temperature": 0.0,
  "max_tokens": 512
}
```

- A. The 512 limit is controlled by `max_tokens`. Neither `max_completion_tokens` nor `max_output_tokens` is sent over HTTP (`max_output_tokens` is only the local config name).
- B. v1 sends no `reasoning`, `reasoning_effort`, reasoning budget, or enable/disable field. OpenRouter currently describes GPT-5 mini reasoning as mandatory, default effort medium, with `high`, `medium`, `low`, and `minimal` supported.
- C. The adapter previously read only `choices[0].message.content`, `finish_reason`, and aggregate prompt/completion usage. It discarded `reasoning`, `reasoning_details`, `native_finish_reason`, refusal/tool/annotation structure, cached/reasoning-token detail, request ID, response model/provider, cost, and provider error metadata.

The payload does serialize `temperature: 0.0`, but the current OpenRouter model metadata does not list temperature among GPT-5 mini's supported parameters. The audit therefore does not claim provider-level sampling determinism from this field; repeated v1 responses did vary.

There are no response-format, stop, or provider-routing fields in v1. Routing is OpenRouter default. Timeout is 60 seconds. The provider makes at most 4 attempts (3 retries) with exponential delays of 0.5, 1.0, and 2.0 seconds, except `Retry-After` is honored for retryable HTTP responses and capped at 60 seconds.

The system prompt is: “Answer the question using only the supplied context. Remain faithful to the evidence. If the context is insufficient, state that clearly. Do not fabricate unsupported information.”

## Phase 2: 136-answer artifact audit

| Statistic | Result |
|---|---:|
| `finish_reason=stop` | 91 |
| `finish_reason=length` | 45 |
| other finish reasons | 0 |
| provider input tokens, min / median / mean / max | 1,893 / 2,132 / 2,119.118 / 2,197 |
| output tokens, min / median / mean / max | 106 / 425 / 395.897 / 512 |
| answers exactly at 512 | 45 |
| characters, min / median / mean / max | 8 / 362.5 / 416.515 / 1,294 |
| words, min / median / mean / max | 2 / 55.5 / 61.522 / 187 |

Output-token distribution: 1 in 0–127, 23 in 128–255, 33 in 256–383, 34 in 384–511, and 45 at exactly 512.

A transparent terminal-punctuation indicator flags 42/45 `length` answers as visibly incomplete, compared with 1/91 `stop` answers. This is an association, not a semantic-quality metric, but it matches manual inspection:

- severe: `q_hard_030` contains only `- Signal` (8 characters)
- moderate: `q_easy_001` ends at the heading `How to enable`
- normal-looking despite `length`: `q_medium_019` ends in a grammatical sentence
- `stop` control: `q_easy_024` is a complete one-sentence list of the three form approaches

Therefore `finish_reason=length` is not a healthy canonical generation even when visible content is non-empty.

### Failure-input comparison

The four failures have rendered contexts of 1,987, 2,070, 1,988, and 2,048 tokens. Across all 40 hard inputs, the range is 1,947–2,081 (median 2,066). Their piece counts (4–6), document counts (2–5), question lengths (20–22 words), markdown/code markers, and character counts fall within the hard-question population, except `q_hard_019` has the maximum angle-bracket count because its evidence discusses Angular template elements.

Three failures are comparison/cross-document-synthesis questions, but 9 other comparison questions and 8 other cross-document-synthesis questions succeeded. `q_hard_032` is a sequence/state-transition question, and another sequence hard question succeeded. Successful hard questions exist at identical or nearly identical context sizes: `q_hard_006`, `q_hard_013`, and `q_hard_014` also have 2,070-token contexts; `q_hard_034` is within 3 tokens of `q_hard_040`; and `q_hard_016` is within 7–8 tokens of `q_hard_019`/`q_hard_032`.

Conclusion: no shared input anomaly explains the four failures independently of the generation budget. Their topics and source documents differ, and all four complete under the selected budget/reasoning configuration.

## Phases 3–5: diagnostics and reproduction

Opt-in diagnostics now record safe per-attempt structure keyed by query ID and prompt fingerprint. They include HTTP status, request ID, model/provider, choice count, finish reasons, content shape/length, reasoning/refusal/tool/annotation presence, prompt/completion/reasoning/cached tokens, and provider/transport errors. Diagnostics and raw-response paths are operational CLI options and are excluded from generation identity. Authorization and prompts are never written.

The targeted frozen-v1 reproduction of `q_hard_019` captured two provider attempts before the external command transport ended. Both attempts were identical in the relevant structure:

- HTTP 200, one choice, OpenAI provider, GPT-5 mini
- `message.content = null`
- `finish_reason = "length"`; native reason `max_output_tokens`
- completion tokens = 512; reasoning tokens = 512
- `reasoning` and `reasoning_details` present
- no refusal, tool calls, annotations, or provider error

Classification: **Case C + Case E**. Reasoning exists while visible final content is absent, and token exhaustion is explicit. It is not Case F: there is no hidden provider error. Each captured attempt cost $0.00110975; the historical four-attempt pattern is approximately $0.004439 for no visible answer.

The raw responses are isolated under ignored `data/benchmark/angular/generation_audit/` and are non-canonical.

## Phases 6–10: controlled experiment

The fixed 13-case set contains:

- four failures: `q_hard_019`, `q_hard_031`, `q_hard_032`, `q_hard_040`
- six v1 truncations: `q_hard_030`, `q_hard_025`, `q_medium_027`, `q_hard_009`, `q_easy_001`, `q_medium_019`
- three `stop` controls: `q_easy_024`, `q_hard_006`, `q_hard_016`

All variants reused exact prepared fixed-size inputs and changed only completion budget/reasoning/integrity identity.

| Variant | Result | Mean completion | Mean reasoning | Mean visible chars | Measured cost |
|---|---|---:|---:|---:|---:|
| 512 / default | existing baseline: 4 empty failures; 45/136 length overall | — | — | — | — |
| 1024 / default | did not pass; at least `q_hard_040` durably hit `length` | — | — | — | $0.019562 for captured raw responses |
| 2048 / default | 13/13 `stop`, first attempts | 748.15 | 462.77 | 1,205.85 | $0.020844 (input cache active) |
| 1024 / low | 13/13 `stop`, first attempts | 408.54 | 167.38 | 1,091.15 | $0.017486 (uncached input) |
| 1024 / low replication | 4/4 former failures `stop`, first attempts | 574.00 | 192.00 | 1,781.50 | $0.004905 |

The 1024/low candidate used 63.8% fewer reasoning tokens than 2048/default. At published uncached token rates, normalizing both 13-case runs removes the input-cache advantage: 2048/default is approximately $0.026316 versus $0.017486 for 1024/low, a 33.6% reduction. Extrapolating the diagnostic mean gives roughly $0.188 for 140 fixed-size answers, but the set is hard-case-heavy and production caching/routing can change this estimate.

Observed clean-run wall time was about 102.5 seconds for 2048/default and 77.4 seconds for 1024/low (24.5% lower). These are end-to-end observations, not controlled provider latency benchmarks.

Using the unchanged repository metrics on the 13 cases:

| Variant | token precision | token recall | token F1 |
|---|---:|---:|---:|
| 2048 / default | 0.2044 | 0.7652 | 0.3064 |
| 1024 / low | 0.2496 | 0.7392 | 0.3470 |

Low reasoning modestly reduced lexical recall but improved precision/F1 and preserved correct, structurally complete controls. Manual ending inspection found no obvious truncation in either passing run. No context overflow, empty content, or parser/schema failure occurred.

No 4096 test was justified because 2048/default passed, and 1024/low then passed both the full set and a fresh four-failure replication.

## Phases 11–13: v2 identity, cache, and integrity

The selected v2 identity explicitly records provider, model, temperature, `max_tokens` request contract, reasoning effort, timeout/retry policy, prompt/system versions, the 4,096-token prepared-context budget reference, the 8,192-token invocation window, default provider routing, empty stop list, and response handling contract. Its fingerprint is `c4f4768ec9b80361dfd0a1e252f74ff348aa4e4c953bcca02761ba345f38b301`, distinct from v1.

V1 identity remains reproducible: constructing the documented v1 CLI config still yields `1045c3382003284e5fc02ebe1b86834d45423dc08313e773c4409acf4bad6cb6`. Numeric fields are normalized so semantically identical `0` and `0.0` no longer create different identities.

V2 uses `nonempty_text_require_stop_v2`: non-empty content with any finish reason other than `stop` raises `GenerationIntegrityError`, is written as an explicit generation failure with finish/token/visible-length diagnostics, and is not cached. Evaluation metrics were not changed.

Cache keys already include the generation fingerprint, so v1 cannot satisfy v2. Identical v2 prompts reuse v2 cache entries. Partial runs remain resumable, and a manifest is still written only after every requested answer completes. Production v2 must use a new output lineage and must not copy the 136 v1 answers; upstream prepared contexts remain reusable.

## Phase 14: verification

Focused tests cover normal, empty, null, and non-text provider content; reasoning metadata; stop/length finish reasons; v2 integrity rejection; fingerprint changes; v1/v2 cache separation; v2 reuse; resumability; and manifest-last commitment.

- focused preflight selection: **87 passed** (the prior 78 plus 9 new generation tests)
- focused generation file: **32 passed**
- full suite: **319 passed** (the prior 310 plus 9 new tests)
- `git diff --check`: passed

## Provider references

- OpenRouter reasoning controls: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
- Current model metadata: https://openrouter.ai/api/v1/models
- Chat completions contract: https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion
