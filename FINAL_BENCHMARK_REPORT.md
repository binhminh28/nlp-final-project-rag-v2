# Final Benchmark Report

## 1. Executive summary

The canonical production benchmark completed successfully: 140 questions × 3 chunking strategies = 420 valid GPT-5 mini v2 answers. Retrieval and answer results show a tradeoff rather than one universal winner. Under the frozen 2,048-token retrieval budget, `fixed_size` has the best Hit@1 and MRR; `structure_aware` has the best Hit@10, evidence coverage, and all-evidence-retrieved rate. `fixed_size` has the best overall deterministic answer token F1 (0.3830), narrowly ahead of `structure_aware` (0.3766) and `prompt_based` (0.3721).

## 2. Benchmark configuration

- Dataset: 140 questions; fingerprint `9866799ea8f87c6a7c118cbaf0d8c298524757bd753db5d640e1a32234206d74`.
- Corpus: 384 processed Angular documents; fingerprint `cc3ccf3401e0005004525466d6424517719dbc879d0c7b0ed9489fe08d33f32c`.
- Strategies: `fixed_size`, `structure_aware`, `prompt_based`.
- Embedding model: `openai/text-embedding-3-small`; fingerprint `7a71fffabd36fbf5fca9018a3a0cb50c7839af6534a4bf6df7898b335d295353`.
- Retrieval: dense candidate k=50; canonical protocol `same_token_budget`; 2,048-token budget.
- Prepared context: 4,096-token budget; invocation context reference 8,192 tokens.
- Generation: OpenRouter `openai/gpt-5-mini`, temperature 0, max tokens 1,024, reasoning effort `low`, `finish_reason=stop` required.
- Generation fingerprint: `c4f4768ec9b80361dfd0a1e252f74ff348aa4e4c953bcca02761ba345f38b301`.
- Execution: fixed `max_concurrency=8` for each strategy; strategies executed sequentially.
- Evaluation fingerprint: `c6867bffbd9775d3ef9b4ce666ae09f1995a6ccb7a7ef14858bbcdb736c1fa55`; benchmark fingerprint `375983ff4b3c4e84b303d7298c4dd93b44782430bbc1fd6dff41db6f3b60af23`.

## 3. Production qualification history

Generation v1 was rejected for canonical production: it had four repeatable empty/null outputs, 45 of 136 responses ended with `finish_reason=length`, 42 of those 45 were visibly incomplete, and the frozen reproduction consumed all 512 completion tokens in reasoning. These artifacts remain historical only.

Generation v2 qualified at 1,024 max tokens, low reasoning effort, and mandatory normal completion. The v1 and v2 caches are isolated by generation fingerprint.

## 4. Retrieval results

Canonical protocol: `same_token_budget`, candidate k=50, token budget=2,048; n=140 per strategy.

| Strategy | Hit@1 | Hit@5 | Hit@10 | MRR | Recall@5 | Recall@10 | Evidence coverage | All evidence retrieved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_size | 0.8071 | 0.9643 | 0.9643 | 0.8732 | 0.9429 | 0.9429 | 0.6905 | 0.5714 |
| structure_aware | 0.7786 | 0.9786 | 1.0000 | 0.8692 | 0.9500 | 0.9857 | 0.7476 | 0.6357 |
| prompt_based | 0.7571 | 0.9786 | 0.9857 | 0.8508 | 0.9571 | 0.9786 | 0.7320 | 0.6143 |

`fixed_size` is strongest at the first rank; `structure_aware` is strongest on deep/source-evidence coverage. This distinction matters because document-level hits and exact evidence recovery are related but not identical.

## 5. Answer-generation results

| Strategy | n | Token precision | Token recall | Token F1 | Normalized exact | Containment |
|---|---:|---:|---:|---:|---:|---:|
| fixed_size | 140 | 0.2858 | 0.7420 | 0.3830 | 0.0071 | 0.0143 |
| structure_aware | 140 | 0.2798 | 0.7398 | 0.3766 | 0.0071 | 0.0071 |
| prompt_based | 140 | 0.2707 | 0.7555 | 0.3721 | 0.0000 | 0.0143 |

## 6. Results by difficulty

Token F1; sample counts are shown explicitly.

| Difficulty | n | fixed_size | structure_aware | prompt_based |
|---|---:|---:|---:|---:|
| easy | 60 | 0.3892 | 0.3831 | 0.3723 |
| medium | 40 | 0.3863 | 0.3862 | 0.3939 |
| hard | 40 | 0.3702 | 0.3571 | 0.3501 |

`fixed_size` leads easy and hard questions; `prompt_based` leads medium questions by a small margin.

## 7. Results by question type

Token F1; very small strata should not be overinterpreted.

| Question type | n | fixed_size | structure_aware | prompt_based |
|---|---:|---:|---:|---:|
| behavior | 22 | 0.3823 | 0.3770 | 0.3682 |
| cause_effect | 2 | 0.2993 | 0.2976 | 0.3085 |
| comparison | 26 | 0.3333 | 0.3493 | 0.3400 |
| definition | 15 | 0.3152 | 0.3085 | 0.2558 |
| fact | 21 | 0.4836 | 0.4564 | 0.5103 |
| list | 4 | 0.6402 | 0.6101 | 0.4812 |
| mechanism | 7 | 0.3882 | 0.3765 | 0.3547 |
| procedure | 11 | 0.3672 | 0.3756 | 0.3842 |
| security_mechanism | 1 | 0.4649 | 0.4521 | 0.4961 |
| sequence | 3 | 0.4080 | 0.3608 | 0.2929 |
| syntax | 1 | 0.3000 | 0.3077 | 0.3200 |
| synthesis | 2 | 0.3460 | 0.3038 | 0.3202 |
| tradeoff | 10 | 0.3758 | 0.3839 | 0.3888 |
| why | 15 | 0.3532 | 0.3361 | 0.3419 |

## 8. Paired comparisons

Pairs use exact question IDs and token F1. “Wins” are for the left strategy; no post-hoc significance test was introduced.

| Pair | Left wins | Ties | Left losses | Mean paired delta |
|---|---:|---:|---:|---:|
| fixed_size vs structure_aware | 75 | 3 | 62 | +0.0064 |
| fixed_size vs prompt_based | 81 | 3 | 56 | +0.0108 |
| structure_aware vs prompt_based | 60 | 3 | 77 | +0.0045 |

## 9. Generation health

| Strategy | Requested/completed | Cache hits | Provider attempts | Retries | Stop | Other finish | Mean/median input | Mean/median output | Measured reasoning total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_size | 140/140 | 4 | 136 | 0 | 140 | 0 | 2118.5/2132.0 | 236.8/213.0 | 13,902 (136/140) |
| structure_aware | 140/140 | 0 | 140 | 0 | 140 | 0 | 2199.7/2198.5 | 238.2/225.0 | 14,233 (140/140) |
| prompt_based | 140/140 | 0 | 140 | 0 | 140 | 0 | 2142.3/2145.5 | 242.4/229.5 | 13,507 (140/140) |

Across 420 answers: 420 `stop`, zero length completions, empty/null outputs, integrity failures, retries, or permanent failures. Provider diagnostics cover 416 calls; four `fixed_size` answers were valid same-fingerprint v2 cache hits, so no production provider call or fresh reasoning breakdown exists for them.

## 10. Cost

No authoritative billed cost was returned in the persisted provider metadata. Estimates use the routed OpenAI provider list rates shown by [OpenRouter’s provider pricing](https://openrouter.ai/openai/gpt-5-mini/providers): $0.25/M uncached input, $0.025/M cached input, and $2/M output. Output token counts include reasoning usage as reported by the provider.

| Strategy | Provider input | Cached input | Provider output | Estimated cost |
|---|---:|---:|---:|---:|
| fixed_size | 288,169 | 78,976 | 32,480 | $0.1192 |
| structure_aware | 307,962 | 0 | 33,355 | $0.1437 |
| prompt_based | 299,916 | 0 | 33,933 | $0.1428 |
| **Combined** | **896,047** | **78,976** | **99,768** | **$0.4058** |

The estimates distinguish cached input from uncached input and do not treat the four no-call generation-cache hits as new billed usage. The v2 qualification audit’s conclusion remains relevant: 1,024/low materially reduced reasoning use versus 2,048/default.

## 11. Execution characteristics

| Strategy | Max concurrency | Elapsed | Answers/min | 429 | 5xx/529 | Network timeout |
|---|---:|---:|---:|---:|---:|---:|
| fixed_size | 8 | 141.6s | 59.3 | 0 | 0/0 | 0 |
| structure_aware | 8 | 89.5s | 93.9 | 0 | 0/0 | 0 |
| prompt_based | 8 | 116.4s | 72.2 | 0 | 0/0 | 0 |

The fixed-size elapsed value is end-to-end wall time including an execution-channel detach, safe cache resume, and final reconciliation; it is not a clean latency comparison. Structure-aware and prompt-based are clean single-process measurements. Throughput is operational metadata, not a chunk-quality metric.

## 12. Retrieval and answer-quality relationship

| Strategy | All evidence retrieved n | Answer F1 | Evidence missed n | Answer F1 |
|---|---:|---:|---:|---:|
| fixed_size | 80 | 0.4084 | 60 | 0.3490 |
| structure_aware | 89 | 0.3913 | 51 | 0.3510 |
| prompt_based | 86 | 0.3843 | 54 | 0.3527 |

Answer F1 is descriptively higher when all evidence is retrieved for every strategy. This is correlational; it does not establish that chunking caused the answer difference. Structure-aware’s retrieval-coverage advantage did not translate into the highest aggregate lexical answer F1.

## 13. Interpretation and limitations

- Retrieval winner depends on the target: fixed-size for rank-one precision/MRR, structure-aware for deeper and exact-evidence coverage.
- Fixed-size leads overall lexical answer quality and the hard subset; prompt-based leads the medium subset; several question-type winners differ.
- The benchmark covers one Angular corpus, one embedding model, one generation model, and one retrieval/context budget.
- Deterministic lexical evaluation rewards overlap and can undervalue semantically correct paraphrases; no LLM judge or manual post-hoc aliases were introduced.
- OpenRouter routing/provider behavior and prompt caching can vary; cost is estimated, not billed cost.
- Small question-type strata (some n=1–4) have high uncertainty.

## 14. Reproducibility and lineage

Canonical lineage is validated as dataset → processed corpus → chunks → embeddings → indexes → retrieval/protocol → prepared context → generation v2 → deterministic evaluation. No generation-v1, smoke, diagnostic, or failed answer enters the canonical evaluation.

- Generation config: `configs/generation_gpt5mini_v2.json`.
- Prepared inputs: `data/benchmark/angular/canonical_v2/inputs/`.
- Generation: `data/benchmark/angular/canonical_v2/generation/{strategy}/`.
- Generation health: `data/benchmark/angular/canonical_v2/generation_health.json`.
- Retrieval: `data/retrieval/angular/canonical_production_v2/9dab4015ca1ae4c4abda04ccc5809a5e030c793fcc7feff932d04ecfb116a6b7/`.
- Answer evaluation: `data/benchmark/angular/canonical_v2/evaluation/`.

Reproduce answer evaluation:

```bash
evaluate-answers --dataset data/evaluation/angular/qa_dataset.jsonl --documents data/processed/angular/documents.jsonl --prepared-inputs data/benchmark/angular/canonical_v2/inputs \
  --generation fixed_size=data/benchmark/angular/canonical_v2/generation/fixed_size \
  --generation structure_aware=data/benchmark/angular/canonical_v2/generation/structure_aware \
  --generation prompt_based=data/benchmark/angular/canonical_v2/generation/prompt_based \
  --output data/benchmark/angular/canonical_v2/evaluation
```

Status: **PRODUCTION BENCHMARK V2: COMPLETE**
