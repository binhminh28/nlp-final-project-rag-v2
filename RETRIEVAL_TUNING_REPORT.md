# Retrieval Tuning & Ablation Report

This phase remained retrieval-only. It did not change `baseline_v1`,
`baseline_v2`, chunk artifacts, production embeddings, or production indexes,
and it did not implement answer generation.

## A. Benchmark audit

All 64 queries and the union of top-10 results from the three chunking
strategies were audited from content, producing 531 query-source judgments.
The audit added 42 binary-relevant sources across 31 queries, removed none,
and retained 25 `PARTIALLY_RELEVANT` and 3 `AMBIGUOUS` judgments outside binary
gold. One query (`para-006`) remains ambiguous and nine have wording concerns;
no category changed. Per user direction, the resulting immutable
`baseline_v2` is the canonical tuning dataset.

- Dataset fingerprint: `0038353dc25f1790b1fdc5dd8ac2b152efe387b9c0848451143a3b9a739ccf90`
- Audit fingerprint: `9cad75975b217919a7b42b541a98dbe02de9163a68d97f184c3b3f4f83e1f6b2`
- E0 benchmark fingerprint: `b7ac1cf53eec34f4bdfd0fc8ea98ea5a1ff70b122da1f305b45f53b7ce1b90ee`

The metric change from v1 to v2 is an evaluation correction, not retrieval
lift. Hit@1 changed by +0.0781/+0.1094/+0.0938 for fixed/structure/prompt.

## B. Reference benchmark

E0 reproduced the untouched dense configuration against `baseline_v2` in
memory-bounded per-strategy runs. All 192 expected evaluations were present;
rankings and aggregate metrics matched the frozen v2 result exactly, with 64
query-cache hits and zero provider calls per strategy run.

| Strategy | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_size | 0.7344 | 0.9062 | 0.9531 | 1.0000 | 0.8270 | 0.8400 | 0.9526 |
| structure_aware | 0.7656 | 0.9375 | 0.9531 | 1.0000 | 0.8592 | 0.7879 | 0.9243 |
| prompt_based | 0.7031 | 0.9375 | 0.9688 | 1.0000 | 0.8214 | 0.8374 | 0.9412 |

## C. Candidate-depth analysis

Ranks 1–10 were invariant when retrieving deeper candidates. All strategies
reached Hit@10 = 1.0 and Recall@50 = 1.0.

| Strategy | Recall@5 | Recall@10 | Recall@20 | Recall@50 | Diagnosis |
|---|---:|---:|---:|---:|---|
| fixed_size | 0.8400 | 0.9526 | 0.9877 | 1.0000 | ranking-limited below top 5 |
| structure_aware | 0.7879 | 0.9243 | 0.9691 | 1.0000 | ranking-limited below top 5 |
| prompt_based | 0.8374 | 0.9412 | 0.9877 | 1.0000 | ranking-limited below top 5 |

No canonical relevant source was absent at depth 50. Three fixed, three
structure, and two prompt queries had their first relevant result below rank 5.
This justified reranking; it did not justify changing candidate generation.

## D. Candidate overlap

At depth 50, mean source-set Jaccard was 0.6147 for fixed/prompt, 0.5210 for
fixed/structure, and 0.6217 for structure/prompt. No strategy had an exclusive
relevant source under the frozen labels. Cross-strategy fusion was therefore
not implemented.

## E. Experiment registry

| ID | Family | Fingerprint | Configuration | Status |
|---|---|---|---|---|
| E0 | control | `b7ac1cf5…b90ee` | untouched dense, depth 10 | reproduced |
| E1-5/10/20/50 | E1 | `496bbfe3…cb77`, `447a2944…5c30`, `a19651db…bd82`, `436c9c95…9976` | dense depth | complete |
| E2-cap3/2/1 | E2 | `74bbbcee…c5c3`, `b2bd2af0…f5c5`, `192edcfa…2e3` | deterministic source cap | complete |
| E3 | E3 | N/A | deterministic query preprocessing | not justified |
| E4-Q1/Q2 | E4 | `8883a6bf…5fd4`, `e95b5134…f2e` | rewrite-only; original+rewrite RRF | complete |
| E5-5/10/20/50 | E5 | `90085454…b903`, `f5ed385d…643e`, `c24bf628…a39e`, `c67bcdaa…4c70` | BM25 rerank to 5 | complete |
| E6 | E6 | `b85d9713…9b8a` | full-corpus BM25 depth 50 | complete |
| E7 | E7 | `5c5878f2…80b4` | dense-50 + BM25-50, RRF k=60 | complete |

## F. Dense tuning results

Top-10 contained only about 5.8 unique sources on average, so source
concentration justified E2. Caps of 3 made no first-rank changes; cap 2 improved
one strategy-query pair; cap 1 improved six and degraded none.

The strongest pure dense configuration is `structure_aware`, dense depth 50,
maximum one chunk per `relative_path`, return depth 10:

| Metric | E0 structure | E2 cap 1 | Delta |
|---|---:|---:|---:|
| Hit@1 | 0.7656 | 0.7656 | 0.0000 |
| Hit@5 | 0.9531 | 0.9688 | +0.0156 |
| MRR | 0.8592 | 0.8610 | +0.0018 |
| Recall@5 | 0.7879 | 0.8709 | +0.0830 |
| Recall@10 | 0.9243 | 0.9682 | +0.0439 |

E3 was not run: no query had repeated whitespace, all 64 would be altered by
lowercasing, and 17 contain API/code punctuation that must be preserved.
Observed misses clustered in paraphrase/how-to rather than a quantified safe
normalization defect.

## G. Query rewrite results

The fixed GPT-4.1 Mini rewrite configuration used temperature 0, prompt v3,
strict JSON, protected-token fidelity checks, and fingerprint
`7ada0b38…b963`. Three unsafe rewrites were recorded and deterministically
replaced by the original query.

| Variant | Key result | Paired first-rank outcomes |
|---|---|---|
| Q1 rewrite only | inconsistent across strategies | 18 improved, 156 unchanged, 18 degraded |
| Q2 original + rewrite, RRF k=60 | small aggregate lift | 17 improved, 161 unchanged, 14 degraded |

Q2 reached fixed MRR 0.8352, prompt MRR 0.8503, and structure MRR 0.8598,
but retained material query-level regressions. It was not selected.

## H. Reranking results

BM25 was used as a deterministic, provider-free reranker abstraction over
dense pools. Pool 5 was strongest; increasing the pool made ranking worse.

| Dense pool → output | Fixed Hit@1/MRR | Prompt Hit@1/MRR | Structure Hit@1/MRR | Improved / degraded |
|---|---|---|---|---:|
| 5 → 5 | 0.7812 / 0.8594 | 0.7656 / 0.8568 | 0.7500 / 0.8372 | 30 / 20 |
| 10 → 5 | 0.7500 / 0.8573 | 0.7188 / 0.8378 | 0.7031 / 0.8307 | 32 / 33 |
| 20 → 5 | 0.7500 / 0.8526 | 0.6719 / 0.7927 | 0.6250 / 0.7805 | 32 / 39 |
| 50 → 5 | 0.6719 / 0.7948 | 0.6094 / 0.7458 | 0.6094 / 0.7362 | 30 / 50 |

The reranker never introduced a chunk outside its candidate set. Its lift was
not robust enough to justify replacing the simpler diversity policy.

## I. Lexical results

BM25 used Unicode-aware word/code-symbol tokenization, k1=1.2, b=0.75, and
separate per-strategy lexical identities. It was weaker than dense overall.

| Strategy | Hit@1 | Hit@5 | MRR | Recall@5 | Recall@50 |
|---|---:|---:|---:|---:|---:|
| fixed_size | 0.6250 | 0.9219 | 0.7645 | 0.7749 | 0.9831 |
| prompt_based | 0.5781 | 0.9062 | 0.7244 | 0.7541 | 0.9870 |
| structure_aware | 0.5625 | 0.9062 | 0.7052 | 0.7545 | 0.9792 |

At Hit@5, lexical recovered one dense miss per strategy. Dense recovered 3, 5,
and 4 lexical misses for fixed, prompt, and structure respectively. This
complementarity justified E7, but not a lexical replacement.

## J. Hybrid results

E7 fused dense depth 50 and BM25 depth 50 with RRF k=60.

| Strategy | Hit@1 | Hit@5 | MRR | Recall@5 | Recall@50 |
|---|---:|---:|---:|---:|---:|
| fixed_size | 0.7812 | 0.9688 | 0.8743 | 0.8400 | 1.0000 |
| prompt_based | 0.7656 | 1.0000 | 0.8620 | 0.8800 | 1.0000 |
| structure_aware | 0.7344 | 0.9844 | 0.8387 | 0.8361 | 0.9883 |

Versus dense, 28 strategy-query pairs improved, 144 were unchanged, and 20
degraded. The lift is promising but not statistically secure enough to pay the
additional index and fusion complexity in the recommended configuration.

## K. Category analysis

All experiments emitted metrics for all eight categories. For the recommended
structure-aware cap-1 configuration, no category's MRR decreased. Recall@5
rose most for paraphrase (0.583→0.771), cross-document (0.449→0.571), how-to
(0.708→0.812), and code-related (0.812→0.938); API lookup and terminology were
unchanged. Hybrid gains were not consistent across all strategies/categories.

## L. Error analysis

The audit corrected the dominant `narrow relevance label` issue before tuning.
At depth 50 there were zero candidate-generation failures under binary gold;
all early misses were `relevant candidate ranked too low`. Remaining evidence
also includes paraphrased terminology, close siblings, and ambiguous wording.
No chunk/index integrity defect, unresolved chunk ID, duplicate fused chunk,
non-finite score, or invented reranker candidate was found. Uncertain cases
remain `UNKNOWN`/`AMBIGUOUS` rather than being forced into gold.

## M. Statistical uncertainty

Paired bootstrap used 5,000 samples and seed 20260811. For E2 cap 1 on
structure-aware, ΔMRR was +0.0018 with 95% CI [0.0000, 0.0049]; ΔHit@5 was
+0.0156 with CI [0.0000, 0.0469]. For hybrid fixed, ΔMRR was +0.0473 with CI
[-0.0018, 0.0993]; for hybrid prompt it was +0.0406 with CI
[-0.0200, 0.1042]. These hybrid intervals cross zero.

With only 64 queries, small deltas must not be treated as conclusive. The data
is too small for a defensible dev/test split without materially weakening both
sets; repeated tuning on all 64 remains an overfitting risk. A future dataset
development phase should create independently validated `retrieval_dev` and
held-out `retrieval_test`, rather than adding handpicked failures here.

## N. Cost

| Stage | Requests | Input tokens | Output tokens | Cache behavior | Estimated incremental cost |
|---|---:|---:|---:|---|---:|
| Original query embeddings | 0 | 0 | N/A | 64 hits | $0 |
| Rewrite model | 64 | 9,024 | 967 | rerun: 64 hits, 0 misses | about $0.0052 |
| Rewrite-query embeddings | 3 | 389 | N/A | final rerun: 64 hits | small, provider-reported tokens retained |
| BM25 reranking/retrieval/fusion | 0 | 0 | 0 | local deterministic | $0 |

Costs are separated by stage. No LLM/cross-encoder reranker was called.

## O. Determinism and latency

E0 and all final experiment reruns were credential-free after cache population.
The rewrite JSONL rerun was byte-identical with 64 hits, 0 misses, and 0 calls.
The six deterministic E7 files (config, per-query, aggregate, category,
comparison, manifest) were byte-identical after rerun. Runtime telemetry is
excluded from identities.

Local depth-50 dense runs took 24.2s fixed, 56.8s structure, and 36.4s prompt
for 64 queries on the linear JSONL backend. Building all three in-memory BM25
indexes took about 4.0s; E5 ranking took 0.68s and E6 ranking 2.15s. These are
diagnostic totals, not production latency claims; per-query median/p95 was not
used because the local linear backend is not a production serving system.

## P. Tests

Tests cover experiment identity, depth prefixes and recall, source caps and
ties, rewrite caching/fidelity/failure behavior, hand-computed RRF, BM25
determinism/tokenization/Unicode/code symbols/ties, canonical candidates,
paired bootstrap determinism, and an offline transform→retrieve→fuse→evaluate
integration path. The normal suite has no network dependency.

- Full pytest: 192 passed.
- `git diff --check`: passed.
- No incomplete artifact is listed as valid; manifests are final commit markers.

## Q. Recommended retrieval configuration

Recommend exactly: **structure-aware dense retrieval, retrieve 50 candidates,
apply deterministic maximum 1 chunk per `relative_path`, return top 10**
(`192edcfab612f22a9e6ee52fd52a10fb93cb23ef8d23467907a5ec556d3142e3`).

It preserves the best reference Hit@1, raises Hit@5 and source recall, has no
observed first-rank regressions, uses no new provider or index, and adds only a
stable source-cap filter. Its weaknesses are dependence on source-level labels,
an aggressive policy that could hurt future chunk-level multi-evidence tasks,
and uncertainty from the 64-query benchmark. Hybrid fixed/prompt produced
higher point estimates on some metrics, but its CI crosses zero and it adds a
lexical index, fusion stage, and more query-level regressions.

## R. Frozen experiment identities

| Role | Fingerprint |
|---|---|
| Reference dataset | `0038353dc25f1790b1fdc5dd8ac2b152efe387b9c0848451143a3b9a739ccf90` |
| Reference benchmark | `b7ac1cf53eec34f4bdfd0fc8ea98ea5a1ff70b122da1f305b45f53b7ce1b90ee` |
| Best dense / recommended | `192edcfab612f22a9e6ee52fd52a10fb93cb23ef8d23467907a5ec556d3142e3` |
| Best reranked | `90085454b61ce15c6c7d02769c6de14bce5d825834c1b67c046d0e9ee62eb903` |
| Best lexical | `b85d97130a497c085eac4ba46cbfe4565c45d1d828d317359bc26ac3fb609b8a` |
| Best hybrid | `5c5878f2a24d0ade45a69e7e34c33709a6fd094a5ca18669d681f41b1b2580b4` |

## S. Final verdict

All ten gates pass under the user-frozen `baseline_v2`. The result is suitable
to feed a separately constructed future RAG evaluation/generation phase, with
the benchmark-size and source-level-label limitations above carried forward.

SAFE TO PROCEED TO RAG DATASET & ANSWER GENERATION PHASE
