# Offline Chunk & Context Efficiency Analysis

## A. Validation & Lineage

**PASS.** Exactly 140 queries, three strategies, 420 unique query-strategy rows, and 3,948 selected pieces were validated. Every context, answer, evaluation, selected chunk ID, raw-token sum, and frozen fingerprint resolved; no stale lineage was admitted.

- Dataset: `9866799ea8f87c6a7c118cbaf0d8c298524757bd753db5d640e1a32234206d74`
- Corpus: `cc3ccf3401e0005004525466d6424517719dbc879d0c7b0ed9489fe08d33f32c`
- Retrieval: `9dab4015ca1ae4c4abda04ccc5809a5e030c793fcc7feff932d04ecfb116a6b7`
- Generation: `c4f4768ec9b80361dfd0a1e252f74ff348aa4e4c953bcca02761ba345f38b301`
- Evaluation: `c6867bffbd9775d3ef9b4ce666ae09f1995a6ccb7a7ef14858bbcdb736c1fa55`
- Freeze: `9c4a0242e17dadb92b7f74479114543aa1f28dd73c63cef04a045b2eef7c7c15`

## B. Executive Summary

Fixed-size produced the strongest observed mean answer Token F1 and rank-one retrieval, but its answer advantage is not statistically significant in the existing paired tests. Structure-aware produced the smallest chunks, packed the raw-token budget most tightly, and achieved the strongest deep/evidence retrieval. Prompt-based preserved gold evidence with the least measured fragmentation. Structure-aware’s many small chunks also caused the largest context-label overhead and deepest rank consumption. Prompt-based occupied the middle ground in chunk granularity and context overhead; it had the strongest medium-difficulty answer F1 but no overall statistically supported advantage.

## C. Chunk Size & Granularity

| Strategy | Chunks | Total token occurrences | Mean | Median | Stddev | p10 | p90 | p95 | p99 | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| Fixed-size | 1,300 | 581,181 | 447.1 | 512.0 | 120.6 | 219.9 | 512.0 | 512.0 | 512.0 | 512 |
| Structure-aware | 3,162 | 522,165 | 165.1 | 129.0 | 127.2 | 32.0 | 365.9 | 447.0 | 501.0 | 512 |
| Prompt-based | 2,054 | 524,005 | 255.1 | 237.0 | 138.5 | 82.0 | 463.0 | 490.0 | 508.0 | 512 |

Fixed-size is strongly concentrated at the 512-token ceiling. Structure-aware is markedly more granular. Prompt-based lies between them. Smaller chunks are not inherently better: they increase selectable units and formatting overhead and may reduce local context per chunk.

Size-band counts and percentages are in `chunk_distribution.csv`.

## D. Corpus Redundancy

Fixed-size has 522,558 source tokens and 581,181 chunk-token occurrences: 58,623 exact overlap occurrences, expansion ratio 1.1122, or 11.22% overhead.

Structure-aware declares no overlap, and prompt-based uses zero chunk overlap/exact source slices. Their exact corpus-level duplication was not independently measurable from a common global alignment, so the report does not convert those policies into invented duplication counts.

## E. Evidence Preservation & Fragmentation

| Strategy | Evidence items | Full coverage | Single-chunk complete | Mean minimum chunks | Median | p95 | Max | >1 chunk |
|---|---|---|---|---|---|---|---|---|
| Fixed-size | 233 | 100.0% | 96.6% | 1.034 | 1 | 1.0 | 2 | 3.4% |
| Structure-aware | 233 | 100.0% | 99.1% | 1.009 | 1 | 1.0 | 2 | 0.9% |
| Prompt-based | 233 | 100.0% | 99.6% | 1.004 | 1 | 1.0 | 2 | 0.4% |

Fragmentation is the minimum number of distinct canonical chunks needed to cover every mapped sentence unit in one evidence item. Overlapping chunks that independently contain the same complete evidence still yield fragmentation 1.

Intrinsic token density was not calculated: source-character spans can be reconstructed, but exact canonical evidence-token intersections and BPE boundary attribution are not persisted consistently across all three strategies.

## F. Context Budget Utilization

| Strategy | Mean chunks | Mean raw | Median raw | Mean utilization | p5 / p95 utilization | Mean unused | ≥95% | ≥99% |
|---|---|---|---|---|---|---|---|---|
| Fixed-size | 4.94 | 2,016.6 | 2,029.0 | 98.47% | 94.2 / 100.0% | 31.4 | 92.9% | 52.9% |
| Structure-aware | 14.64 | 2,039.3 | 2,042.0 | 99.57% | 98.6 / 100.0% | 8.7 | 100.0% | 86.4% |
| Prompt-based | 8.62 | 2,027.0 | 2,030.5 | 98.97% | 97.2 / 100.0% | 21.0 | 100.0% | 57.9% |

## G. Retrieval Rank Behavior

| Strategy | Mean max rank | Mean selected rank | Mean skipped ranks | Selected top 5 | Selected top 10 | Selected >10 |
|---|---|---|---|---|---|---|
| Fixed-size | 13.47 | 4.80 | 8.53 | 86.0% | 90.8% | 9.2% |
| Structure-aware | 29.37 | 9.59 | 14.74 | 36.0% | 68.6% | 31.4% |
| Prompt-based | 18.71 | 6.36 | 10.09 | 60.2% | 86.4% | 13.6% |

Deeper rank use is a packing behavior, not intrinsically a quality benefit.

## H. Retrieval Effectiveness

| Strategy | Hit@1 | Hit@5 | MRR | Recall@10 | Evidence coverage | All evidence retrieved |
|---|---|---|---|---|---|---|
| Fixed-size | 0.8071 | 0.9643 | 0.8732 | 0.9429 | 0.6905 | 57.1% |
| Structure-aware | 0.7786 | 0.9786 | 0.8692 | 0.9857 | 0.7476 | 63.6% |
| Prompt-based | 0.7571 | 0.9786 | 0.8508 | 0.9786 | 0.7320 | 61.4% |

Fixed-size leads Hit@1 and MRR. Structure-aware leads Recall@10, exact evidence coverage, and all-evidence retrieval.

## I. Answer Quality

| Strategy | Mean Token F1 | Median | Easy | Medium | Hard |
|---|---|---|---|---|---|
| Fixed-size | 0.3830 | 0.3675 | 0.3892 | 0.3863 | 0.3702 |
| Structure-aware | 0.3766 | 0.3581 | 0.3831 | 0.3862 | 0.3571 |
| Prompt-based | 0.3721 | 0.3560 | 0.3723 | 0.3939 | 0.3501 |

All three Holm-adjusted paired permutation and Wilcoxon comparisons are non-significant at α=0.05. The differences are observed descriptive differences, not demonstrated superiority.

## J. Context & Provider Token Efficiency

| Strategy | Rendered mean | Overhead mean | Overhead ratio | F1/1K raw mean | F1/1K rendered mean | Provider input mean / total | Completion mean / total | Reasoning coverage |
|---|---|---|---|---|---|---|---|---|
| Fixed-size | 2,043.7 | 27.1 | 1.35% | 0.1900 | 0.1875 | 2,118.5 / 296,595 | 236.8 / 33,153 | 136/140 |
| Structure-aware | 2,114.8 | 75.5 | 3.70% | 0.1847 | 0.1780 | 2,199.7 / 307,962 | 238.2 / 33,355 | 140/140 |
| Prompt-based | 2,064.8 | 37.8 | 1.86% | 0.1836 | 0.1803 | 2,142.3 / 299,916 | 242.4 / 33,933 | 140/140 |

Raw context, rendered context, and provider input are separate quantities. Reasoning is included in completion usage and is not double-counted. F1-per-token ratios are secondary because the protocol keeps raw tokens close to the same budget.

## K. Analysis by Difficulty / Question Type

Difficulty results appear above. Prompt-based has the highest observed medium F1; fixed-size leads easy and hard.

### Question types with at least five questions

| Question type | n | Fixed F1 | Structure F1 | Prompt F1 |
|---|---|---|---|---|
| behavior | 22 | 0.3823 | 0.3770 | 0.3682 |
| comparison | 26 | 0.3333 | 0.3493 | 0.3400 |
| definition | 15 | 0.3152 | 0.3085 | 0.2558 |
| fact | 21 | 0.4836 | 0.4564 | 0.5103 |
| mechanism | 7 | 0.3882 | 0.3765 | 0.3547 |
| procedure | 11 | 0.3672 | 0.3756 | 0.3842 |
| tradeoff | 10 | 0.3758 | 0.3839 | 0.3888 |
| why | 15 | 0.3532 | 0.3361 | 0.3419 |

### Single- versus multi-evidence

| Group | Strategy | n | Fragmentation | Selected chunks | Budget use | Retrieval evidence | F1 |
|---|---|---|---|---|---|---|---|
| single | Fixed-size | 60 | 1.017 | 5.00 | 98.07% | 0.593 | 0.3892 |
| single | Structure-aware | 60 | 1.000 | 15.68 | 99.64% | 0.710 | 0.3831 |
| single | Prompt-based | 60 | 1.000 | 8.97 | 99.07% | 0.677 | 0.3723 |
| multi | Fixed-size | 80 | 1.040 | 4.90 | 98.76% | 0.763 | 0.3782 |
| multi | Structure-aware | 80 | 1.012 | 13.85 | 99.53% | 0.776 | 0.3717 |
| multi | Prompt-based | 80 | 1.006 | 8.36 | 98.90% | 0.774 | 0.3720 |

Detailed evidence-count and all question-type aggregates are in `chunk_efficiency_summary.json`. Small groups are descriptive and must not be overinterpreted.

## L. Correlation & Query-Level Win Analysis

| Strategy | Token F1 vs | Spearman ρ | n |
|---|---|---|---|
| Fixed-size | selected_chunk_count | -0.017 | 140 |
| Fixed-size | raw_selected_chunk_tokens | -0.060 | 140 |
| Fixed-size | rendered_context_tokens | -0.072 | 140 |
| Fixed-size | formatting_overhead_tokens | -0.062 | 140 |
| Fixed-size | maximum_selected_rank | 0.036 | 140 |
| Fixed-size | mean_evidence_item_fragmentation | -0.011 | 140 |
| Fixed-size | retrieved_evidence_coverage | 0.162 | 140 |
| Structure-aware | selected_chunk_count | 0.065 | 140 |
| Structure-aware | raw_selected_chunk_tokens | -0.047 | 140 |
| Structure-aware | rendered_context_tokens | 0.061 | 140 |
| Structure-aware | formatting_overhead_tokens | 0.073 | 140 |
| Structure-aware | maximum_selected_rank | 0.102 | 140 |
| Structure-aware | mean_evidence_item_fragmentation | 0.019 | 140 |
| Structure-aware | retrieved_evidence_coverage | 0.123 | 140 |
| Prompt-based | selected_chunk_count | -0.073 | 140 |
| Prompt-based | raw_selected_chunk_tokens | -0.058 | 140 |
| Prompt-based | rendered_context_tokens | -0.077 | 140 |
| Prompt-based | formatting_overhead_tokens | -0.061 | 140 |
| Prompt-based | maximum_selected_rank | 0.016 | 140 |
| Prompt-based | mean_evidence_item_fragmentation | -0.018 | 140 |
| Prompt-based | retrieved_evidence_coverage | 0.135 | 140 |

These are exploratory associations and do not establish causality.

| Pair | Left wins | Ties | Right wins |
|---|---|---|---|
| Fixed-size vs Structure-aware | 75 | 3 | 62 |
| Fixed-size vs Prompt-based | 81 | 3 | 56 |
| Structure-aware vs Prompt-based | 60 | 3 | 77 |

Mean left-minus-right characteristic differences within win subsets:

| Pair | Subset | Fragmentation Δ | Chunk-count Δ | Budget-use Δ | Max-rank Δ |
|---|---|---|---|---|---|
| Fixed-size vs Structure-aware | left wins | 0.024 | -9.21 | -1.43 pp | -16.31 |
| Fixed-size vs Structure-aware | right wins | 0.022 | -10.16 | -0.73 pp | -15.95 |
| Fixed-size vs Prompt-based | left wins | 0.012 | -3.42 | -0.73 pp | -6.59 |
| Fixed-size vs Prompt-based | right wins | 0.048 | -3.98 | -0.15 pp | -3.12 |
| Structure-aware vs Prompt-based | left wins | 0.000 | 6.32 | 0.51 pp | 9.45 |
| Structure-aware vs Prompt-based | right wins | 0.006 | 5.79 | 0.68 pp | 10.90 |

These conditional patterns are diagnostic and not causal evidence.

## M. Representative Cases

### `q_easy_021` — fixed-size clear win

easy; fact; 1 evidence item(s). Since which Angular version are reactive forms strictly typed by default?

| Strategy | Chunks | Raw/rendered | Ranks | Fragmentation | Hit@1 / evidence coverage | F1 |
|---|---|---|---|---|---|---|
| Fixed-size | 4 | 1899/1921 | 1,2,3,4 | 1.00 | 1 / 1.00 | 1.0000 |
| Structure-aware | 17 | 2037/2129 | 1,2,3,4,5,6,7,8,9,10,11,12,14,16,17,31,45 | 1.00 | 1 / 1.00 | 0.7500 |
| Prompt-based | 6 | 2040/2070 | 1,2,3,4,5,8 | 1.00 | 1 / 1.00 | 0.7500 |

### `q_easy_052` — structure-aware clear win

easy; fact; 1 evidence item(s). What file extension does Angular use for XLIFF 2 translation files?

| Strategy | Chunks | Raw/rendered | Ranks | Fragmentation | Hit@1 / evidence coverage | F1 |
|---|---|---|---|---|---|---|
| Fixed-size | 5 | 1892/1919 | 1,2,3,4,5 | 1.00 | 0 / 0.00 | 0.2143 |
| Structure-aware | 15 | 2036/2113 | 1,2,3,4,5,6,7,8,9,10,11,12,20,22,49 | 1.00 | 0 / 0.00 | 0.7200 |
| Prompt-based | 10 | 2043/2089 | 1,2,3,4,5,6,8,12,18,40 | 1.00 | 0 / 0.00 | 0.2029 |

### `q_easy_047` — prompt-based clear win

easy; fact; 1 evidence item(s). Which Angular directive can dynamically render a template fragment at an ng-container location?

| Strategy | Chunks | Raw/rendered | Ranks | Fragmentation | Hit@1 / evidence coverage | F1 |
|---|---|---|---|---|---|---|
| Fixed-size | 6 | 2028/2060 | 1,2,3,4,18,21 | 1.00 | 0 / 1.00 | 0.1538 |
| Structure-aware | 16 | 2047/2127 | 1,2,3,4,5,6,7,8,9,10,11,12,13,15,17,18 | 1.00 | 1 / 1.00 | 0.1935 |
| Prompt-based | 9 | 2047/2087 | 1,2,3,4,5,6,11,14,15 | 1.00 | 0 / 1.00 | 0.5957 |

### `q_easy_038` — similar answers

easy; fact; 1 evidence item(s). What object does @ViewChildren create to hold query results?

| Strategy | Chunks | Raw/rendered | Ranks | Fragmentation | Hit@1 / evidence coverage | F1 |
|---|---|---|---|---|---|---|
| Fixed-size | 6 | 2007/2038 | 1,2,3,4,19,33 | 1.00 | 1 / 1.00 | 0.6923 |
| Structure-aware | 11 | 2041/2097 | 1,2,3,4,5,6,7,8,9,11,17 | 1.00 | 1 / 1.00 | 0.6923 |
| Prompt-based | 8 | 2047/2082 | 1,2,3,4,5,6,7,9 | 1.00 | 1 / 1.00 | 0.6923 |

### `q_medium_027` — high evidence fragmentation

medium; synthesis; 2 evidence item(s). How does ng-container support dynamic rendering without adding an extra DOM element?

| Strategy | Chunks | Raw/rendered | Ranks | Fragmentation | Hit@1 / evidence coverage | F1 |
|---|---|---|---|---|---|---|
| Fixed-size | 5 | 1998/2026 | 1,2,3,4,16 | 1.00 | 1 / 1.00 | 0.4144 |
| Structure-aware | 12 | 2043/2105 | 1,2,3,4,5,6,7,8,9,10,11,12 | 1.50 | 1 / 1.00 | 0.3506 |
| Prompt-based | 10 | 2033/2079 | 1,2,3,4,5,6,7,9,18,22 | 1.00 | 1 / 1.00 | 0.3776 |

### `q_easy_051` — high formatting overhead

easy; fact; 1 evidence item(s). Which extract-i18n option sets the translation output file format?

| Strategy | Chunks | Raw/rendered | Ranks | Fragmentation | Hit@1 / evidence coverage | F1 |
|---|---|---|---|---|---|---|
| Fixed-size | 4 | 2048/2069 | 1,2,3,4 | 1.00 | 1 / 1.00 | 0.7586 |
| Structure-aware | 26 | 2044/2184 | 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,29,33,36,38 | 1.00 | 1 / 1.00 | 0.7586 |
| Prompt-based | 13 | 2039/2100 | 1,2,3,4,5,6,7,9,10,13,23,24,29 | 1.00 | 1 / 1.00 | 0.6316 |

### `q_hard_006` — hard multi-evidence

hard; sequence; 2 evidence item(s). If a nested child @defer block triggers hydration while its parent is still dehydrated, in what order does Angular hydrate the hierarchy and why?

| Strategy | Chunks | Raw/rendered | Ranks | Fragmentation | Hit@1 / evidence coverage | F1 |
|---|---|---|---|---|---|---|
| Fixed-size | 4 | 2048/2070 | 1,2,3,4 | 1.00 | 1 / 1.00 | 0.7234 |
| Structure-aware | 16 | 2047/2129 | 1,2,3,4,5,6,7,8,9,10,11,12,13,14,32,47 | 1.00 | 1 / 1.00 | 0.6667 |
| Prompt-based | 9 | 2013/2050 | 1,2,3,4,5,6,7,8,15 | 1.00 | 1 / 1.00 | 0.4722 |

## N. Strategy Profiles

### Fixed-size

Strengths: best observed overall/easy/hard Token F1, Hit@1, and MRR; simple deterministic implementation. Weaknesses: exact 11.22% corpus token expansion from overlap, coarse chunks, and least efficient raw-budget packing. Downstream: highest observed answer quality, but not statistically significant.

### Structure-aware

Strengths: smallest/most granular chunks, tightest budget packing, best deep and evidence retrieval. Weaknesses: most chunks per prompt, deepest rank scanning, and largest label/separator overhead. Downstream: retrieval advantages did not become the highest overall Token F1.

### Prompt-based

Strengths: least intrinsic evidence fragmentation, intermediate granularity/overhead, and highest observed medium-difficulty F1. Weaknesses: no overall retrieval or answer-quality lead and greater creation complexity. Downstream: its overall answer F1 is lowest, without a significant pairwise difference.

## O. Answers to Q1–Q13

1. **Structure-aware** creates the smallest and most granular chunks.
2. **Fixed-size** has the most exactly demonstrated redundancy: 58,623 overlap occurrences, 11.22% overhead.
3. **Prompt-based** preserves gold evidence with the least measured fragmentation, narrowly ahead of fixed-size and structure-aware.
4. **Structure-aware** fits the 2,048-token raw budget most efficiently.
5. **Structure-aware** incurs the most formatting overhead because it selects many more small labeled chunks.
6. **Mixed:** fixed-size is best at rank one/MRR; structure-aware is best for deep and exact evidence retrieval.
7. **Fixed-size** has the highest observed mean Token F1.
8. **No.** Existing Holm-adjusted paired tests find no significant answer-quality differences.
9. **Partly.** Structure-aware's fine granularity aligns with better evidence retrieval, while prompt-based's lowest measured fragmentation does not yield the best retrieval; neither advantage yields the highest overall answer F1.
10. **Partly.** All-evidence retrieval is associated with higher F1 in the canonical report, but the strategy with best evidence retrieval did not have best overall F1.
11. **Descriptively:** fixed-size leads easy/hard; prompt-based leads medium. Small question-type strata are inconclusive.
12. **Fixed:** rank-one retrieval and observed answer F1. **Structure:** granularity, evidence retrieval, budget packing. **Prompt:** least evidence fragmentation, intermediate granularity, and observed medium F1.
13. **Fixed-size is the strongest practical default for this benchmark** given equal-budget observed answer quality, simplicity, and lack of significant evidence that the more complex alternatives improve final answers. This conclusion is limited to this Angular benchmark.

## P. Limitations

- Evidence density is unavailable on a comparable exact-token basis.
- Zero-overlap policy is not the same as an independently measured zero-duplication corpus result.
- Four fixed-size cache hits lack reasoning-token diagnostics.
- Correlations and case studies are exploratory, not causal.
- One corpus, embedding model, generation model, and budget are evaluated.
- Token F1 rewards lexical overlap and may undervalue valid paraphrases.

## Q. Final Conclusion

The data demonstrates a real trade-off: prompt-based gives the least measured evidence fragmentation; structure-aware gives the strongest evidence retrieval and most efficient raw-budget packing; fixed-size gives the highest observed downstream answer F1 and strongest early-rank retrieval; prompt-based otherwise has intermediate granularity/overhead and leads the medium-difficulty descriptive slice. None of the overall answer differences is statistically significant.

A plausible explanation is that semantic granularity helps retrieve distributed evidence while coarse fixed chunks preserve broader local context, but the frozen artifacts do not establish that causal mechanism.
