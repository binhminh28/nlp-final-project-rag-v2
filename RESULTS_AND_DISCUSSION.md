# Results and Discussion

## Results

### Experimental comparability

The frozen canonical v2 benchmark compared three chunking strategies—fixed-size, structure-aware, and prompt-based—on the same 384-document Angular corpus and the same 140-question QA dataset. The comparison comprised 420 paired answer evaluations. All strategies used the same `openai/text-embedding-3-small` embedding configuration, dense retrieval mechanism, candidate depth of 50, deterministic context construction, `openai/gpt-5-mini` answer-generation configuration, and deterministic answer evaluation. The chunking strategy was therefore the principal experimental variable in the downstream comparison.

The canonical generation inputs were selected under a common budget of 2,048 raw chunk tokens rather than a common fixed top-k. This distinction is important: because chunk sizes differ substantially, strategies can contribute different numbers of chunks while consuming approximately the same raw context allowance. Contexts were subsequently rendered using the same `[CONTEXT n]` labels and separators. The prompt-based artifact evaluated here is the historical `prompt_based_v1` artifact recorded in the frozen chunk manifest, not the repository's current `prompt_based_v2` default.

### Chunk characteristics and corpus redundancy

Chunking changed the corpus representation substantially (Table 1). Fixed-size produced the fewest and largest chunks: 1,300 chunks with a mean length of 447.1 tokens and a median of 512. Its p90, p95, p99, and maximum were all 512 tokens, demonstrating strong concentration at the configured ceiling. Structure-aware produced 3,162 chunks and was the most granular strategy, with a mean of 165.1 and median of 129 tokens. The evaluated prompt-based artifact lay between these approaches, producing 2,054 chunks with a mean of 255.1 and median of 237 tokens. These measurements characterize granularity; they do not imply that smaller chunks are intrinsically better.

**Table 1. Chunk characteristics and intrinsic evidence preservation.** Token totals are chunk-token occurrences. Exact redundancy is reported only where independently measurable.

| Strategy | Chunks | Mean tokens | Median tokens | p95 | Total token occurrences | Redundancy or overlap | Evidence items covered by one chunk |
|---|---:|---:|---:|---:|---:|---|---:|
| Fixed-size | 1,300 | 447.1 | 512 | 512 | 581,181 | 58,623 exact overlap occurrences; 11.22% overhead | 96.6% |
| Structure-aware | 3,162 | 165.1 | 129 | 447 | 522,165 | No-overlap policy; exact corpus duplication not independently measured | 99.1% |
| Prompt-based v1 artifact | 2,054 | 255.1 | 237 | 490 | 524,005 | Zero-overlap/source-slice policy; exact corpus duplication not independently measured | 99.6% |

For fixed-size, the processed source contained 522,558 tokens, whereas the chunks contained 581,181 token occurrences. The difference was 58,623 positional-overlap occurrences, corresponding to an expansion ratio of 1.1122 and 11.22% overhead. Structure-aware explicitly records a no-overlap policy, and the prompt-based artifact records zero chunk overlap and exact normalized source slices. However, the frozen analysis did not independently measure corpus-wide duplication for these strategies using a common global alignment. Consequently, no comparable redundancy percentage is assigned to them.

### Evidence preservation and fragmentation

All 233 gold evidence items were completely represented somewhere in the canonical chunk population for every strategy. Differences appeared only in whether a single chunk could contain all sentence-level source spans belonging to an evidence item. Prompt-based had the lowest measured fragmentation: 99.6% of evidence items required exactly one chunk, and its mean minimum was 1.004 chunks per item. Structure-aware followed at 99.1% and 1.009 chunks, while fixed-size achieved 96.6% and 1.034 chunks. The maximum was two chunks for all strategies, and the median and p95 were one.

These values were close to ceiling. Prompt-based ranked first on this measure, but the absolute differences between prompt-based and structure-aware were small. Fragmentation was defined as the minimum number of distinct canonical chunks whose exact reconstructed source spans covered every sentence unit in an evidence item; overlapping chunks that independently contained the same complete evidence counted as one, not as additional fragmentation. Evidence density was not compared because exact token-level evidence intersections and a common BPE-boundary attribution policy were not persisted consistently across strategies.

### Context construction and token-budget utilization

The common token budget produced markedly different contexts (Table 2). Fixed-size selected 4.94 chunks per query on average and used 2,016.6 raw tokens, or 98.47% of the budget. Structure-aware selected 14.64 chunks—almost three times as many—and used 2,039.3 raw tokens, corresponding to 99.57% utilization. Prompt-based selected 8.62 chunks and used 2,027.0 raw tokens, or 98.97% utilization. Mean unused capacity was therefore 31.4, 8.7, and 21.0 tokens, respectively. Structure-aware used at least 99% of the budget for 86.4% of queries, compared with 52.9% for fixed-size and 57.9% for prompt-based.

**Table 2. Canonical context utilization.** Values are means per query unless stated otherwise.

| Strategy | Selected chunks | Raw tokens | Budget utilization | Unused tokens | Rendered tokens | Formatting overhead | Overhead ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed-size | 4.94 | 2,016.6 | 98.47% | 31.4 | 2,043.7 | 27.1 | 1.35% |
| Structure-aware | 14.64 | 2,039.3 | 99.57% | 8.7 | 2,114.8 | 75.5 | 3.70% |
| Prompt-based v1 artifact | 8.62 | 2,027.0 | 98.97% | 21.0 | 2,064.8 | 37.8 | 1.86% |

The finer structure-aware units therefore packed the raw-token budget most completely, but they also required the most context blocks. After identical ordinal labels and separators were inserted, structure-aware incurred 75.5 formatting tokens per query on average, compared with 37.8 for prompt-based and 27.1 for fixed-size. Its raw-token packing advantage was thus partially offset by higher rendered-context overhead.

The strategies also reached different depths in the candidate ranking. Mean maximum selected rank was 13.47 for fixed-size, 29.37 for structure-aware, and 18.71 for prompt-based. Of the selected chunks, 9.2%, 31.4%, and 13.6%, respectively, came from ranks above 10. This result reflects the whole-chunk packing policy: smaller chunks allowed later candidates to fit after larger intervening candidates were skipped. Deeper rank use is a retrieval behavior, not inherently a quality improvement.

### Retrieval effectiveness

Retrieval results did not identify a single winner across all criteria (Table 3). Fixed-size achieved the highest Hit@1 (0.8071) and MRR (0.8732), indicating the strongest observed early-rank placement. Structure-aware and prompt-based both exceeded fixed-size at Hit@5 (0.9786 versus 0.9643). Structure-aware achieved the highest Recall@10 (0.9857), mean exact evidence coverage (0.7476), and all-evidence-retrieved rate (63.6%). Prompt-based was second on Recall@10 (0.9786) and evidence coverage (0.7320), while fixed-size recorded 0.9429 and 0.6905.

**Table 3. Canonical retrieval results under the 2,048-token selection protocol.**

| Strategy | Hit@1 | Hit@5 | MRR | Recall@10 | Evidence coverage | All evidence retrieved |
|---|---:|---:|---:|---:|---:|---:|
| Fixed-size | **0.8071** | 0.9643 | **0.8732** | 0.9429 | 0.6905 | 57.1% |
| Structure-aware | 0.7786 | **0.9786** | 0.8692 | **0.9857** | **0.7476** | **63.6%** |
| Prompt-based v1 artifact | 0.7571 | **0.9786** | 0.8508 | 0.9786 | 0.7320 | 61.4% |

These metrics describe different aspects of retrieval. Hit@1 and MRR emphasize early relevant results, whereas Recall@10 and evidence coverage assess recovery deeper in the selected candidate set. Retrieval was treated as a diagnostic layer; final answer quality remained the primary downstream endpoint.

### Final answer quality and statistical uncertainty

Fixed-size achieved the highest observed mean Token F1 (0.3830), followed by structure-aware (0.3766) and prompt-based (0.3721). The corresponding paired-bootstrap 95% confidence intervals for the strategy means were [0.3570, 0.4097], [0.3507, 0.4032], and [0.3481, 0.3965] (Table 4).

**Table 4. Overall answer quality.** Confidence intervals use 50,000 question-level paired bootstrap resamples with seed 2026.

| Strategy | Mean Token F1 | Median | Bootstrap 95% CI |
|---|---:|---:|---:|
| Fixed-size | **0.3830** | 0.3675 | [0.3570, 0.4097] |
| Structure-aware | 0.3766 | 0.3581 | [0.3507, 0.4032] |
| Prompt-based v1 artifact | 0.3721 | 0.3560 | [0.3481, 0.3965] |

The prespecified paired tests did not detect a statistically reliable difference for any pair (Table 5). Fixed-size minus structure-aware had a mean delta of 0.0064 with paired-bootstrap 95% CI [−0.0114, 0.0235] and Holm-adjusted permutation p = 0.9585. Fixed-size minus prompt-based had a mean delta of 0.0108, CI [−0.0083, 0.0301], and adjusted p = 0.8260. Structure-aware minus prompt-based had a mean delta of 0.0045, CI [−0.0159, 0.0262], and adjusted p = 0.9585. Every interval included zero.

**Table 5. Paired Token F1 comparisons.** The primary tests were two-sided Monte Carlo sign-flip tests with 100,000 resamples and Holm–Bonferroni correction across the three comparisons.

| Pair (left − right) | Mean delta | Paired 95% CI | Holm-adjusted p | Decision |
|---|---:|---:|---:|---|
| Fixed-size − structure-aware | 0.0064 | [−0.0114, 0.0235] | 0.9585 | No statistically reliable difference detected |
| Fixed-size − prompt-based | 0.0108 | [−0.0083, 0.0301] | 0.8260 | No statistically reliable difference detected |
| Structure-aware − prompt-based | 0.0045 | [−0.0159, 0.0262] | 0.9585 | No statistically reliable difference detected |

Wilcoxon signed-rank robustness tests supported the same conclusion: all Holm-adjusted p-values were 0.6346. In descriptive bootstrap ranking stability, fixed-size had the highest resampled mean in 68.7% of resamples, structure-aware in 20.9%, and prompt-based in 10.4%. These frequencies describe the stability of the observed ranking under bootstrap resampling; they are not probabilities that a strategy is truly best. The inferential conclusion remains that no reliable overall difference was detected.

### Difficulty and question-type results

Difficulty analyses were secondary and exploratory. The easy group contained 60 questions, and fixed-size had the highest observed mean F1 (0.3892), followed by structure-aware (0.3831) and prompt-based (0.3723). The medium group contained 40 questions; prompt-based ranked first descriptively (0.3939), with fixed-size (0.3863) and structure-aware (0.3862) nearly tied. Among 40 hard questions, fixed-size ranked first (0.3702), followed by structure-aware (0.3571) and prompt-based (0.3501). These subgroup rankings were not established as statistically reliable and do not override the overall paired endpoint.

Among larger question-type strata, observed rankings varied. Fixed-size led behavior (n = 22), definition (n = 15), list (n = 4, descriptive only), sequence (n = 3, descriptive only), and why questions (n = 15); structure-aware led comparison (n = 26); prompt-based led fact (n = 21), procedure (n = 11), and trade-off questions (n = 10). Categories with very small samples, including n = 1–4 strata, are descriptive only and should not support general claims.

### Relationships across pipeline layers

Upstream chunk properties did not translate monotonically into downstream answer quality. Prompt-based preserved the largest proportion of evidence items within a single chunk (99.6%) but did not achieve the highest overall retrieval or answer score. Structure-aware packed the token budget most tightly and achieved the highest Recall@10 and evidence coverage, yet its mean Token F1 was below fixed-size. Conversely, fixed-size incurred measurable overlap redundancy and recovered less exact evidence overall, while achieving the highest Hit@1, MRR, and observed mean Token F1.

Exploratory Spearman correlations between Token F1 and context characteristics were generally weak. Across strategies, correlations with selected chunk count ranged from −0.073 to 0.065, with raw selected tokens from −0.060 to −0.047, with rendered tokens from −0.077 to 0.061, and with maximum selected rank from 0.016 to 0.102. Retrieval evidence coverage showed small positive associations with Token F1 (ρ = 0.162 fixed-size, 0.123 structure-aware, and 0.135 prompt-based; n = 140 each). These associations are non-causal. Consistent with the canonical retrieval–answer analysis, answers had higher descriptive F1 when all evidence was retrieved, but the strategy with the strongest aggregate evidence retrieval did not produce the highest aggregate answer F1.

### Token-efficiency results

Three token quantities must be distinguished. Raw selected chunk tokens count only the canonical selected chunk texts and were controlled by the 2,048-token protocol. Rendered context tokens additionally include `[CONTEXT n]` labels and separators. Provider input tokens further include system instructions, the question, prompt formatting, chat framing, and provider-specific accounting.

Mean provider-reported input was 2,118.5 tokens for fixed-size, 2,199.7 for structure-aware, and 2,142.3 for prompt-based. Mean completion usage was 236.8, 238.2, and 242.4 tokens, respectively. Descriptive Token F1 per 1,000 raw tokens was 0.1900, 0.1847, and 0.1836; per 1,000 rendered context tokens it was 0.1875, 0.1780, and 0.1803. Because the experiment intentionally made the raw-token denominators similar, these ratios largely track Token F1 and are secondary efficiency descriptions rather than independent primary outcomes. Provider reasoning-token diagnostics covered 136/140 fixed-size calls and all 140 calls for each other strategy; reasoning usage is part of completion usage and was not double-counted.

## Discussion

### Fixed-size: strong downstream simplicity with measurable redundancy

Fixed-size presents a favorable practical trade-off in this benchmark. It achieved the highest observed overall Token F1, Hit@1, and MRR, while requiring only 4.94 context blocks per query and incurring the lowest formatting overhead. It is deterministic, simple to implement, and does not require a model-assisted chunk-planning stage.

These strengths coexist with clear limitations. Fixed-size generated substantially larger chunks, concentrated heavily at 512 tokens, and introduced 58,623 duplicate positional token occurrences. It also left more of the raw selection budget unused than the other strategies. Larger chunks may contain more locally coherent surrounding information, but they may also include unrelated content; this experiment did not isolate either mechanism. Most importantly, the observed downstream advantage was small and statistically uncertain. The results support describing fixed-size as the highest-scoring observed strategy, not as conclusively better.

### Structure-aware: retrieval granularity at a context-formatting cost

Structure-aware created the finest-grained representation and used the raw-token budget most completely. Its smaller units allowed the selection policy to include more discrete chunks and reach farther into the ranked candidate list. This behavior was accompanied by the strongest Recall@10, exact evidence coverage, and all-evidence retrieval.

The same granularity increased the number of context blocks nearly threefold relative to fixed-size and raised mean formatting overhead to 75.5 tokens. Its mean rendered context was therefore the largest despite the common raw-token budget. Fine-grained retrieval may improve evidence access while simultaneously fragmenting the generation input across more labeled units or admitting lower-ranked distractors. Those are plausible mediators, not experimentally isolated causes. The measured result is that structure-aware's upstream retrieval advantages did not yield an overall answer-quality advantage.

### Prompt-based v1: evidence preservation without downstream improvement

The evaluated prompt-based v1 artifact provided the strongest intrinsic evidence-containment result: 99.6% of evidence items were completely covered by one chunk. Its chunk granularity, budget packing, number of selected blocks, and formatting overhead generally fell between fixed-size and structure-aware. It also achieved the highest observed mean Token F1 on medium-difficulty questions.

This evidence-preservation advantage was numerically close to ceiling and did not translate into the highest overall retrieval or answer score. Semantic or block-boundary planning may help preserve evidence units, but the experiment did not isolate planning as the cause. Prompt-based construction also introduces qualitative engineering costs: a planner dependency, additional latency and potential external cost during chunk creation, and greater reproducibility and version-management complexity. The canonical results specifically describe the historical `prompt_based_v1` artifact. They must not be generalized to the current `prompt_based_v2` implementation without regenerating and reevaluating that artifact.

### Why better chunks did not necessarily produce better answers

The results demonstrate that chunk quality is multidimensional. Evidence containment, semantic boundaries, retrieval granularity, token packing, early-rank placement, and final answer overlap measure different properties. Improving one dimension does not guarantee improvement in another. Here, prompt-based ranked first in evidence containment, structure-aware ranked first in deep evidence retrieval and budget packing, and fixed-size ranked first in early retrieval and observed answer F1.

Several mechanisms may mediate these relationships. Chunk boundaries can interact with embedding similarity and rank ordering; selecting many small chunks can alter distractor density and context continuity; fixed overlaps can repeat information; and the generator may be robust to missing or fragmented evidence in some cases but sensitive in others. The lexical Token F1 metric can also reward answers that use reference wording and undervalue semantically correct paraphrases. None of these mechanisms was manipulated independently, so they are possible explanations rather than causal conclusions.

### Token efficiency and context design

Structure-aware's 99.57% raw-budget utilization shows that smaller pieces enable finer whole-chunk packing. However, every piece receives a context label, so its 14.64 selected chunks produced 75.5 formatting tokens on average. The resulting provider input was 81.2 tokens larger per query than fixed-size. This illustrates why raw selected tokens, rendered context, and provider input should not be treated as interchangeable efficiency measures.

Because the primary protocol already controlled raw context consumption, F1 per 1,000 raw tokens mainly reflects the same answer-quality ordering. Rendered-context efficiency additionally penalizes block formatting and therefore makes the structure-aware trade-off more visible. A different context serialization policy could change this overhead without changing chunking or retrieval, so it should not be interpreted solely as an intrinsic chunk-quality property.

### Practical implications

For this Angular benchmark and tested pipeline, fixed-size is the most defensible practical default. It combined the highest observed answer quality with strong early retrieval, low context-formatting overhead, and substantially lower chunk-construction complexity. The more sophisticated approaches exhibited genuine upstream advantages but did not demonstrate a statistically reliable improvement in the primary downstream endpoint.

This recommendation is conditional. Structure-aware may be preferable when fine-grained retrieval, exact evidence coverage, semantic section boundaries, or tight raw-budget packing are primary requirements. Prompt-based chunking may be attractive when preserving semantic evidence units is especially important and model-assisted construction complexity is acceptable. These conditions are suggested by the measured dimensions; they are not universal rankings. No composite score is used because the experiment does not provide a defensible weighting of answer quality, retrieval coverage, token use, redundancy, and engineering cost.

### Threats to validity

The benchmark covers one Angular technical-document corpus. Its conclusions do not automatically generalize to arbitrary Markdown, scientific literature, legal documents, web pages, or conversational data. Although 140 paired questions support an informative overall comparison, subgroup power is limited, and several question-type strata contain only one to four examples.

The pipeline used one embedding model, one dense retriever, one candidate depth, one 2,048-token raw selection budget, one context renderer, and one pinned answer model and configuration. Other embeddings, sparse or hybrid retrieval, rerankers, generators, or token budgets may interact differently with chunk granularity. Token F1 is deterministic and reproducible but lexical; it does not capture all aspects of semantic correctness. Provider behavior and prompt caching may vary, and four fixed-size responses lack fresh reasoning diagnostics because they were valid cache hits.

Exact fixed-size positional redundancy was measurable, but corresponding corpus-wide duplication percentages were not independently established for structure-aware and prompt-based. Evidence containment was near ceiling for every strategy, limiting its power to discriminate methods. Finally, the canonical prompt-based results apply to the historical `prompt_based_v1` artifact, while the current implementation defaults to `prompt_based_v2`. No conclusion about v2 follows from this benchmark.

## Answers to the Research Questions

**RQ1: How do the strategies differ in size and granularity?** Fixed-size produced the fewest and largest chunks, structure-aware produced the most and smallest, and prompt-based v1 occupied the middle.

**RQ2: Which strategy preserved evidence most effectively?** Prompt-based v1 had the lowest measured fragmentation, with 99.6% of evidence items contained completely in one chunk. All strategies were above 96%, so the advantage was modest in absolute terms.

**RQ3: Which strategy used the token budget most efficiently?** Structure-aware achieved the highest raw-token utilization (99.57%) and lowest mean unused capacity (8.7 tokens), while incurring the largest formatting overhead.

**RQ4: Which strategy performed best in retrieval?** No single strategy led every metric. Fixed-size led Hit@1 and MRR; structure-aware led Recall@10, evidence coverage, and all-evidence retrieval; structure-aware and prompt-based tied on Hit@5.

**RQ5: Which achieved the highest observed answer quality?** Fixed-size, with mean Token F1 0.3830.

**RQ6: Were answer-quality differences statistically reliable?** No. All paired-bootstrap intervals included zero, and all Holm-adjusted permutation and Wilcoxon tests were non-significant at α = 0.05.

**RQ7: Did improved evidence preservation or retrieval consistently translate into better answers?** No. Prompt-based led evidence containment and structure-aware led deep evidence retrieval, but neither achieved the highest overall Token F1.

**RQ8: What trade-off did each expose?** Fixed-size traded overlap redundancy and coarse packing for simplicity, early-rank strength, and the highest observed answer F1. Structure-aware traded many blocks and higher formatting overhead for granularity, tight packing, and evidence retrieval. Prompt-based traded planner complexity for strong evidence containment and intermediate granularity.

**RQ9: What is the most defensible practical default?** Fixed-size is the most defensible default for this specific benchmark because its simpler pipeline achieved the highest observed downstream quality and the complex alternatives did not demonstrate a reliable answer-quality gain. This is not a universal recommendation.

## Suggested figures

The existing derived data support four non-decorative figures without further experimentation: (1) chunk-size distributions from `chunk_distribution.csv`; (2) per-query raw-budget utilization and rendered overhead from `chunk_efficiency_per_query.jsonl`; (3) paired Token F1 distributions from the same per-query artifact; and (4) a retrieval-versus-answer panel contrasting Hit@1/evidence coverage with Token F1. These figures would visualize granularity, packing, answer uncertainty, and the disconnect between upstream and downstream rankings, respectively.

## Evidence provenance

All numerical claims were checked against the frozen canonical v2 lineage and the following authoritative sources: `freeze_manifest.json`, canonical chunk manifests and statistics, canonical retrieval `per_query.jsonl`, canonical prepared generation inputs, canonical answer evaluations, `FINAL_STATISTICAL_ANALYSIS.md`, statistical-analysis JSON artifacts, and `CHUNK_EFFICIENCY_ANALYSIS.md`. No canonical v1, smoke, synthetic, diagnostic, or historical retrieval run was included.
