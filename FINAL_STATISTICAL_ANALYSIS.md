# Final Statistical Analysis — Canonical Benchmark V2

## A. Analysis identity

- Status: frozen-input verification **PASS** across 34 authoritative artifacts.
- Statistical-analysis fingerprint: `711a4967c4570109a8cf504ea7dc761a0636644f7321a5fbc11b939f616403a1`.
- Source freeze fingerprint: `9c4a0242e17dadb92b7f74479114543aa1f28dd73c63cef04a045b2eef7c7c15`.
- Dataset / generation / evaluation: `9866799ea8f87c6a7c118cbaf0d8c298524757bd753db5d640e1a32234206d74` / `c4f4768ec9b80361dfd0a1e252f74ff348aa4e4c953bcca02761ba345f38b301` / `c6867bffbd9775d3ef9b4ce666ae09f1995a6ccb7a7ef14858bbcdb736c1fa55`.
- Retrieval / answer benchmark: `9dab4015ca1ae4c4abda04ccc5809a5e030c793fcc7feff932d04ecfb116a6b7` / `375983ff4b3c4e84b303d7298c4dd93b44782430bbc1fd6dff41db6f3b60af23`.
- Created at: `2026-08-12T20:11:21.553787Z`. Paired sample: **n=140**.

## B. Methods

The primary endpoint is raw per-question Token F1 in a paired design. The canonical uncertainty analysis uses 50,000 question-level percentile-bootstrap resamples with seed 2026; every replicate applies one shared index sample to all strategies. The primary hypothesis-test family is a two-sided 100,000-draw paired sign-flip randomization test of mean delta with add-one Monte Carlo p-values. SciPy Wilcoxon signed-rank tests are prespecified robustness checks (`zero_method=wilcox`, exact zeros discarded, asymptotic two-sided method, no continuity correction). Holm-Bonferroni correction is applied separately within each three-comparison family at alpha 0.05. Effect sizes are absolute mean delta, paired Cohen's dz, and matched rank-biserial correlation.

Normality diagnostics are descriptive only. No test was selected after inspecting its p-value, independent-sample inference is not used, and no practical-equivalence margin was preregistered; therefore no formal equivalence test is performed.

## C. Overall descriptive results

| Strategy | n | Mean | Median | SD | Q1 | Q3 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_size | 140 | 0.382956 | 0.367544 | 0.160110 | 0.276181 | 0.464721 | 0.045113 | 1.000000 |
| structure_aware | 140 | 0.376574 | 0.358059 | 0.159171 | 0.255844 | 0.456555 | 0.116788 | 1.000000 |
| prompt_based | 140 | 0.372108 | 0.355981 | 0.146232 | 0.263203 | 0.473684 | 0.052045 | 0.750000 |

## D. Bootstrap confidence intervals

Canonical method: paired question-level percentile bootstrap.

| Strategy | Observed mean | Bootstrap mean | Bootstrap SE | 95% CI |
|---|---:|---:|---:|---:|
| fixed_size | 0.382956 | 0.382841 | 0.013466 | [0.356980, 0.409676] |
| structure_aware | 0.376574 | 0.376490 | 0.013401 | [0.350720, 0.403212] |
| prompt_based | 0.372108 | 0.372086 | 0.012341 | [0.348056, 0.396464] |

## E. Primary paired comparisons

| Pair | Mean delta | Paired 95% CI | Bootstrap P(delta>0) | Permutation raw p | Permutation Holm p | Cohen's dz | Rank-biserial | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fixed_size vs structure_aware | +0.006382 | [-0.011355, +0.023517] | 0.763320 | 0.479245 | 0.95849 | 0.060360 | 0.123030 | No statistically reliable difference detected |
| fixed_size vs prompt_based | +0.010847 | [-0.008294, +0.030061] | 0.864920 | 0.275327 | 0.825982 | 0.093417 | 0.118375 | No statistically reliable difference detected |
| structure_aware vs prompt_based | +0.004465 | [-0.015878, +0.026161] | 0.652420 | 0.679813 | 0.95849 | 0.035119 | -0.065905 | No statistically reliable difference detected |

Bootstrap probabilities are descriptive resampling frequencies, not Bayesian posterior probabilities. Confidence intervals and tests use paired raw scores, not rounded aggregates.

## F. Multiple-comparison correction and robustness

| Pair | Wilcoxon statistic | Wilcoxon raw p | Wilcoxon Holm p | Significant? | Zeros discarded |
|---|---:|---:|---:|---|---:|
| fixed_size vs structure_aware | 4145.000000 | 0.211532 | 0.634595 | False | 3 |
| fixed_size vs prompt_based | 4167.000000 | 0.229326 | 0.634595 | False | 3 |
| structure_aware vs prompt_based | 4415.000000 | 0.503327 | 0.634595 | False | 3 |

Primary conclusions use the corrected permutation family; Wilcoxon is a prespecified robustness analysis and is not substituted opportunistically.

## G. Bootstrap ranking stability

Across 50000 canonical bootstrap replicates:

- `fixed_size` sole-winner frequency: 0.687240.
- `structure_aware` sole-winner frequency: 0.209220.
- `prompt_based` sole-winner frequency: 0.103540.
- Exact tie frequency: 0.000000.

These are descriptive bootstrap frequencies, not posterior model probabilities.

## H. Sensitivity analysis

The 3-seed × 2-resample-count grid (42, 2026, 314159 × 10,000, 50,000) was numerically stable:

- `fixed_size_vs_structure_aware`: CI lower range [-0.011585, -0.011298], upper range [+0.023256, +0.023579], P(delta>0) range [0.757200, 0.764800], zero-inclusion decision stable: `True`.
- `fixed_size_vs_prompt_based`: CI lower range [-0.008784, -0.008073], upper range [+0.029851, +0.030631], P(delta>0) range [0.861500, 0.869700], zero-inclusion decision stable: `True`.
- `structure_aware_vs_prompt_based`: CI lower range [-0.016427, -0.015220], upper range [+0.025977, +0.026286], P(delta>0) range [0.652420, 0.655600], zero-inclusion decision stable: `True`.

## I. Difficulty analysis

Difficulty results are secondary/exploratory and cannot override the primary n=140 endpoint.

| Difficulty | n | fixed_size | structure_aware | prompt_based |
|---|---:|---:|---:|---:|
| easy | 60 | 0.389230 | 0.383112 | 0.372292 |
| medium | 40 | 0.386294 | 0.386237 | 0.393857 |
| hard | 40 | 0.370205 | 0.357103 | 0.350084 |

- `easy`, fixed_size vs structure_aware: delta +0.006118, wins/ties/losses 29/3/28, exploratory 95% CI [-0.030271, +0.041026].
- `easy`, fixed_size vs prompt_based: delta +0.016938, wins/ties/losses 31/3/26, exploratory 95% CI [-0.021315, +0.055853].
- `easy`, structure_aware vs prompt_based: delta +0.010820, wins/ties/losses 24/2/34, exploratory 95% CI [-0.031160, +0.055650].
- `medium`, fixed_size vs structure_aware: delta +0.000057, wins/ties/losses 19/0/21, exploratory 95% CI [-0.021106, +0.021808].
- `medium`, fixed_size vs prompt_based: delta -0.007563, wins/ties/losses 22/0/18, exploratory 95% CI [-0.030097, +0.014431].
- `medium`, structure_aware vs prompt_based: delta -0.007620, wins/ties/losses 15/1/24, exploratory 95% CI [-0.030776, +0.016850].
- `hard`, fixed_size vs structure_aware: delta +0.013102, wins/ties/losses 27/0/13, exploratory 95% CI [-0.007522, +0.032758].
- `hard`, fixed_size vs prompt_based: delta +0.020120, wins/ties/losses 28/0/12, exploratory 95% CI [-0.003635, +0.043749].
- `hard`, structure_aware vs prompt_based: delta +0.007019, wins/ties/losses 21/0/19, exploratory 95% CI [-0.014733, +0.029275].

## J. Question-type analysis

All question-type comparisons are exploratory. Every stratum with n<10 is explicitly descriptive only.

| Type | n | fixed_size | structure_aware | prompt_based | Observed winner | Scope |
|---|---:|---:|---:|---:|---|---|
| behavior | 22 | 0.382344 | 0.376969 | 0.368172 | fixed_size | EXPLORATORY DESCRIPTIVE |
| cause_effect | 2 | 0.299299 | 0.297579 | 0.308530 | prompt_based | LOW SAMPLE SIZE — DESCRIPTIVE ONLY |
| comparison | 26 | 0.333263 | 0.349324 | 0.339993 | structure_aware | EXPLORATORY DESCRIPTIVE |
| definition | 15 | 0.315221 | 0.308532 | 0.255816 | fixed_size | EXPLORATORY DESCRIPTIVE |
| fact | 21 | 0.483624 | 0.456358 | 0.510267 | prompt_based | EXPLORATORY DESCRIPTIVE |
| list | 4 | 0.640193 | 0.610082 | 0.481238 | fixed_size | LOW SAMPLE SIZE — DESCRIPTIVE ONLY |
| mechanism | 7 | 0.388201 | 0.376537 | 0.354711 | fixed_size | LOW SAMPLE SIZE — DESCRIPTIVE ONLY |
| procedure | 11 | 0.367245 | 0.375633 | 0.384245 | prompt_based | EXPLORATORY DESCRIPTIVE |
| security_mechanism | 1 | 0.464945 | 0.452055 | 0.496063 | prompt_based | LOW SAMPLE SIZE — DESCRIPTIVE ONLY |
| sequence | 3 | 0.408040 | 0.360796 | 0.292923 | fixed_size | LOW SAMPLE SIZE — DESCRIPTIVE ONLY |
| syntax | 1 | 0.300000 | 0.307692 | 0.320000 | prompt_based | LOW SAMPLE SIZE — DESCRIPTIVE ONLY |
| synthesis | 2 | 0.345997 | 0.303765 | 0.320154 | fixed_size | LOW SAMPLE SIZE — DESCRIPTIVE ONLY |
| tradeoff | 10 | 0.375812 | 0.383934 | 0.388783 | prompt_based | EXPLORATORY DESCRIPTIVE |
| why | 15 | 0.353151 | 0.336059 | 0.341872 | fixed_size | EXPLORATORY DESCRIPTIVE |

## K. Retrieval-answer relationship

These within-strategy associations are exploratory and non-causal.

| Strategy | Hit@1 F1 difference | Hit@5 F1 difference | All-evidence F1 difference | Coverage Spearman rho |
|---|---:|---:|---:|---:|
| fixed_size | +0.063378 | +0.096626 | +0.059348 | +0.161681 |
| structure_aware | +0.028214 | -0.042641 | +0.040299 | +0.123075 |
| prompt_based | -0.008489 | +0.049213 | +0.031546 | +0.134639 |

Better retrieval/evidence outcomes can coincide with higher answer overlap without establishing causality. Strategy-level evidence coverage and answer F1 need not rank identically because retrieval coverage is only one input to generation; answer wording, context composition, irrelevant material, and lexical metric sensitivity also matter.

## L. Practical significance

- `fixed_size vs structure_aware`: absolute delta +0.006382 Token F1; relative to the right mean +1.695%; paired dz +0.060360. The absolute difference is primary; the percentage is supplementary.
- `fixed_size vs prompt_based`: absolute delta +0.010847 Token F1; relative to the right mean +2.915%; paired dz +0.093417. The absolute difference is primary; the percentage is supplementary.
- `structure_aware vs prompt_based`: absolute delta +0.004465 Token F1; relative to the right mean +1.200%; paired dz +0.035119. The absolute difference is primary; the percentage is supplementary.

Statistical detection does not by itself establish a meaningful operational gain. Because no smallest effect size of interest was preregistered, this report does not claim formal practical equivalence or assign universal small/medium/large labels.

## M. Limitations

- The paired sample contains 140 questions from one Angular benchmark domain.
- Results are specific to the frozen generation model, retrieval/index configuration, context budget, and evaluation implementation.
- Token F1 is deterministic and auditable but can undervalue semantically correct paraphrases and reward lexical overlap.
- Question-type strata include groups with n=1–4 and cannot support inferential claims.
- No practical-equivalence margin was preregistered.
- Bootstrap and randomization inference characterize this benchmark sample; they do not establish universal chunking-strategy superiority.

## N. Final statistical conclusion

`fixed_size` achieved the highest observed mean Token F1, but its paired differences were not statistically distinguishable from zero after Holm correction. The observed ordering should be treated as inconclusive.

Secondary precision/recall and retrieval analyses provide context only and do not replace the primary Token F1 inference. No benchmark redesign, tuning, or new experimental run was performed.

## Test results

Focused statistical correctness tests: 9 passed in 1.37s. Relevant freeze/evaluation/statistical tests: 60 passed in 1.83s. Full repository suite: 342 passed in 3.91s (333-test baseline plus 9 focused statistical tests); zero regressions.

**FINAL STATISTICAL VALIDATION: COMPLETE**
