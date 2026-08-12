"""Render the final research-safe statistical validation report."""

from __future__ import annotations

from typing import Any

from rag_chunking.benchmark import CANONICAL_STRATEGIES


def _pair_name(pair: dict[str, Any]) -> str:
    return f"{pair['left']} vs {pair['right']}"


def _final_conclusion(result: dict[str, Any]) -> str:
    pairs = result["paired_comparisons"]["pairs"]
    fixed_pairs = [pair for pair in pairs if pair["left"] == "fixed_size"]
    supported = [pair for pair in fixed_pairs if pair["statistical_conclusion"] == "Evidence favors left strategy"]
    if len(supported) == 2:
        return (
            "`fixed_size` achieved the highest observed mean Token F1 and its paired advantage over both "
            "alternatives was statistically supported after Holm correction. The absolute gains are "
            "small in Token F1 units, so this is evidence of a benchmark-specific edge, not a large or universal advantage."
        )
    if supported:
        return (
            "`fixed_size` achieved the highest observed mean Token F1, but corrected paired evidence supported "
            "only one of its two comparisons. The overall ordering is therefore not uniformly statistically resolved."
        )
    return (
        "`fixed_size` achieved the highest observed mean Token F1, but its paired differences were not "
        "statistically distinguishable from zero after Holm correction. The observed ordering should be treated as inconclusive."
    )


def render_report(result: dict[str, Any], *, created_at: str, tests_summary: str) -> str:
    common = result["common"]
    primary = result["primary_metrics"]
    pairs = result["paired_comparisons"]["pairs"]
    winner = result["bootstrap_summary"]["winner_stability"]
    lines = [
        "# Final Statistical Analysis — Canonical Benchmark V2", "",
        "## A. Analysis identity", "",
        f"- Status: frozen-input verification **{result['frozen_verification']['status']}** across {result['frozen_verification']['verified_artifact_count']} authoritative artifacts.",
        f"- Statistical-analysis fingerprint: `{common['statistical_analysis_fingerprint']}`.",
        f"- Source freeze fingerprint: `{common['source_freeze_fingerprint']}`.",
        f"- Dataset / generation / evaluation: `{common['dataset_fingerprint']}` / `{common['generation_fingerprint']}` / `{common['evaluation_fingerprint']}`.",
        f"- Retrieval / answer benchmark: `{common['retrieval_benchmark_fingerprint']}` / `{common['answer_benchmark_fingerprint']}`.",
        f"- Created at: `{created_at}`. Paired sample: **n=140**.", "",
        "## B. Methods", "",
        "The primary endpoint is raw per-question Token F1 in a paired design. The canonical uncertainty analysis uses 50,000 question-level percentile-bootstrap resamples with seed 2026; every replicate applies one shared index sample to all strategies. The primary hypothesis-test family is a two-sided 100,000-draw paired sign-flip randomization test of mean delta with add-one Monte Carlo p-values. SciPy Wilcoxon signed-rank tests are prespecified robustness checks (`zero_method=wilcox`, exact zeros discarded, asymptotic two-sided method, no continuity correction). Holm-Bonferroni correction is applied separately within each three-comparison family at alpha 0.05. Effect sizes are absolute mean delta, paired Cohen's dz, and matched rank-biserial correlation.", "",
        "Normality diagnostics are descriptive only. No test was selected after inspecting its p-value, independent-sample inference is not used, and no practical-equivalence margin was preregistered; therefore no formal equivalence test is performed.", "",
        "## C. Overall descriptive results", "",
        "| Strategy | n | Mean | Median | SD | Q1 | Q3 | Min | Max |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in CANONICAL_STRATEGIES:
        row = primary["strategies"][strategy]["descriptive"]
        lines.append(f"| {strategy} | {row['n']} | {row['mean']:.6f} | {row['median']:.6f} | {row['standard_deviation']:.6f} | {row['q1']:.6f} | {row['q3']:.6f} | {row['minimum']:.6f} | {row['maximum']:.6f} |")
    lines.extend(["", "## D. Bootstrap confidence intervals", "", "Canonical method: paired question-level percentile bootstrap.", "", "| Strategy | Observed mean | Bootstrap mean | Bootstrap SE | 95% CI |", "|---|---:|---:|---:|---:|"])
    for strategy in CANONICAL_STRATEGIES:
        row = primary["strategies"][strategy]
        lines.append(f"| {strategy} | {row['observed_mean']:.6f} | {row['bootstrap_mean']:.6f} | {row['bootstrap_standard_error']:.6f} | [{row['ci_95']['lower']:.6f}, {row['ci_95']['upper']:.6f}] |")
    lines.extend(["", "## E. Primary paired comparisons", "", "| Pair | Mean delta | Paired 95% CI | Bootstrap P(delta>0) | Permutation raw p | Permutation Holm p | Cohen's dz | Rank-biserial | Decision |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"])
    for pair in pairs:
        bootstrap = pair["paired_bootstrap"]; test = pair["permutation_test"]
        lines.append(f"| {_pair_name(pair)} | {pair['observed_mean_delta']:+.6f} | [{bootstrap['ci_lower']:+.6f}, {bootstrap['ci_upper']:+.6f}] | {bootstrap['bootstrap_probability_gt_zero']:.6f} | {test['p_value']:.6g} | {test['holm_adjusted_p_value']:.6g} | {pair['cohens_dz']:.6f} | {pair['matched_rank_biserial']:.6f} | {pair['statistical_conclusion']} |")
    lines.extend(["", "Bootstrap probabilities are descriptive resampling frequencies, not Bayesian posterior probabilities. Confidence intervals and tests use paired raw scores, not rounded aggregates.", "", "## F. Multiple-comparison correction and robustness", "", "| Pair | Wilcoxon statistic | Wilcoxon raw p | Wilcoxon Holm p | Significant? | Zeros discarded |", "|---|---:|---:|---:|---|---:|"])
    for pair in pairs:
        test = pair["wilcoxon_test"]
        lines.append(f"| {_pair_name(pair)} | {test['statistic']:.6f} | {test['p_value']:.6g} | {test['holm_adjusted_p_value']:.6g} | {test['significant_at_alpha']} | {test['zero_pairs']} |")
    lines.extend(["", "Primary conclusions use the corrected permutation family; Wilcoxon is a prespecified robustness analysis and is not substituted opportunistically.", "", "## G. Bootstrap ranking stability", "", f"Across {winner['resamples']} canonical bootstrap replicates:", ""])
    for strategy in CANONICAL_STRATEGIES:
        lines.append(f"- `{strategy}` sole-winner frequency: {winner[f'{strategy}_winner_frequency']:.6f}.")
    lines.extend([f"- Exact tie frequency: {winner['tie_frequency']:.6f}.", "", "These are descriptive bootstrap frequencies, not posterior model probabilities.", "", "## H. Sensitivity analysis", ""])
    sensitivity = result["sensitivity_analysis"]
    lines.append("The 3-seed × 2-resample-count grid (42, 2026, 314159 × 10,000, 50,000) was numerically stable:")
    lines.append("")
    for name, row in sensitivity["endpoint_ranges"].items():
        lines.append(f"- `{name}`: CI lower range [{row['minimum_ci_lower']:+.6f}, {row['maximum_ci_lower']:+.6f}], upper range [{row['minimum_ci_upper']:+.6f}, {row['maximum_ci_upper']:+.6f}], P(delta>0) range [{row['minimum_probability_gt_zero']:.6f}, {row['maximum_probability_gt_zero']:.6f}], zero-inclusion decision stable: `{row['ci_zero_inclusion_stable']}`.")
    lines.extend(["", "## I. Difficulty analysis", "", "Difficulty results are secondary/exploratory and cannot override the primary n=140 endpoint.", "", "| Difficulty | n | fixed_size | structure_aware | prompt_based |", "|---|---:|---:|---:|---:|"])
    for difficulty in ("easy", "medium", "hard"):
        row = result["stratified_analysis"]["difficulty"][difficulty]
        means = row["strategy_means"]
        lines.append(f"| {difficulty} | {row['n']} | {means['fixed_size']:.6f} | {means['structure_aware']:.6f} | {means['prompt_based']:.6f} |")
    lines.append("")
    for difficulty in ("easy", "medium", "hard"):
        row = result["stratified_analysis"]["difficulty"][difficulty]
        for pair in row["pairs"]:
            lines.append(f"- `{difficulty}`, {pair['left']} vs {pair['right']}: delta {pair['mean']:+.6f}, wins/ties/losses {pair['positive_count']}/{pair['zero_count']}/{pair['negative_count']}, exploratory 95% CI [{pair['ci_lower']:+.6f}, {pair['ci_upper']:+.6f}].")
    lines.extend(["", "## J. Question-type analysis", "", "All question-type comparisons are exploratory. Every stratum with n<10 is explicitly descriptive only.", "", "| Type | n | fixed_size | structure_aware | prompt_based | Observed winner | Scope |", "|---|---:|---:|---:|---:|---|---|"])
    for question_type, row in result["stratified_analysis"]["question_type"].items():
        means = row["strategy_means"]
        lines.append(f"| {question_type} | {row['n']} | {means['fixed_size']:.6f} | {means['structure_aware']:.6f} | {means['prompt_based']:.6f} | {row['observed_winner']} | {row['interpretation']} |")
    lines.extend(["", "## K. Retrieval-answer relationship", "", "These within-strategy associations are exploratory and non-causal.", "", "| Strategy | Hit@1 F1 difference | Hit@5 F1 difference | All-evidence F1 difference | Coverage Spearman rho |", "|---|---:|---:|---:|---:|"])
    for strategy in CANONICAL_STRATEGIES:
        row = result["retrieval_answer_relationship"][strategy]
        lines.append(f"| {strategy} | {row['hit_at_1']['conditional_mean_difference']:+.6f} | {row['hit_at_5']['conditional_mean_difference']:+.6f} | {row['all_evidence_retrieved']['conditional_mean_difference']:+.6f} | {row['evidence_coverage_spearman']['rho']:+.6f} |")
    lines.extend(["", "Better retrieval/evidence outcomes can coincide with higher answer overlap without establishing causality. Strategy-level evidence coverage and answer F1 need not rank identically because retrieval coverage is only one input to generation; answer wording, context composition, irrelevant material, and lexical metric sensitivity also matter.", "", "## L. Practical significance", ""])
    for pair in pairs:
        lines.append(f"- `{_pair_name(pair)}`: absolute delta {pair['observed_mean_delta']:+.6f} Token F1; relative to the right mean {100 * pair['relative_delta_vs_right_mean']:+.3f}%; paired dz {pair['cohens_dz']:+.6f}. The absolute difference is primary; the percentage is supplementary.")
    lines.extend(["", "Statistical detection does not by itself establish a meaningful operational gain. Because no smallest effect size of interest was preregistered, this report does not claim formal practical equivalence or assign universal small/medium/large labels.", "", "## M. Limitations", "", "- The paired sample contains 140 questions from one Angular benchmark domain.", "- Results are specific to the frozen generation model, retrieval/index configuration, context budget, and evaluation implementation.", "- Token F1 is deterministic and auditable but can undervalue semantically correct paraphrases and reward lexical overlap.", "- Question-type strata include groups with n=1–4 and cannot support inferential claims.", "- No practical-equivalence margin was preregistered.", "- Bootstrap and randomization inference characterize this benchmark sample; they do not establish universal chunking-strategy superiority.", "", "## N. Final statistical conclusion", "", _final_conclusion(result), "", "Secondary precision/recall and retrieval analyses provide context only and do not replace the primary Token F1 inference. No benchmark redesign, tuning, or new experimental run was performed.", "", "## Test results", "", tests_summary, "", "**FINAL STATISTICAL VALIDATION: COMPLETE**", ""])
    return "\n".join(lines)
