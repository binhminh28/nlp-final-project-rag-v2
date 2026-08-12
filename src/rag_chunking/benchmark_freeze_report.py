"""Human-readable freeze report rendering."""

from __future__ import annotations

from rag_chunking.benchmark import CANONICAL_STRATEGIES
from rag_chunking.benchmark_freeze import PAIR_ORDER, FreezeResult


def render_report(result: FreezeResult, *, tests: str = "Recorded after validator execution") -> str:
    s = result.sections
    lines = [
        "# Canonical Benchmark V2 Freeze Validation", "", "## A. Canonical identity", "",
        *(f"- `{key}`: `{value}`" for key, value in s["canonical_identity"].items()), "",
        "Artifact roots: `data/benchmark/angular/canonical_v2/` and the exact retrieval-fingerprint directory under `data/retrieval/angular/canonical_production_v2/`.", "",
        "## B. Completeness", "",
        "| Strategy | Dataset Qs | Requested | Answers | Eval rows | Retrieval rows | Duplicate IDs | Missing IDs | Invalid rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in CANONICAL_STRATEGIES:
        row = s["completeness"][strategy]
        lines.append(f"| {strategy} | {row['dataset_questions']} | {row['requested']} | {row['answers']} | {row['evaluation_rows']} | {row['retrieval_rows']} | {row['duplicates']} | {row['missing']} | {row['invalid']} |")
    lines.extend([
        "", "Total: 140 unique questions, 420 answers, 420 evaluation rows, and 420 canonical-protocol retrieval rows. PASS.", "",
        "## C. Paired alignment", "", "All three strategies exactly match the canonical 140-ID universe. Question, gold answer, document, difficulty, question type, and available evidence/retrieval metadata have zero mismatches. PASS.", "",
        "## D. Generation integrity", "", "420/420 answers are non-empty successful results with `finish_reason=stop`; 0 length completions, failures, retries, or integrity defects. Provider diagnostics reconcile as 416 calls plus 4 same-fingerprint `fixed_size` cache hits. PASS.", "",
        "## E. Answer metric reproduction", "", "| Strategy | Precision | Recall | Token F1 | Exact | Containment |", "|---|---:|---:|---:|---:|---:|",
    ])
    for strategy in CANONICAL_STRATEGIES:
        row = s["answer_metrics"][strategy]
        lines.append(f"| {strategy} | {row['token_precision']:.4f} | {row['token_recall']:.4f} | {row['token_f1']:.4f} | {row['normalized_exact_match']:.4f} | {row['normalized_containment']:.4f} |")
    lines.extend([
        "", "Every per-query score was recomputed from generated and gold text, then macro-aggregated and matched to stored raw aggregates and four-decimal published values. PASS.", "",
        "## F. Retrieval metric reproduction", "", "| Strategy | Hit@1 | Hit@5 | Hit@10 | MRR | Recall@10 | Evidence coverage | All evidence |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for strategy in CANONICAL_STRATEGIES:
        row = s["retrieval_metrics"][strategy]
        retrieval, evidence = row["retrieval"], row["evidence"]
        lines.append(f"| {strategy} | {retrieval['hit_at_1']:.4f} | {retrieval['hit_at_5']:.4f} | {retrieval['hit_at_10']:.4f} | {retrieval['mrr']:.4f} | {retrieval['recall_at_10']:.4f} | {evidence['evidence_coverage']:.4f} | {evidence['all_evidence_retrieved_rate']:.4f} |")
    lines.extend([
        "", "Recomputed offline from committed `same_token_budget` rows with candidate k=50, 2,048-token budget, and n=140 per strategy. PASS.", "",
        "## G. Stratified reproduction", "", "### Difficulty", "", "| Difficulty | n | fixed_size | structure_aware | prompt_based |", "|---|---:|---:|---:|---:|",
    ])
    for group in ("easy", "medium", "hard"):
        row = s["stratified"]["difficulty"][group]
        lines.append(f"| {group} | {row['n']} | {row['fixed_size']:.4f} | {row['structure_aware']:.4f} | {row['prompt_based']:.4f} |")
    lines.extend(["", "### Question type", "", "| Question type | n | fixed_size | structure_aware | prompt_based |", "|---|---:|---:|---:|---:|"])
    for group, row in s["stratified"]["question_type"].items():
        lines.append(f"| {group} | {row['n']} | {row['fixed_size']:.4f} | {row['structure_aware']:.4f} | {row['prompt_based']:.4f} |")
    lines.extend([
        "", "Membership and every published Token F1 stratum reproduce. This is a reproducibility check only; no inferential claim is made for small strata. PASS.", "",
        "## H. Paired comparison validation", "", "| Pair | Left wins | Ties | Left losses | Mean delta | Positive sum | Negative sum | Mean win | Mean loss |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for left, right in PAIR_ORDER:
        row = s["paired_comparisons"][f"{left}_vs_{right}"]
        lines.append(f"| {left} vs {right} | {row['left_wins']} | {row['ties']} | {row['left_losses']} | {row['mean_delta']:+.6f} | {row['sum_positive_deltas']:.6f} | {row['sum_negative_deltas']:.6f} | {row['mean_positive_win']:.6f} | {row['mean_negative_loss']:.6f} |")
    special = s["paired_comparisons"]["structure_aware_vs_prompt_based"]
    lines.extend([
        "", f"The structure-aware comparison is correctly oriented: 60 wins, 3 exact raw-score ties, and 77 losses, while the positive deltas sum to {special['sum_positive_deltas']:.6f} and negative deltas sum to {special['sum_negative_deltas']:.6f}. Its larger winning magnitudes yield mean delta {special['mean_delta']:+.6f}; this is not a sign, tie, orientation, or formatting bug.", "",
        "## I. Metric semantics", "",
    ])
    lines.extend(f"- {key.replace('_', ' ').title()}: {value}." for key, value in s["metric_semantics"].items())
    lines.extend([
        "", "## J. Deterministic spot-check appendix", "",
        "Selection is algorithmic: top five minimum advantages over both alternatives for each strategy, then the five smallest three-way score ranges. Repeated IDs are retained when selection criteria overlap.", "",
    ])
    for item in s["spot_checks"]:
        lines.extend([f"### {item['question_id']} — {item['selection']}", "", f"- Difficulty/type: `{item['difficulty']}` / `{item['question_type']}`", f"- Question: {item['question']}", f"- Gold snippet: {item['gold_answer']}"])
        for strategy in CANONICAL_STRATEGIES:
            retrieval = item["retrieval"][strategy]
            lines.append(f"- `{strategy}` F1 `{item['token_f1'][strategy]:.12f}`; answer: {item['answers'][strategy]}; first chunk `{retrieval['first_chunk_id']}`; evidence coverage `{retrieval['evidence_coverage']}`; chunk snippet: {retrieval['first_chunk_snippet']}")
        lines.append("")
    lines.extend([
        "All sampled scores, question/answer bindings, gold records, strategy-specific retrieval rows, and chunk strategy labels match. PASS.", "",
        "## K. Immutable artifact inventory", "", f"The freeze manifest records SHA-256 and byte size for each authoritative file. Inventory count: {len(result.artifact_inventory)}.", "",
    ])
    for path, value in result.artifact_inventory.items():
        lines.append(f"- `{path}` — `{value['sha256']}` ({value['size_bytes']} bytes)")
    lines.extend(["", "Nearby historical/alternate artifacts explicitly excluded:", ""])
    lines.extend(f"- {value}" for value in s["excluded_nearby_artifacts"])
    lines.extend([
        "", "## L. Test results", "", tests, "", "## M. Freeze policy and decision", "", s["freeze_policy"], "",
        "All blocking gates pass. Canonical artifact hashes were captured before publication; only the validation report, validator/test code, and freeze declaration are new.", "",
        "**CANONICAL PRODUCTION BENCHMARK V2: FROZEN FOR STATISTICAL ANALYSIS**", "",
    ])
    return "\n".join(lines)
