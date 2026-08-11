"""Conservative, versioned lexical answer normalization and metrics."""

from __future__ import annotations

from collections import Counter
import re
import unicodedata

from .answer_models import ANSWER_NORMALIZATION_VERSION, TOKENIZATION_POLICY


_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def normalize_answer(value: str, *, version: str = ANSWER_NORMALIZATION_VERSION) -> str:
    if version != ANSWER_NORMALIZATION_VERSION:
        raise ValueError(f"unsupported answer normalization version {version!r}")
    if not isinstance(value, str):
        raise ValueError("answer must be a string")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def tokenize_answer(value: str, *, policy: str = TOKENIZATION_POLICY) -> list[str]:
    if policy != TOKENIZATION_POLICY:
        raise ValueError(f"unsupported tokenization policy {policy!r}")
    return _TOKEN_PATTERN.findall(value)


def normalized_exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_prf(prediction: str, reference: str) -> tuple[float, float, float]:
    predicted = tokenize_answer(normalize_answer(prediction))
    expected = tokenize_answer(normalize_answer(reference))
    if not predicted and not expected:
        return 1.0, 1.0, 1.0
    if not predicted or not expected:
        return 0.0, 0.0, 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def normalized_containment(prediction: str, reference: str) -> float:
    """Whether reference tokens occur contiguously in prediction tokens.

    Token sequence matching avoids raw-substring false positives such as a gold
    answer ``art`` matching ``partial``. This is a secondary diagnostic only.
    """

    predicted = tokenize_answer(normalize_answer(prediction))
    expected = tokenize_answer(normalize_answer(reference))
    if not expected:
        return float(not predicted)
    if len(expected) > len(predicted):
        return 0.0
    return float(any(predicted[index:index + len(expected)] == expected for index in range(len(predicted) - len(expected) + 1)))


def evaluate_references(
    prediction: str, references: tuple[str, ...], enabled_metrics: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, int]]:
    if not references:
        raise ValueError("at least one reference answer is required")
    per_reference = []
    for reference in references:
        precision, recall, f1 = token_prf(prediction, reference)
        per_reference.append({
            "normalized_exact_match": normalized_exact_match(prediction, reference),
            "token_precision": precision,
            "token_recall": recall,
            "token_f1": f1,
            "normalized_containment": normalized_containment(prediction, reference),
        })
    scores: dict[str, float] = {}
    indexes: dict[str, int] = {}
    for metric in enabled_metrics:
        index = max(range(len(references)), key=lambda item: (per_reference[item][metric], -item))
        scores[metric] = per_reference[index][metric]
        indexes[metric] = index
        if metric == "token_f1":
            scores["token_precision"] = per_reference[index]["token_precision"]
            scores["token_recall"] = per_reference[index]["token_recall"]
            indexes["token_precision"] = index
            indexes["token_recall"] = index
    return scores, indexes
