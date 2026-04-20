from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict
from typing import Iterable

from .types import PredictionRecord


YES_SET = {"yes", "y", "হ্যাঁ", "হ্যা", "হঁ্যা", "haan", "ha", "yes."}
NO_SET = {"no", "n", "না", "nah", "na", "no."}
REFUSAL_PATTERNS = (
    "i cannot",
    "i can't",
    "not sure",
    "cannot determine",
    "unable to",
    "no idea",
    "unknown",
    "refuse",
    "না জানি",
    "জানি না",
)


def _normalize_digits(text: str) -> str:
    mapping = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
    return text.translate(mapping)


def normalize_answer(text: str) -> str:
    text = _normalize_digits(text or "").strip().lower()
    text = re.sub(r"[\s\u200b]+", " ", text)
    text = re.sub(r"[^\w\d\u0980-\u09ff\s\.-]", "", text)
    return text.strip()


def is_refusal(text: str) -> bool:
    normalized = normalize_answer(text)
    return any(pattern in normalized for pattern in REFUSAL_PATTERNS)


def _extract_numeric_value(text: str) -> float | None:
    normalized = normalize_answer(text)
    match = re.search(r"-?\d+(?:\.\d+)?", normalized)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _is_yes_no(answer: str) -> bool:
    normalized = normalize_answer(answer)
    return normalized in YES_SET or normalized in NO_SET


def _type_label(question_type: str, answer: str, question: str) -> str:
    qtype = (question_type or "").strip().lower()
    if qtype and qtype != "unknown":
        return qtype
    if _is_yes_no(answer):
        return "polar"
    if _extract_numeric_value(answer) is not None:
        return "numeric"
    normalized_question = normalize_answer(question)
    if any(token in normalized_question for token in ("কত", "how many", "সংখ্যা", "count")):
        return "numeric"
    if any(token in normalized_question for token in ("কী", "কি", "what", "which", "who", "where", "when", "why", "how")):
        return "descriptive"
    return "descriptive"


def _confidence_to_unit(value: float | None) -> float | None:
    if value is None:
        return None
    confidence = float(value)
    if confidence > 1.0:
        confidence /= 100.0
    return max(0.0, min(1.0, confidence))


def compute_metrics(predictions: list[PredictionRecord]) -> dict[str, object]:
    total = len(predictions)
    answered = [pred for pred in predictions if not pred.refused]
    correct = [pred for pred in predictions if pred.correct]

    by_type: dict[str, list[PredictionRecord]] = defaultdict(list)
    for prediction in predictions:
        by_type[prediction.answer_type or "unknown"].append(prediction)

    def accuracy(rows: Iterable[PredictionRecord]) -> float:
        rows = list(rows)
        return (sum(1 for row in rows if row.correct) / len(rows)) if rows else 0.0

    metrics: dict[str, object] = {
        "total_samples": total,
        "attempt_rate": (len(answered) / total) if total else 0.0,
        "cga": (len(correct) / len(answered)) if answered else 0.0,
        "f_score": 0.0,
        "accuracy": accuracy(predictions),
        "polar_accuracy": accuracy(pred for pred in predictions if _type_label(pred.question_type, pred.ground_truth, pred.question) == "polar"),
        "numeric_accuracy": accuracy(pred for pred in predictions if _type_label(pred.question_type, pred.ground_truth, pred.question) == "numeric"),
        "descriptive_accuracy": accuracy(pred for pred in predictions if _type_label(pred.question_type, pred.ground_truth, pred.question) == "descriptive"),
        "hallucination_rate": 0.0,
        "ece": 0.0,
        "logical_consistency": 0.0,
        "per_type_accuracy": {},
    }

    attempt_rate = float(metrics["attempt_rate"])
    cga = float(metrics["cga"])
    metrics["f_score"] = (2 * attempt_rate * cga / (attempt_rate + cga)) if (attempt_rate + cga) else 0.0

    conf_rows = [pred for pred in predictions if pred.confidence is not None]
    if conf_rows:
        bins = 10
        bin_totals = [0] * bins
        bin_correct = [0] * bins
        bin_confidence = [0.0] * bins
        hallucinations = 0
        for row in conf_rows:
            confidence = _confidence_to_unit(row.confidence)
            if confidence is None:
                continue
            index = min(bins - 1, int(confidence * bins))
            bin_totals[index] += 1
            bin_correct[index] += int(row.correct)
            bin_confidence[index] += confidence
            if confidence >= 0.7 and not row.correct:
                hallucinations += 1
        ece = 0.0
        for total_in_bin, correct_in_bin, confidence_sum in zip(bin_totals, bin_correct, bin_confidence):
            if total_in_bin == 0:
                continue
            avg_acc = correct_in_bin / total_in_bin
            avg_conf = confidence_sum / total_in_bin
            ece += (total_in_bin / len(conf_rows)) * abs(avg_acc - avg_conf)
        metrics["ece"] = ece
        metrics["hallucination_rate"] = hallucinations / len(conf_rows)

    per_type_accuracy = {
        label: accuracy(rows)
        for label, rows in sorted(by_type.items())
    }
    metrics["per_type_accuracy"] = per_type_accuracy

    pair_groups: dict[str, list[PredictionRecord]] = defaultdict(list)
    for row in predictions:
        if row.pair_id:
            pair_groups[row.pair_id].append(row)
    pair_scores = []
    for rows in pair_groups.values():
        if len(rows) < 2:
            continue
        pair_scores.append(1.0 if rows[0].prediction != rows[1].prediction else 0.0)
    metrics["logical_consistency"] = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0

    return metrics


def record_to_json(record: PredictionRecord) -> dict[str, object]:
    return asdict(record)
