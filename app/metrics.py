from __future__ import annotations

import re
import unicodedata
from typing import Any


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
TRAILING_PUNCT_RE = re.compile(r"[।?!]+$")


def _normalize_digits(text: str) -> str:
    mapping = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
    return text.translate(mapping)


def normalize_answer(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text or "")
    normalized = normalized.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "…": "..."}))
    normalized = normalized.strip().lower()
    normalized = _normalize_digits(normalized)
    normalized = re.sub(r"[\s\u200b]+", " ", normalized)
    normalized = TRAILING_PUNCT_RE.sub("", normalized)
    normalized = re.sub(r"[^\w\d\u0980-\u09ff\s\.-]", "", normalized)
    return normalized.strip()


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


def infer_segment(question_type: str, answer: str, question: str) -> str:
    qtype = (question_type or "").strip().lower()
    if qtype and qtype != "unknown":
        if qtype in {"yes/no", "yesno", "boolean", "bool"}:
            return "polar"
        if qtype in {"number", "count"}:
            return "numeric"
        if qtype in {"description", "open", "freeform"}:
            return "descriptive"
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


def normalize_for_segment(text: str, segment: str) -> str:
    normalized = normalize_answer(text)
    if segment == "polar":
        if normalized in YES_SET:
            return "হ্যাঁ"
        if normalized in NO_SET:
            return "না"
    if segment == "numeric":
        return _normalize_digits(normalized)
    return normalized


def compare_answers(predicted: str, ground_truth: str, segment: str) -> bool:
    return normalize_for_segment(predicted, segment) == normalize_for_segment(ground_truth, segment)


def _confidence_to_unit(value: float | None) -> float | None:
    if value is None:
        return None
    confidence = float(value)
    if confidence > 1.0:
        confidence /= 100.0
    return max(0.0, min(1.0, confidence))


def _safe_confidence(item: dict[str, Any]) -> float:
    value = _confidence_to_unit(item.get("confidence"))
    return value if value is not None else 0.5


def compute_metrics(results: list[dict[str, Any]], ece_bins: int = 10, oer_threshold: float = 0.7) -> dict[str, object]:
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "accuracy": 0.0,
            "oer": 0.0,
            "ece": 0.0,
            "brier": 0.0,
            "mcw": 0.0,
        }

    correct = sum(1 for item in results if bool(item.get("correct", False)))
    wrong_confidences = [_safe_confidence(item) for item in results if not bool(item.get("correct", False))]
    oer_count = sum(1 for item in results if (not bool(item.get("correct", False))) and _safe_confidence(item) >= oer_threshold)
    brier = sum((_safe_confidence(item) - (1.0 if bool(item.get("correct", False)) else 0.0)) ** 2 for item in results) / total

    bin_totals = [0] * max(1, ece_bins)
    bin_correct = [0] * max(1, ece_bins)
    bin_confidence = [0.0] * max(1, ece_bins)
    for item in results:
        conf = _safe_confidence(item)
        index = min(len(bin_totals) - 1, int(conf * len(bin_totals)))
        bin_totals[index] += 1
        bin_correct[index] += int(bool(item.get("correct", False)))
        bin_confidence[index] += conf

    ece = 0.0
    for total_in_bin, correct_in_bin, confidence_sum in zip(bin_totals, bin_correct, bin_confidence):
        if total_in_bin == 0:
            continue
        avg_acc = correct_in_bin / total_in_bin
        avg_conf = confidence_sum / total_in_bin
        ece += (total_in_bin / total) * abs(avg_acc - avg_conf)

    return {
        "total": total,
        "accuracy": correct / total,
        "oer": oer_count / total,
        "ece": ece,
        "brier": brier,
        "mcw": (sum(wrong_confidences) / len(wrong_confidences)) if wrong_confidences else 0.0,
    }
