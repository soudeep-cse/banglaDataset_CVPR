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


# Caption evaluation metrics

def _normalize_caption(text: str) -> str:
    """Normalize caption for comparison."""
    normalized = unicodedata.normalize("NFC", text or "")
    normalized = normalized.lower()
    # Remove punctuation and extra spaces
    normalized = re.sub(r"[^\w\s\u0980-\u09ff]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _tokenize_bangla(text: str) -> list[str]:
    """Simple tokenization for Bangla text."""
    normalized = _normalize_caption(text)
    return normalized.split()


def caption_exact_match(predicted: str, reference: str) -> bool:
    """Check if predicted caption exactly matches reference."""
    return _normalize_caption(predicted) == _normalize_caption(reference)


def caption_token_overlap(predicted: str, reference: str) -> float:
    """Calculate token overlap between predicted and reference captions.

    Returns overlap ratio (0-1) based on common tokens.
    """
    pred_tokens = set(_tokenize_bangla(predicted))
    ref_tokens = set(_tokenize_bangla(reference))

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = pred_tokens & ref_tokens
    union = pred_tokens | ref_tokens

    return len(common) / len(union) if union else 0.0


def caption_similarity_metrics(predicted: str, reference: str) -> dict[str, float]:
    """Compute multiple similarity metrics for caption evaluation.

    Returns dict with:
    - exact_match: 1 if exact match, 0 otherwise
    - token_overlap: Jaccard-like token overlap ratio
    - length_ratio: predicted length / reference length
    """
    pred_norm = _normalize_caption(predicted)
    ref_norm = _normalize_caption(reference)

    pred_tokens = _tokenize_bangla(predicted)
    ref_tokens = _tokenize_bangla(reference)

    # Exact match
    exact = 1.0 if pred_norm == ref_norm else 0.0

    # Token overlap (Jaccard)
    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)
    if pred_set and ref_set:
        jaccard = len(pred_set & ref_set) / len(pred_set | ref_set)
    else:
        jaccard = 0.0

    # Length ratio
    if ref_tokens:
        length_ratio = len(pred_tokens) / len(ref_tokens)
    else:
        length_ratio = 0.0

    return {
        "exact_match": exact,
        "token_overlap": jaccard,
        "length_ratio": length_ratio,
    }


def compute_caption_metrics(results: list[dict[str, Any]], ece_bins: int = 10, oer_threshold: float = 0.7) -> dict[str, object]:
    """Compute metrics for caption evaluation.

    Uses token overlap as "correctness" metric since captions are free-form.
    """
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "accuracy": 0.0,
            "oer": 0.0,
            "ece": 0.0,
            "brier": 0.0,
            "mcw": 0.0,
            "avg_token_overlap": 0.0,
            "avg_length_ratio": 0.0,
        }

    # For captions, use token overlap as correctness threshold
    # Overlap > 0.5 is considered "correct"
    correct_threshold = 0.5

    similarities = []
    for item in results:
        pred = item.get("predicted", "")
        ref = item.get("ground_truth", "")
        sims = caption_similarity_metrics(pred, ref)
        similarities.append(sims)
        item["correct"] = sims["token_overlap"] >= correct_threshold
        item["token_overlap"] = sims["token_overlap"]

    # Now compute standard metrics
    metrics = compute_metrics(results, ece_bins=ece_bins, oer_threshold=oer_threshold)

    # Add caption-specific metrics
    avg_overlap = sum(s["token_overlap"] for s in similarities) / total
    avg_length_ratio = sum(s["length_ratio"] for s in similarities) / total

    metrics["avg_token_overlap"] = avg_overlap
    metrics["avg_length_ratio"] = avg_length_ratio

    return metrics
