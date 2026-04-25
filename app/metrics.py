from __future__ import annotations

import os
import re
import unicodedata
from typing import Any

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed   

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


def compare_answers(predicted: str, ground_truth: str, segment: str, question: str = "") -> bool:
    """
    Compare predicted answer with ground truth.

    For descriptive answers, uses LLM-as-Judge for semantic similarity
    instead of exact string matching. Question context is provided for better judgment.
    
    Args:
        predicted: Model's predicted answer
        ground_truth: Ground truth answer
        segment: Answer segment type (polar, numeric, descriptive)
        question: The question text (for descriptive answers, helps with semantic matching)
    """
    norm_pred = normalize_for_segment(predicted, segment)
    norm_truth = normalize_for_segment(ground_truth, segment)
    # For descriptive answers, use semantic similarity with question context
    if segment == "descriptive":
        return _semantic_match(norm_pred, norm_truth, question)
    # For polar and numeric, use exact match
    return norm_pred == norm_truth


def _confidence_to_unit(value: float | None) -> float | None:
    if value is None:
        return None
    confidence = float(value)
    if confidence > 1.0:
        confidence /= 100.0
    return max(0.0, min(1.0, confidence))


# Hybrid Semantic Similarity: Embedding + LLM Judge
# Embedding is fast and handles most synonyms, LLM handles edge cases
_EMBEDDING_MODEL = None
_EMBEDDING_CACHE = {}
_LLM_JUDGE_CACHE = {}
_LLM_JUDGE_MODEL = None
_LLM_HOST = None


def _get_embedding_model():
    """Lazy-load the embedding model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDING_MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        except ImportError:
            return None
    return _EMBEDDING_MODEL


def _get_llm_judge_config():
    """Get LLM judge configuration from environment."""
    global _LLM_JUDGE_MODEL, _LLM_HOST
    if _LLM_JUDGE_MODEL is None:
        _LLM_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen2.5:7b")
        _LLM_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    return _LLM_JUDGE_MODEL, _LLM_HOST


def _compute_embedding_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity using embeddings (fast)."""
    cache_key = (text1.strip().lower(), text2.strip().lower())
    if cache_key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[cache_key]
    
    model = _get_embedding_model()
    if model is None:
        return 0.5  # Neutral if model not available
    
    try:
        from sentence_transformers import util
        embeddings = model.encode([text1, text2], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        _EMBEDDING_CACHE[cache_key] = similarity
        return similarity
    except Exception:
        return 0.5


def _llm_judge_similarity(predicted: str, ground_truth: str, question: str = "", embedding_score: float = 0.0) -> bool:
    """
    Use LLM to judge if two answers are semantically equivalent.
    Only called for borderline cases where embedding similarity is unclear.
    Returns True if they mean the same thing.
    """
    cache_key = (predicted.strip().lower(), ground_truth.strip().lower(), question.strip().lower())
    if cache_key in _LLM_JUDGE_CACHE:
        return _LLM_JUDGE_CACHE[cache_key]
    
    model, host = _get_llm_judge_config()
    
    # Build prompt with embedding score context
    prompt = f"""You are a semantic similarity judge for Bangla (Bengali) answers.

Question: {question}
Ground Truth: {ground_truth}
Predicted: {predicted}

Task: Do these two answers convey the same meaning in the context of the question?
Important: Be LENIENT - if they refer to the same thing/concept, answer "yes" even if wording differs.

Examples where answers are equivalent (YES):
- "জলের বাটি" vs "পানির বালি" → both are water containers (yes)
- "জল" vs "পানির দিকে" → both refer to water (yes)
- "না" vs "নেই" → both mean "no/none" (yes)
- "বোটে" vs "নৌকায়" → both mean "in boat" (yes)

Examples where answers are different (NO):
- "বিড়াল" vs "কুকুর" → different animals (no)
- "লাল" vs "নীল" → different colors (no)
- "ডানদিকে" vs "বাঁদিকে" → opposite directions (no)

Previous similarity score: {embedding_score:.2f} (0=different, 1=same)

Respond with ONLY "yes" or "no" (be lenient, prefer "yes" if uncertain):

Answer:"""
    try:
        response = requests.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 100}
            },
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        answer = result.get("response", "").strip().lower()
        print(f"LLM raw response: '{answer}'")
        
        # Parse yes/no - be more flexible
        is_match = False
        if answer:
            # Check for yes (English or Bangla)
            if any(x in answer for x in ["yes", "হ্যাঁ", "true", "1", "equivalent", "same"]):
                is_match = True
            elif any(x in answer for x in ["no", "না", "false", "0", "different"]):
                is_match = False
            else:
                # If unclear, default to substring matching
                is_match = predicted.strip().lower() in ground_truth.strip().lower() or \
                          ground_truth.strip().lower() in predicted.strip().lower()
        
        _LLM_JUDGE_CACHE[cache_key] = is_match
        return is_match
        
    except Exception as e:
        # Fallback to simple matching if LLM fails
        return predicted.strip().lower() == ground_truth.strip().lower()


def _semantic_match(predicted: str, ground_truth: str, question: str = "", threshold: float = 0.65) -> bool:
    """
    Check if predicted semantically matches ground truth using hybrid approach.
    
    1. Fast embedding similarity (covers most synonyms)
    2. LLM judge only for borderline cases (0.3 < similarity < 0.8)
    
    Args:
        predicted: Model's predicted answer
        ground_truth: Ground truth answer
        question: The question text (provides context for better judgment)
        threshold: Threshold for embedding similarity (default 0.65)
    
    Returns:
        True if answers are semantically equivalent
    """
    norm_pred = predicted.strip()
    norm_truth = ground_truth.strip()
    
    # Strategy 1: Exact match (fast path)
    if norm_pred.lower() == norm_truth.lower():
        return True
    
    # Strategy 2: Substring match (fast path for short answers)
    if len(norm_pred) <= 20 or len(norm_truth) <= 20:
        if norm_pred in norm_truth or norm_truth in norm_pred:
            return True
    
    # Strategy 3: Embedding-based similarity (fast)
    embedding_sim = _compute_embedding_similarity(norm_pred, norm_truth)
    
    # High similarity = definite match (no LLM call needed)
    if embedding_sim >= 0.75:
        return True
    
    # Low similarity = definite mismatch (no LLM call needed)
    if embedding_sim <= 0.35:
        return False
    
    # Borderline case: use LLM judge
    # Only call LLM for non-trivial cases with borderline embedding scores
    if len(norm_pred) > 2 and len(norm_truth) > 2:
        return _llm_judge_similarity(norm_pred, norm_truth, question, embedding_sim)
    
    # Default to embedding result for very short answers
    return embedding_sim >= threshold


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
