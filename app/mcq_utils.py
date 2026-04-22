"""MCQ utilities for BanglaVerse VQA/CSU tasks."""
from __future__ import annotations

import re
from typing import Any

from .types import BanglaVerseSample, MCQResponse


def build_mcq_prompt(question: str, options: list[str]) -> str:
    """Build MCQ prompt with A/B/C/D options.

    Args:
        question: The question text
        options: List of 4 options

    Returns:
        Formatted prompt string
    """
    letters = ["A", "B", "C", "D"]
    lines = [f"Question: {question}", ""]

    for i, option in enumerate(options[:4]):
        letter = letters[i] if i < len(letters) else str(i + 1)
        lines.append(f"{letter}) {option}")

    lines.extend([
        "",
        "Answer with only the letter (A, B, C, or D) or the option text.",
        "Also provide a confidence score from 0 to 100.",
        "Return JSON: {\"answer\": \"A\", \"confidence\": 85}",
    ])

    return "\n".join(lines)


def extract_mcq_answer(response_text: str, options: list[str]) -> MCQResponse:
    """Extract MCQ answer from model response.

    Handles various response formats:
    - JSON: {"answer": "A", "confidence": 85}
    - JSON: {"answer": "option text", "confidence": 85}
    - Single letter: "A"
    - Full text: "option text"
    - Letter with text: "A) option text"

    Args:
        response_text: Raw model response
        options: List of valid options

    Returns:
        MCQResponse with selected option and index
    """
    raw_text = response_text.strip()
    selected_option = ""
    selected_index: int | None = None
    confidence: float | None = None

    # Try JSON extraction first
    match = re.search(r"\{[^}]*\}", raw_text)
    if match:
        try:
            import json
            payload = json.loads(match.group(0))
            answer_val = str(payload.get("answer", "")).strip()
            confidence_val = payload.get("confidence", payload.get("score"))
            if confidence_val is not None:
                confidence = float(confidence_val)

            # Map answer to option
            result = _match_answer_to_option(answer_val, options)
            selected_option = result["option"]
            selected_index = result["index"]
        except (json.JSONDecodeError, ValueError):
            pass

    # If no JSON, try pattern matching on full text
    if selected_index is None:
        result = _match_answer_to_option(raw_text, options)
        selected_option = result["option"]
        selected_index = result["index"]

        # Try to extract confidence from text patterns
        conf_match = re.search(r"confidence\s*[:=]\s*(\d+(?:\.\d+)?)", raw_text, re.I)
        if conf_match:
            confidence = float(conf_match.group(1))

    return MCQResponse(
        selected_option=selected_option,
        selected_index=selected_index,
        confidence=confidence,
        raw_text=raw_text,
        refused=False,
    )


def _match_answer_to_option(answer_text: str, options: list[str]) -> dict[str, Any]:
    """Match answer text to one of the options.

    Returns dict with "option" and "index" keys.
    """
    letters = ["A", "B", "C", "D"]
    answer_clean = answer_text.strip().upper()

    # Check if answer is a single letter A/B/C/D
    if answer_clean in letters:
        idx = letters.index(answer_clean)
        if idx < len(options):
            return {"option": options[idx], "index": idx}

    # Check for pattern like "A) option text"
    letter_match = re.match(r"^([A-D])[).:]?\s*(.+)", answer_text.strip(), re.I)
    if letter_match:
        idx = letters.index(letter_match.group(1).upper())
        if idx < len(options):
            return {"option": options[idx], "index": idx}

    # Try exact match with options (case-insensitive)
    answer_lower = answer_text.strip().lower()
    for i, opt in enumerate(options):
        if answer_lower == opt.lower():
            return {"option": opt, "index": i}

    # Try substring match (partial)
    for i, opt in enumerate(options):
        if opt.lower() in answer_lower or answer_lower in opt.lower():
            return {"option": opt, "index": i}

    # No match found - return raw answer as option
    return {"option": answer_text, "index": None}


def is_mcq_correct(predicted: MCQResponse, sample: BanglaVerseSample) -> bool:
    """Check if MCQ prediction is correct.

    Compares either by index (if available) or by option text.
    """
    if sample.answer_index is not None and predicted.selected_index is not None:
        return predicted.selected_index == sample.answer_index

    # Fall back to text comparison
    pred_text = predicted.selected_option.strip().lower()
    answer_text = sample.answer.strip().lower()

    # Direct match
    if pred_text == answer_text:
        return True

    # Check if prediction matches any option with same index
    if predicted.selected_index is not None and sample.answer_index is not None:
        if sample.answer_index < len(sample.options):
            correct_option = sample.options[sample.answer_index]
            return pred_text == correct_option.lower()

    return False


def build_caption_prompt() -> str:
    """Build prompt for image captioning task."""
    return (
        "Describe this image in Bengali. "
        "Provide a detailed caption that captures the main subject, context, and any important details. "
        "Return JSON: {\"caption\": \"your caption here\", \"confidence\": 90}"
    )


def extract_caption(response_text: str) -> tuple[str, float | None]:
    """Extract caption from model response.

    Returns:
        Tuple of (caption_text, confidence)
    """
    raw_text = response_text.strip()
    caption = raw_text
    confidence: float | None = None

    # Try JSON extraction
    match = re.search(r"\{[^}]*\}", raw_text)
    if match:
        try:
            import json
            payload = json.loads(match.group(0))
            caption = str(payload.get("caption", payload.get("answer", raw_text))).strip()
            confidence_val = payload.get("confidence", payload.get("score"))
            if confidence_val is not None:
                confidence = float(confidence_val)
        except (json.JSONDecodeError, ValueError):
            pass
    else:
        # Try to extract caption field
        caption_match = re.search(r'caption\s*[:=]\s*["\']?([^"\']+)["\']?', raw_text, re.I)
        if caption_match:
            caption = caption_match.group(1).strip()

        # Try confidence
        conf_match = re.search(r"confidence\s*[:=]\s*(\d+(?:\.\d+)?)", raw_text, re.I)
        if conf_match:
            confidence = float(conf_match.group(1))

    return caption, confidence
