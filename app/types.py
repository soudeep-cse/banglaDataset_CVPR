from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class QuestionSample:
    sample_id: str
    question: str
    answer: str
    image_path: Path
    question_type: str = "unknown"
    answer_type: str = "unknown"
    pair_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelResponse:
    answer: str
    confidence: float | None = None
    raw_text: str = ""
    refused: bool = False


@dataclass(slots=True)
class PredictionRecord:
    sample_id: str
    question: str
    ground_truth: str
    prediction: str
    confidence: float | None
    correct: bool
    refused: bool
    question_type: str
    answer_type: str
    pair_id: str | None = None
    raw_text: str = ""
    image_path: str = ""


@dataclass(slots=True)
class BanglaVerseSample:
    """Sample for BanglaVerse dataset (VQA, CSU, Captions)."""
    sample_id: str
    task: str  # "vqa", "csu", "captions"
    dialect: str
    domain: str
    image_path: Path
    question: str = ""  # For VQA/CSU
    options: list[str] = field(default_factory=list)  # For VQA/CSU MCQ
    answer: str = ""  # For VQA/CSU: correct option text; For Captions: reference caption
    answer_index: int | None = None  # For VQA/CSU MCQ
    caption: str = ""  # For Captions task
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MCQResponse:
    """Response for MCQ tasks with option selection."""
    selected_option: str
    selected_index: int | None
    confidence: float | None = None
    raw_text: str = ""
    refused: bool = False
