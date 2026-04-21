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
