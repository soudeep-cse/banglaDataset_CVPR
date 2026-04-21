from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image

from ..metrics import is_refusal
from ..types import ModelResponse, QuestionSample


SYSTEM_PROMPT = (
    "You are a careful vision-language assistant for Bangla VQA. "
    "Answer only with the final answer and a confidence score. "
    "Return JSON with keys answer and confidence, where confidence is a number from 0 to 100."
)


def load_image(image_path: str | Path) -> Image.Image:
    image = Image.open(image_path)
    return image.convert("RGB")


def parse_model_output(text: str) -> ModelResponse:
    raw_text = text.strip()
    answer = raw_text
    confidence: float | None = None
    match = re.search(r"\{.*\}", raw_text, flags=re.S)
    if match:
        try:
            payload = json.loads(match.group(0))
            answer = str(payload.get("answer", payload.get("prediction", raw_text))).strip()
            confidence_value = payload.get("confidence", payload.get("score"))
            confidence = float(confidence_value) if confidence_value is not None else None
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    else:
        confidence_match = re.search(r"confidence\s*[:=]\s*(\d+(?:\.\d+)?)", raw_text, flags=re.I)
        if confidence_match:
            confidence = float(confidence_match.group(1))
        answer_match = re.search(r"answer\s*[:=]\s*(.+)", raw_text, flags=re.I)
        if answer_match:
            answer = answer_match.group(1).strip()
    refused = is_refusal(answer)
    return ModelResponse(answer=answer, confidence=confidence, raw_text=raw_text, refused=refused)


class BaseVLM(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name

    def build_prompt(self, question: str) -> str:
        return (
            f"{SYSTEM_PROMPT}\n\n"
            "Question: " + question.strip() + "\n"
            "Respond with JSON only."
        )

    @abstractmethod
    def generate(self, sample: QuestionSample) -> ModelResponse:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name}
