from __future__ import annotations

import base64
import os
from pathlib import Path

from .base import BaseVLM, parse_model_output
from ..types import ModelResponse, QuestionSample


class OllamaVLM(BaseVLM):
    def __init__(self, name: str, model_id: str, host: str | None = None):
        super().__init__(name=name)
        self.model_id = model_id
        # Host is passed from CLI, or fall back to env var / default
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def generate(self, sample: QuestionSample) -> ModelResponse:
        try:
            import ollama
        except ImportError:
            raise RuntimeError("Install ollama Python package: pip install ollama")

        with open(Path(sample.image_path), "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt = self.build_prompt(sample.question)

        client = ollama.Client(host=self.host)

        response = client.chat(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt, "images": [image_b64]}],
        )

        return parse_model_output(response.message.content)


def build_ollama_model(name: str, model_id: str, host: str | None = None) -> OllamaVLM:
    return OllamaVLM(name=name, model_id=model_id, host=host)
