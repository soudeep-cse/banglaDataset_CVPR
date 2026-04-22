"""Ollama HTTP client using requests (alternative to ollama Python package)."""
from __future__ import annotations

import base64
import os
from pathlib import Path

import requests

from .base import BaseVLM, parse_model_output
from ..types import ModelResponse, QuestionSample


class OllamaHTTPVLM(BaseVLM):
    """Ollama VLM using direct HTTP requests (like abc.py)."""

    def __init__(self, name: str, model_id: str, host: str | None = None):
        super().__init__(name=name)
        self.model_id = model_id
        # Host is passed from CLI, or fall back to env var / default
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        # Ensure no trailing slash and construct API URL
        self.api_url = self.host.rstrip("/") + "/api/chat"

    def generate(self, sample: QuestionSample) -> ModelResponse:
        # Read and encode image
        with open(Path(sample.image_path), "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt = self.build_prompt(sample.question)

        # Make HTTP request like abc.py
        response = requests.post(
            self.api_url,
            json={
                "model": self.model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_b64],
                    }
                ],
                "stream": False,
            },
            timeout=300,  # 5 minute timeout for large images
        )
        response.raise_for_status()

        data = response.json()
        # Extract message content from response
        message = data.get("message", {})
        content = message.get("content", "")

        return parse_model_output(content)


def build_ollama_http_model(name: str, model_id: str, host: str | None = None) -> OllamaHTTPVLM:
    return OllamaHTTPVLM(name=name, model_id=model_id, host=host)
