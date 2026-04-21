from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class EvaluateRequest(BaseModel):
    model: str = Field(default="ollama/qwen2.5vl:7b")
    data_dir: str = Field(default="data")
    results_dir: str = Field(default="results")
    sample: int | None = Field(default=10, ge=1)
    request_delay: float = Field(default=0.0, ge=0.0)
    max_retries: int = Field(default=3, ge=0)
    retry_backoff: float = Field(default=2.0, ge=0.0)


class GenerateRequest(BaseModel):
    model: str = Field(default="ollama/qwen2.5vl:7b")
    question: str
    image_path: str


class GenerateResponse(BaseModel):
    answer: str
    confidence: float | None = None
    refused: bool
    raw_text: str
