from __future__ import annotations

import tempfile
from pathlib import Path

from banglabench.evaluator import evaluate
from banglabench.models import build_model
from banglabench.types import ModelResponse, QuestionSample


def run_evaluation(
    model: str,
    data_dir: str,
    results_dir: str,
    sample: int | None,
    request_delay: float,
    max_retries: int,
    retry_backoff: float,
) -> dict[str, object]:
    return evaluate(
        model_name=model,
        data_dir=data_dir,
        results_dir=results_dir,
        sample=sample,
        request_delay=request_delay,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
    )


def run_single_generation(model: str, question: str, image_path: str) -> ModelResponse:
    sample = QuestionSample(
        sample_id="adhoc",
        question=question,
        answer="",
        image_path=Path(image_path),
        question_type="unknown",
        answer_type="unknown",
    )
    vlm = build_model(model)
    return vlm.generate(sample)


def run_single_generation_from_upload(model: str, question: str, image_bytes: bytes, filename: str) -> ModelResponse:
    suffix = Path(filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(image_bytes)
        temp_path = Path(tmp.name)
    try:
        return run_single_generation(model=model, question=question, image_path=str(temp_path))
    finally:
        temp_path.unlink(missing_ok=True)
