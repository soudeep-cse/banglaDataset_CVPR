from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .data import load_dataset
from .metrics import compute_metrics, normalize_answer
from .models import build_model
from .types import PredictionRecord


def _is_correct(predicted: str, ground_truth: str) -> bool:
    return normalize_answer(predicted) == normalize_answer(ground_truth)


def _is_retryable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    retry_markers = (
        "quota",
        "resource_exhausted",
        "429",
        "rate limit",
        "too many requests",
        "temporarily unavailable",
    )
    return any(marker in message for marker in retry_markers)


def evaluate(
    model_name: str,
    data_dir: str | Path = "data",
    results_dir: str | Path = "results",
    sample: int | None = None,
    load_in_4bit: bool = False,
    request_delay: float = 0.0,
    max_retries: int = 3,
    retry_backoff: float = 2.0,
) -> dict[str, object]:
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(data_dir, limit=sample)
    model = build_model(model_name, load_in_4bit=load_in_4bit)

    records: list[PredictionRecord] = []
    for sample_row in samples:
        attempt = 0
        while True:
            try:
                response = model.generate(sample_row)
                break
            except Exception as exc:
                retryable = _is_retryable_error(exc)
                if attempt >= max_retries or not retryable:
                    raise
                wait_seconds = retry_backoff * (2 ** attempt)
                time.sleep(wait_seconds)
                attempt += 1
        prediction = response.answer.strip()
        record = PredictionRecord(
            sample_id=sample_row.sample_id,
            question=sample_row.question,
            ground_truth=sample_row.answer,
            prediction=prediction,
            confidence=response.confidence,
            correct=_is_correct(prediction, sample_row.answer),
            refused=response.refused,
            question_type=sample_row.question_type,
            answer_type=sample_row.answer_type,
            pair_id=sample_row.pair_id,
            raw_text=response.raw_text,
            image_path=str(sample_row.image_path),
        )
        records.append(record)
        if request_delay > 0:
            time.sleep(request_delay)

    metrics = compute_metrics(records)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model_name = model_name.replace("/", "_")
    predictions_path = results_dir / f"{safe_model_name}_predictions.json"
    metrics_path = results_dir / f"{safe_model_name}_metrics.json"

    with predictions_path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(record) for record in records], handle, ensure_ascii=False, indent=2)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": model_name,
                "timestamp": timestamp,
                "metrics": metrics,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    return {"model": model_name, "metrics": metrics, "predictions_path": str(predictions_path), "metrics_path": str(metrics_path)}
