from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = lambda x, **kwargs: x  # Fallback

from .data import load_dataset_from_file
from .metrics import compare_answers, compute_metrics
from .models import build_model
from .preprocess import prepare_segment_files


def _safe_output_stem(model_name: str) -> str:
    # Normalize model names into cross-platform safe filenames.
    safe = re.sub(r"[\\/:*?\"<>|]+", "_", model_name.strip())
    safe = re.sub(r"\s+", "_", safe)
    safe = re.sub(r"_+", "_", safe)
    safe = safe.strip("._")
    return safe or "model"


def _confidence_to_unit(value: float | None) -> float:
    if value is None:
        return 0.5
    confidence = float(value)
    if confidence > 1.0:
        confidence /= 100.0
    return max(0.0, min(1.0, confidence))


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
    data_dir: str | Path = "dataset",
    results_dir: str | Path = "results",
    sample: int | None = None,
    load_in_4bit: bool = False,
    ollama_host: str | None = None,
    preprocessed_dir: str | Path = "preprocessed_dataset",
    skip_preprocess: bool = False,
    oer_threshold: float = 0.7,
    ece_bins: int = 10,
    request_delay: float = 0.0,
    max_retries: int = 3,
    retry_backoff: float = 2.0,
    save_raw: bool = False,
) -> dict[str, object]:
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not skip_preprocess:
        prep_info = prepare_segment_files(data_dir=data_dir, output_dir=preprocessed_dir, limit=sample)
    else:
        prep_info = {
            "output_dir": str(preprocessed_dir),
            "counts": {},
            "files": {
                "polar": str(Path(preprocessed_dir) / "qa_polar.json"),
                "numeric": str(Path(preprocessed_dir) / "qa_numeric.json"),
                "descriptive": str(Path(preprocessed_dir) / "qa_descriptive.json"),
                "all": str(Path(preprocessed_dir) / "qa_all.json"),
                "english": str(Path(preprocessed_dir) / "qa_english.json"),
            },
            "total": 0,
        }

    model = build_model(model_name, load_in_4bit=load_in_4bit, host=ollama_host)

    # Setup raw output directory
    raw_output_dir: Path | None = None
    if save_raw:
        raw_output_dir = results_dir / "raw_outputs"
        raw_output_dir.mkdir(parents=True, exist_ok=True)
        # Clean old files
        for f in raw_output_dir.glob("raw_outputs_*.jsonl"):
            f.unlink()

    all_results: list[dict[str, Any]] = []
    images_dir = data_dir / "images"
    output_root = Path(preprocessed_dir)

    for segment in ("polar", "numeric", "descriptive"):
        qa_file = output_root / f"qa_{segment}.json"
        if not qa_file.exists():
            continue

        samples = load_dataset_from_file(qa_file, images_dir=images_dir)
        if not samples:
            continue

        print(f"\nEvaluating {segment} segment ({len(samples)} samples)...")
        desc = f"Processing {segment}"
        iterator = tqdm(samples, desc=desc, disable=not HAS_TQDM)

        for sample_row in iterator:
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

            predicted = response.answer.strip()
            confidence = _confidence_to_unit(response.confidence)
            correct = compare_answers(predicted=predicted, ground_truth=sample_row.answer, segment=segment)

            result = {
                "qa_id": sample_row.sample_id,
                "type": segment,
                "question": sample_row.question,
                "ground_truth": sample_row.answer,
                "predicted": predicted,
                "confidence": confidence,
                "correct": correct,
                "raw_output": response.raw_text if hasattr(response, 'raw_text') else None,
                "image_path": str(sample_row.image_path) if hasattr(sample_row, 'image_path') else None,
            }
            all_results.append(result)

            # Save raw output incrementally
            if save_raw and raw_output_dir:
                raw_file = raw_output_dir / f"raw_outputs_{segment}.jsonl"
                with raw_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

            # Update progress bar
            if HAS_TQDM:
                correct_count = sum(1 for r in all_results if r.get("correct"))
                accuracy = correct_count / len(all_results) if all_results else 0
                iterator.set_postfix({"acc": f"{accuracy:.2%}", "n": len(all_results)})

            if request_delay > 0:
                time.sleep(request_delay)

    overall = compute_metrics(all_results, ece_bins=ece_bins, oer_threshold=oer_threshold)
    segment_wise = {
        segment: {"n": len(rows), **compute_metrics(rows, ece_bins=ece_bins, oer_threshold=oer_threshold)}
        for segment in ("polar", "numeric", "descriptive")
        for rows in ([entry for entry in all_results if entry.get("type") == segment],)
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "metadata": {
            "model": model_name,
            "timestamp": timestamp,
            "oer_threshold": oer_threshold,
            "ece_bins": ece_bins,
            "preprocessed_dir": str(preprocessed_dir),
            "preprocess_counts": prep_info.get("counts", {}),
            "total_results": len(all_results),
        },
        "overall": overall,
        "segment_wise": segment_wise,
        "results": all_results,
    }
    # Save report with timestamp like BanglaVerse
    safe_name = _safe_output_stem(model_name)
    report_path = results_dir / f"banglabayanno_report_{safe_name}_{timestamp}.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    # Also save raw outputs summary if enabled
    if save_raw and raw_output_dir:
        print(f"\nRaw outputs saved to: {raw_output_dir}")

    return {"model": model_name, "report_path": str(report_path), "report": report}
