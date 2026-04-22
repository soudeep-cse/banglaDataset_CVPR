"""BanglaVerse dataset evaluator for VQA, CSU, and Captions tasks."""
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

from .data import load_banglaverse_dataset
from .mcq_utils import (
    build_caption_prompt,
    build_mcq_prompt,
    extract_caption,
    extract_mcq_answer,
    is_mcq_correct,
)
from .metrics import compute_caption_metrics, compute_metrics
from .models import build_model
from .types import BanglaVerseSample, MCQResponse, ModelResponse


def _safe_output_stem(model_name: str) -> str:
    """Normalize model names into cross-platform safe filenames."""
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


class BanglaVerseEvaluator:
    """Evaluator for BanglaVerse dataset across VQA, CSU, and Captions tasks."""

    def __init__(
        self,
        model_name: str,
        data_dir: str | Path = "banglaverse_processing/output",
        ollama_host: str | None = None,
        load_in_4bit: bool = False,
    ):
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.ollama_host = ollama_host
        self.load_in_4bit = load_in_4bit
        self.model = build_model(model_name, load_in_4bit=load_in_4bit, host=ollama_host)

    def _generate_with_retry(
        self,
        sample: BanglaVerseSample,
        prompt: str,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> ModelResponse:
        """Generate response with retry logic for transient errors."""
        from .types import QuestionSample

        attempt = 0
        while True:
            try:
                # Create a temporary QuestionSample for model interface
                temp_sample = QuestionSample(
                    sample_id=sample.sample_id,
                    question=prompt,
                    answer=sample.answer,
                    image_path=sample.image_path,
                    metadata=sample.metadata,
                )
                return self.model.generate(temp_sample)
            except Exception as exc:
                retryable = _is_retryable_error(exc)
                if attempt >= max_retries or not retryable:
                    raise
                wait_seconds = retry_backoff * (2 ** attempt)
                time.sleep(wait_seconds)
                attempt += 1

    def evaluate_mcq(
        self,
        sample: BanglaVerseSample,
        request_delay: float = 0.0,
    ) -> dict[str, Any]:
        """Evaluate a single MCQ sample (VQA or CSU)."""
        prompt = build_mcq_prompt(sample.question, sample.options)
        response = self._generate_with_retry(sample, prompt)

        mcq_response = extract_mcq_answer(response.raw_text, sample.options)
        correct = is_mcq_correct(mcq_response, sample)

        if request_delay > 0:
            time.sleep(request_delay)

        return {
            "sample_id": sample.sample_id,
            "task": sample.task,
            "dialect": sample.dialect,
            "domain": sample.domain,
            "ground_truth": sample.answer,
            "predicted": mcq_response.selected_option,
            "predicted_index": mcq_response.selected_index,
            "correct_index": sample.answer_index,
            "confidence": _confidence_to_unit(mcq_response.confidence),
            "correct": correct,
            "raw_text": response.raw_text,
            "image_path": str(sample.image_path),
        }

    def evaluate_caption(
        self,
        sample: BanglaVerseSample,
        request_delay: float = 0.0,
    ) -> dict[str, Any]:
        """Evaluate a single caption sample."""
        prompt = build_caption_prompt()
        response = self._generate_with_retry(sample, prompt)

        caption, confidence = extract_caption(response.raw_text)

        if request_delay > 0:
            time.sleep(request_delay)

        return {
            "sample_id": sample.sample_id,
            "task": sample.task,
            "dialect": sample.dialect,
            "domain": sample.domain,
            "ground_truth": sample.caption,
            "predicted": caption,
            "confidence": _confidence_to_unit(confidence),
            "raw_text": response.raw_text,
            "image_path": str(sample.image_path),
        }

    def evaluate_task(
        self,
        samples: list[BanglaVerseSample],
        request_delay: float = 0.0,
        save_raw_path: Path | None = None,
        task_name: str = "",
    ) -> list[dict[str, Any]]:
        """Evaluate all samples for a specific task.

        Args:
            samples: List of samples to evaluate
            request_delay: Seconds between API calls
            save_raw_path: If provided, save raw outputs to this directory
            task_name: Task name for progress bar
        """
        results = []
        raw_outputs: list[dict[str, Any]] = []

        desc = f"Evaluating {task_name}" if task_name else "Evaluating"
        iterator = tqdm(samples, desc=desc, disable=not HAS_TQDM)

        for sample in iterator:
            if sample.task in {"vqa", "csu"}:
                result = self.evaluate_mcq(sample, request_delay)
            elif sample.task == "captions":
                result = self.evaluate_caption(sample, request_delay)
            else:
                continue

            results.append(result)

            # Save raw output
            if save_raw_path:
                raw_output = {
                    "sample_id": sample.sample_id,
                    "task": sample.task,
                    "dialect": sample.dialect,
                    "domain": sample.domain,
                    "image_path": str(sample.image_path),
                    "question": sample.question if sample.task in {"vqa", "csu"} else None,
                    "options": sample.options if sample.task in {"vqa", "csu"} else None,
                    "caption": sample.caption if sample.task == "captions" else None,
                    "ground_truth": sample.answer,
                    "predicted": result.get("predicted"),
                    "correct": result.get("correct"),
                    "confidence": result.get("confidence"),
                    "raw_model_output": result.get("raw_text"),
                }
                raw_outputs.append(raw_output)

                # Write incrementally to avoid memory issues
                raw_file = save_raw_path / f"raw_outputs_{task_name}.jsonl"
                with raw_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(raw_output, ensure_ascii=False) + "\n")

            # Update progress bar postfix
            if HAS_TQDM:
                correct_count = sum(1 for r in results if r.get("correct"))
                accuracy = correct_count / len(results) if results else 0
                iterator.set_postfix({"acc": f"{accuracy:.2%}", "correct": f"{correct_count}/{len(results)}"})

        return results


def evaluate_banglaverse(
    model_name: str,
    data_dir: str | Path = "banglaverse_processing/output",
    tasks: list[str] | None = None,
    results_dir: str | Path = "results",
    sample_limit: int | None = None,
    ollama_host: str | None = None,
    load_in_4bit: bool = False,
    oer_threshold: float = 0.7,
    ece_bins: int = 10,
    request_delay: float = 0.0,
    max_retries: int = 3,
    retry_backoff: float = 2.0,
    save_raw: bool = False,
) -> dict[str, Any]:
    """Run complete BanglaVerse evaluation across tasks.

    Args:
        model_name: Ollama model name (e.g., "ollama/qwen2.5vl:7b")
        data_dir: Directory with JSONL files
        tasks: List of tasks to evaluate ("vqa", "csu", "captions"). Default: all.
        results_dir: Directory for output report
        sample_limit: Per-task sample limit
        ollama_host: Ollama API host URL
        load_in_4bit: Legacy flag (ignored for Ollama)
        oer_threshold: Confidence threshold for OER metric
        ece_bins: Number of bins for ECE calculation
        request_delay: Seconds to wait between API calls
        max_retries: Max retries for transient errors
        retry_backoff: Base backoff in seconds
        save_raw: If True, save raw model outputs for each sample

    Returns:
        Dictionary with report path and metrics
    """
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup raw output directory
    raw_output_dir: Path | None = None
    if save_raw:
        raw_output_dir = results_dir / "raw_outputs"
        raw_output_dir.mkdir(parents=True, exist_ok=True)
        # Clean old files
        for f in raw_output_dir.glob("raw_outputs_*.jsonl"):
            f.unlink()

    if tasks is None:
        tasks = ["vqa", "csu", "captions"]

    # Initialize evaluator
    evaluator = BanglaVerseEvaluator(
        model_name=model_name,
        data_dir=data_dir,
        ollama_host=ollama_host,
        load_in_4bit=load_in_4bit,
    )

    # Load and evaluate each task
    all_results: list[dict[str, Any]] = []
    task_results: dict[str, list[dict[str, Any]]] = {}

    for task in tasks:
        jsonl_file = data_dir / f"{task}_records.jsonl"
        if not jsonl_file.exists():
            print(f"Warning: {jsonl_file} not found, skipping {task}")
            continue

        print(f"\nLoading {task} samples from {jsonl_file}...")
        from .data import load_banglaverse_from_jsonl
        samples = load_banglaverse_from_jsonl(jsonl_file, limit=sample_limit, task_filter=task)
        print(f"Loaded {len(samples)} samples for {task}")

        if not samples:
            continue

        print(f"Evaluating {task}...")
        results = evaluator.evaluate_task(
            samples,
            request_delay=request_delay,
            save_raw_path=raw_output_dir,
            task_name=task,
        )
        task_results[task] = results
        all_results.extend(results)

        print(f"  Completed: {len(results)} samples")
        if raw_output_dir:
            print(f"  Raw outputs saved to: {raw_output_dir / f'raw_outputs_{task}.jsonl'}")

    # Compute metrics
    print("\nComputing metrics...")

    # Overall metrics (MCQ tasks use accuracy, captions use similarity)
    mcq_results = [r for r in all_results if r["task"] in {"vqa", "csu"}]
    caption_results = [r for r in all_results if r["task"] == "captions"]

    if mcq_results:
        overall_metrics = compute_metrics(mcq_results, ece_bins=ece_bins, oer_threshold=oer_threshold)
    else:
        overall_metrics = {"total": 0, "accuracy": 0.0, "oer": 0.0, "ece": 0.0, "brier": 0.0, "mcw": 0.0}

    if caption_results:
        caption_metrics = compute_caption_metrics(caption_results, ece_bins=ece_bins, oer_threshold=oer_threshold)
    else:
        caption_metrics = {"total": 0, "accuracy": 0.0, "oer": 0.0, "ece": 0.0, "brier": 0.0, "mcw": 0.0}

    # Task-wise metrics
    task_wise: dict[str, dict[str, Any]] = {}
    for task, results in task_results.items():
        if task in {"vqa", "csu"}:
            task_wise[task] = compute_metrics(results, ece_bins=ece_bins, oer_threshold=oer_threshold)
        else:
            task_wise[task] = compute_caption_metrics(results, ece_bins=ece_bins, oer_threshold=oer_threshold)
        task_wise[task]["n"] = len(results)

    # Dialect-wise breakdown
    dialect_wise: dict[str, dict[str, Any]] = {}
    dialects = set(r["dialect"] for r in all_results)
    for dialect in sorted(dialects):
        dialect_results = [r for r in all_results if r["dialect"] == dialect]
        dialect_mcq = [r for r in dialect_results if r["task"] in {"vqa", "csu"}]
        dialect_caption = [r for r in dialect_results if r["task"] == "captions"]

        metrics = {}
        if dialect_mcq:
            metrics.update(compute_metrics(dialect_mcq, ece_bins=ece_bins, oer_threshold=oer_threshold))
        if dialect_caption:
            caption_m = compute_caption_metrics(dialect_caption, ece_bins=ece_bins, oer_threshold=oer_threshold)
            metrics["avg_token_overlap"] = caption_m.get("avg_token_overlap", 0.0)
        metrics["n"] = len(dialect_results)
        dialect_wise[dialect] = metrics

    # Build report
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "metadata": {
            "model": model_name,
            "timestamp": timestamp,
            "oer_threshold": oer_threshold,
            "ece_bins": ece_bins,
            "data_dir": str(data_dir),
            "tasks": tasks,
            "total_results": len(all_results),
        },
        "overall": {
            **overall_metrics,
            "caption_token_overlap": caption_metrics.get("avg_token_overlap", 0.0),
            "caption_avg_length_ratio": caption_metrics.get("avg_length_ratio", 0.0),
        },
        "task_wise": task_wise,
        "dialect_wise": dialect_wise,
        "results": all_results,
    }

    # Save report
    safe_name = _safe_output_stem(model_name)
    report_path = results_dir / f"banglaverse_report_{safe_name}_{timestamp}.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    return {
        "model": model_name,
        "report_path": str(report_path),
        "report": report,
    }
