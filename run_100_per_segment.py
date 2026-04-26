"""Run 100 samples from each segment (polar, numeric, descriptive).

- Saves incrementally to results/run_100_*.jsonl after every sample.
- Continues on any failure (records the error and moves on).
- Prints live progress and final summary.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.data import load_dataset_from_file
from app.metrics import compare_answers
from app.models import build_model
from app.preprocess import prepare_segment_files


def _safe_stem(name: str) -> str:
    import re
    safe = re.sub(r"[\\/:*?\"<>|]+", "_", name.strip())
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 100 samples per segment with incremental save")
    parser.add_argument("--model", required=True, help="Ollama model name, e.g. ollama/qwen2.5vl:latest")
    parser.add_argument("--data_dir", default="dataset", help="Dataset directory")
    parser.add_argument("--preprocessed_dir", default="preprocessed_dataset", help="Preprocessed directory")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--per_segment", type=int, default=100, help="Samples per segment (default 100)")
    parser.add_argument("--ollama_host", default=None, help="Ollama host override")
    parser.add_argument("--skip_preprocess", action="store_true", help="Skip preprocessing")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds between requests")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir = Path(args.preprocessed_dir)

    host = args.ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    print(f"Using Ollama host: {host}")

    # Preprocess if needed
    if not args.skip_preprocess:
        prepare_segment_files(data_dir=data_dir, output_dir=preprocessed_dir, limit=None)

    # Build model once
    model = build_model(args.model, host=host)
    safe_name = _safe_stem(args.model)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Output files
    out_jsonl = results_dir / f"run_100_{safe_name}_{timestamp}.jsonl"
    out_summary = results_dir / f"run_100_{safe_name}_{timestamp}_summary.json"

    images_dir = data_dir / "images"
    all_results: list[dict[str, Any]] = []

    segments = ("polar", "numeric", "descriptive")

    for segment in segments:
        qa_file = preprocessed_dir / f"qa_{segment}.json"
        if not qa_file.exists():
            print(f"\n[SKIP] {segment}: file not found ({qa_file})")
            continue

        samples = load_dataset_from_file(qa_file, images_dir=images_dir, limit=args.per_segment)
        if not samples:
            print(f"\n[SKIP] {segment}: no samples loaded")
            continue

        total_seg = len(samples)
        print(f"\n=== {segment.upper()} ({total_seg} samples) ===")

        iterator = tqdm(samples, desc=segment) if HAS_TQDM else samples

        for i, sample_row in enumerate(iterator):
            start_time = time.time()
            record: dict[str, Any] = {
                "qa_id": sample_row.sample_id,
                "type": segment,
                "question": sample_row.question,
                "ground_truth": sample_row.answer,
                "predicted": None,
                "confidence": None,
                "correct": False,
                "error": None,
                "error_trace": None,
                "latency_sec": 0.0,
                "image_path": str(sample_row.image_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            try:
                response = model.generate(sample_row)
                predicted = response.answer.strip()
                confidence = _confidence_to_unit(response.confidence)
                correct = compare_answers(
                    predicted=predicted,
                    ground_truth=sample_row.answer,
                    segment=segment,
                    question=sample_row.question,
                )

                record["predicted"] = predicted
                record["confidence"] = confidence
                record["correct"] = correct
                record["raw_output"] = response.raw_text if hasattr(response, "raw_text") else None

            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["error_trace"] = traceback.format_exc()
                # Still save and continue

            record["latency_sec"] = round(time.time() - start_time, 3)

            # Append immediately to JSONL
            with out_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            all_results.append(record)

            # Live stats
            seg_done = i + 1
            seg_ok = sum(1 for r in all_results if r["type"] == segment and r["error"] is None)
            seg_err = seg_done - seg_ok
            seg_correct = sum(
                1 for r in all_results if r["type"] == segment and r.get("correct") is True
            )
            if not HAS_TQDM:
                print(
                    f"  [{segment}] {seg_done}/{total_seg} | "
                    f"ok={seg_ok} err={seg_err} correct={seg_correct} "
                    f"lat={record['latency_sec']:.1f}s"
                )
            elif hasattr(iterator, "set_postfix"):
                iterator.set_postfix(
                    ok=seg_ok, err=seg_err, acc=f"{seg_correct / seg_done:.1%}"
                )

            if args.delay > 0:
                time.sleep(args.delay)

    # Summary
    total = len(all_results)
    errors = sum(1 for r in all_results if r["error"] is not None)
    correct = sum(1 for r in all_results if r.get("correct") is True)
    attempted = total - errors

    summary = {
        "model": args.model,
        "host": host,
        "per_segment_requested": args.per_segment,
        "timestamp": timestamp,
        "total_samples": total,
        "errors": errors,
        "attempted": attempted,
        "correct": correct,
        "accuracy": round(correct / attempted, 4) if attempted else 0.0,
        "segment_breakdown": {},
        "output_jsonl": str(out_jsonl),
    }

    for segment in segments:
        seg_results = [r for r in all_results if r["type"] == segment]
        if not seg_results:
            continue
        seg_errors = sum(1 for r in seg_results if r["error"] is not None)
        seg_attempted = len(seg_results) - seg_errors
        seg_correct = sum(1 for r in seg_results if r.get("correct") is True)
        summary["segment_breakdown"][segment] = {
            "total": len(seg_results),
            "errors": seg_errors,
            "attempted": seg_attempted,
            "correct": seg_correct,
            "accuracy": round(seg_correct / seg_attempted, 4) if seg_attempted else 0.0,
        }

    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== DONE ===")
    print(f"Total: {total} | Errors: {errors} | Correct: {correct}/{attempted}")
    print(f"Accuracy: {summary['accuracy']:.2%}")
    for seg, stats in summary["segment_breakdown"].items():
        print(f"  [{seg}] n={stats['total']} err={stats['errors']} acc={stats['accuracy']:.2%}")
    print(f"\nFiles saved:")
    print(f"  JSONL : {out_jsonl}")
    print(f"  Summary: {out_summary}")


if __name__ == "__main__":
    main()
