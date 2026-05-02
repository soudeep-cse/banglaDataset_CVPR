"""Run N samples per segment (polar, numeric, descriptive) with balanced sampling.

- Balanced: uses min(per_segment, smallest_segment_size) for all segments
- Saves incrementally to results/run_100_*.jsonl after every sample
- Continues on any failure (records the error and moves on)
- Full metrics in summary.json
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
from app.metrics import compare_answers, compute_metrics
from app.models import build_model
from app.preprocess import prepare_segment_files


def _safe_stem(name: str) -> str:
    import re
    safe = re.sub(r"[\\/:*?\"<>|]+", "_", name.strip())
    safe = re.sub(r"\s+", "_", safe)
    safe = re.sub(r"_+", "_", safe)
    safe = safe.strip("._")
    return safe or "model"


def _write_realtime_summary(
    out_summary: Path,
    all_results: list[dict[str, Any]],
    args: argparse.Namespace,
    host: str,
    timestamp: str,
    balanced_n: int,
    segments: tuple,
) -> None:
    """Write summary JSON in real-time after each sample."""
    from app.metrics import compute_metrics
    
    overall_metrics = compute_metrics(all_results)
    segment_wise: dict[str, Any] = {}
    for segment in segments:
        seg_results = [r for r in all_results if r["answer_type"] == segment]
        if seg_results:
            segment_wise[segment] = compute_metrics(seg_results)
    
    summary = {
        "model": args.model,
        "host": host,
        "per_segment_requested": args.per_segment,
        "balanced_samples_per_segment": balanced_n,
        "timestamp": timestamp,
        "total_processed_so_far": len(all_results),
        "overall": overall_metrics,
        "segment_wise": segment_wise,
        "output_jsonl": str(out_summary.parent / f"run_100_{_safe_stem(args.model)}_{timestamp}.jsonl"),
        "status": "in_progress" if overall_metrics["total_samples"] < balanced_n * 3 else "completed",
    }
    
    # Write atomically to avoid corruption
    tmp_path = out_summary.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    tmp_path.replace(out_summary)


def _confidence_to_unit(value: float | None) -> float:
    if value is None:
        return 0.5
    confidence = float(value)
    if confidence > 1.0:
        confidence /= 100.0
    return max(0.0, min(1.0, confidence))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run N samples per segment with balanced sampling")
    parser.add_argument("--model", required=True, help="Ollama model name, e.g. ollama/qwen2.5vl:latest")
    parser.add_argument("--data_dir", default="dataset", help="Dataset directory")
    parser.add_argument("--preprocessed_dir", default="preprocessed_dataset", help="Preprocessed directory")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--per_segment", type=int, default=None, help="Max samples per segment (default: use all, balanced to min available)")
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
    
    # Print available models from model registry
    print("\nConfigured models: llava:7b, bakllava:7b, moondream:1.8b, qwen2.5vl:latest, llama3.2-vision:latest")
    print(f"Using model: {args.model}")
    print("Real-time saving: JSONL + Summary JSON updated after every sample\n")

    if not args.skip_preprocess:
        prepare_segment_files(data_dir=data_dir, output_dir=preprocessed_dir, limit=None)

    model = build_model(args.model, host=host)
    safe_name = _safe_stem(args.model)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    out_jsonl = results_dir / f"run_100_{safe_name}_{timestamp}.jsonl"
    out_summary = results_dir / f"run_100_{safe_name}_{timestamp}_summary.json"

    images_dir = data_dir / "images"
    all_results: list[dict[str, Any]] = []
    segments = ("polar", "numeric", "descriptive")

    # Load all segments first to determine balanced size
    segment_samples: dict[str, list] = {}
    for segment in segments:
        qa_file = preprocessed_dir / f"qa_{segment}.json"
        if not qa_file.exists():
            print(f"\n[SKIP] {segment}: file not found ({qa_file})")
            continue
        loaded = load_dataset_from_file(qa_file, images_dir=images_dir)
        if loaded:
            segment_samples[segment] = loaded
        else:
            print(f"\n[SKIP] {segment}: no samples loaded")

    if not segment_samples:
        print("No segments found. Exiting.")
        return

    # Balanced: min of requested per_segment and smallest segment size
    min_available = min(len(s) for s in segment_samples.values())
    balanced_n = min(args.per_segment, min_available) if args.per_segment else min_available
    print(f"\nBalanced sampling: {balanced_n} per segment")
    print(f"  Requested: {args.per_segment} | Min available: {min_available}")
    for seg, samps in segment_samples.items():
        print(f"  {seg}: {len(samps)} available → using {balanced_n}")

    for segment, all_samples in segment_samples.items():
        samples = all_samples[:balanced_n]
        total_seg = len(samples)
        print(f"\n=== {segment.upper()} ({total_seg} samples) ===")

        iterator = tqdm(samples, desc=segment) if HAS_TQDM else samples

        for i, sample_row in enumerate(iterator):
            start_time = time.time()
            record: dict[str, Any] = {
                "qa_id": sample_row.sample_id,
                "image_file": Path(sample_row.image_path).name,
                "answer_type": segment,
                "question_bn": sample_row.question,
                "ground_truth": sample_row.answer,
                "predicted_answer": None,
                "confidence": None,
                "correct": False,
                "parse_success": False,
                "error": None,
                "latency_sec": 0.0,
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
                record["predicted_answer"] = predicted
                record["confidence"] = confidence
                record["correct"] = correct
                record["parse_success"] = True
                record["raw_output"] = response.raw_text if hasattr(response, "raw_text") else None

            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["parse_success"] = False

            record["latency_sec"] = round(time.time() - start_time, 3)

            with out_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            all_results.append(record)
            
            # Real-time summary update (every sample for live monitoring)
            _write_realtime_summary(out_summary, all_results, args, host, timestamp, balanced_n, segments)

            seg_done = i + 1
            seg_ok = sum(1 for r in all_results if r["answer_type"] == segment and r["parse_success"])
            seg_err = seg_done - seg_ok
            seg_correct = sum(1 for r in all_results if r["answer_type"] == segment and r.get("correct") is True)
            if not HAS_TQDM:
                print(f"  [{segment}] {seg_done}/{total_seg} | ok={seg_ok} err={seg_err} correct={seg_correct} lat={record['latency_sec']:.1f}s")
            elif hasattr(iterator, "set_postfix"):
                iterator.set_postfix(ok=seg_ok, err=seg_err, acc=f"{seg_correct / seg_done:.1%}")

            if args.delay > 0:
                time.sleep(args.delay)

    # Full metrics in summary
    overall_metrics = compute_metrics(all_results)
    segment_wise: dict[str, Any] = {}
    for segment in segments:
        seg_results = [r for r in all_results if r["answer_type"] == segment]
        if seg_results:
            segment_wise[segment] = compute_metrics(seg_results)

    summary = {
        "model": args.model,
        "host": host,
        "per_segment_requested": args.per_segment,
        "balanced_samples_per_segment": balanced_n,
        "timestamp": timestamp,
        "overall": overall_metrics,
        "segment_wise": segment_wise,
        "output_jsonl": str(out_jsonl),
    }

    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== DONE ===")
    print(f"Balanced N per segment : {balanced_n}")
    print(f"Total                  : {overall_metrics['total_samples']}")
    print(f"Attempted              : {overall_metrics['attempted']}")
    print(f"Errors                 : {overall_metrics['errors']}")
    print(f"Accuracy               : {overall_metrics['accuracy']:.4f}")
    print(f"Mean Confidence        : {overall_metrics['mean_confidence']:.4f}")
    print(f"Overconfidence Gap     : {overall_metrics['overconfidence_gap']:.4f}")
    print(f"ECE                    : {overall_metrics['ece']:.4f}")
    print(f"High-Conf Error Rate   : {overall_metrics['high_conf_error_rate']:.4f}")
    print(f"High-Conf Wrong Count  : {overall_metrics['high_conf_wrong_count']}")
    print("\nSegment-wise:")
    for seg, metrics in segment_wise.items():
        print(f"  [{seg}] n={metrics['total_samples']} acc={metrics['accuracy']:.4f} gap={metrics['overconfidence_gap']:.4f} ece={metrics['ece']:.4f}")
    print(f"\nFiles saved:")
    print(f"  JSONL  : {out_jsonl}")
    print(f"  Summary: {out_summary}")


if __name__ == "__main__":
    main()
