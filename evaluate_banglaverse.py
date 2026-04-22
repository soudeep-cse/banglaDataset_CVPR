"""CLI entry point for BanglaVerse dataset evaluation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.banglaverse_evaluator import evaluate_banglaverse


def load_environment(env_file: str) -> None:
    """Load environment variables from .env file."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True, verbose=False)
    else:
        # Try to find .env in parent directories
        load_dotenv(override=True, verbose=False)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Run BanglaVerse benchmark evaluation (VQA, CSU, Captions)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all tasks with default settings
  python evaluate_banglaverse.py --model ollama/qwen2.5vl:7b

  # Evaluate only VQA with 50 samples
  python evaluate_banglaverse.py --model ollama/llava --tasks vqa --sample 50

  # Evaluate with custom Ollama host
  python evaluate_banglaverse.py --model qwen2.5vl --ollama_host http://localhost:11434

  # Evaluate specific dialect (from all_records.jsonl with filter)
  python evaluate_banglaverse.py --model qwen2.5vl --tasks vqa csu

Supported tasks:
  - vqa: Visual Question Answering (MCQ format)
  - csu: Cultural Scene Understanding (MCQ format)
  - captions: Image Captioning

Supported models (Ollama aliases):
  - qwen2.5vl, llava, bakllava, moondream, minicpm-v, llama3.2-vision
        """,
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model name, e.g., ollama/qwen2.5vl:7b or just qwen2.5vl",
    )

    parser.add_argument(
        "--data_dir",
        default="banglaverse_processing/output",
        help="Directory containing BanglaVerse JSONL files (default: banglaverse_processing/output)",
    )

    parser.add_argument(
        "--results_dir",
        default="results",
        help="Directory for evaluation reports (default: results)",
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["vqa", "csu", "captions", "all"],
        default=["all"],
        help="Tasks to evaluate (default: all)",
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional per-task sample limit",
    )

    parser.add_argument(
        "--ollama_host",
        default=None,
        help="Override Ollama host, e.g., http://69.30.85.131:22054",
    )

    parser.add_argument(
        "--env_file",
        default=".env",
        help="Path to .env file for OLLAMA_HOST configuration",
    )

    parser.add_argument(
        "--oer_threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for OER metric (default: 0.7)",
    )

    parser.add_argument(
        "--ece_bins",
        type=int,
        default=10,
        help="Number of bins for ECE calculation (default: 10)",
    )

    parser.add_argument(
        "--request_delay",
        type=float,
        default=0.0,
        help="Delay in seconds between API requests (default: 0)",
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retries for transient API errors (default: 3)",
    )

    parser.add_argument(
        "--retry_backoff",
        type=float,
        default=2.0,
        help="Base backoff in seconds for retries (default: 2.0)",
    )

    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw model outputs for each sample to results/raw_outputs/",
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Load environment
    load_environment(args.env_file)

    # Debug: Show env file path and loaded OLLAMA_HOST
    env_path = Path(args.env_file)
    env_host = os.getenv("OLLAMA_HOST")

    # Fallback: read .env file directly if load_dotenv didn't work
    if not env_host and env_path.exists():
        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("OLLAMA_HOST="):
                    env_host = line[len("OLLAMA_HOST="):].strip()
                    break

    # Determine tasks
    if "all" in args.tasks:
        tasks = ["vqa", "csu", "captions"]
    else:
        tasks = args.tasks

    # Determine actual Ollama host (CLI arg > env var > default)
    if args.ollama_host:
        actual_host = args.ollama_host
    elif env_host:
        actual_host = env_host
    else:
        actual_host = "http://localhost:11434"

    print(f"\n{'='*60}")
    print("BanglaVerse Benchmark Evaluation")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Tasks: {tasks}")
    print(f"Data directory: {args.data_dir}")
    print(f"Sample limit: {args.sample or 'All samples'}")
    print(f"Ollama host: {actual_host}")
    print(f"{'='*60}\n")

    try:
        result = evaluate_banglaverse(
            model_name=args.model,
            data_dir=args.data_dir,
            tasks=tasks,
            results_dir=args.results_dir,
            sample_limit=args.sample,
            ollama_host=actual_host,
            oer_threshold=args.oer_threshold,
            ece_bins=args.ece_bins,
            request_delay=args.request_delay,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
            save_raw=args.save_raw,
        )

        report = result["report"]
        overall = report.get("overall", {})
        task_wise = report.get("task_wise", {})
        dialect_wise = report.get("dialect_wise", {})

        # Print summary
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")

        print("\n=== Overall Metrics ===")
        print(f"  Total samples: {overall.get('total', 0)}")
        if overall.get('accuracy', 0) > 0:
            print(f"  MCQ Accuracy: {overall.get('accuracy', 0.0):.4f}")
            print(f"  OER: {overall.get('oer', 0.0):.4f}")
            print(f"  ECE: {overall.get('ece', 0.0):.4f}")
            print(f"  Brier: {overall.get('brier', 0.0):.4f}")
            print(f"  MCW: {overall.get('mcw', 0.0):.4f}")
        if overall.get('caption_token_overlap', 0) > 0:
            print(f"  Caption Token Overlap: {overall.get('caption_token_overlap', 0.0):.4f}")
            print(f"  Caption Length Ratio: {overall.get('caption_avg_length_ratio', 0.0):.4f}")

        print("\n=== Task-wise Metrics ===")
        for task, metrics in sorted(task_wise.items()):
            n = metrics.get('n', 0)
            print(f"\n  [{task.upper()}] (n={n})")
            if task in {"vqa", "csu"}:
                print(f"    Accuracy: {metrics.get('accuracy', 0.0):.4f}")
                print(f"    OER: {metrics.get('oer', 0.0):.4f}")
                print(f"    ECE: {metrics.get('ece', 0.0):.4f}")
            else:  # captions
                print(f"    Token Overlap: {metrics.get('avg_token_overlap', 0.0):.4f}")
                print(f"    Accuracy (overlap>0.5): {metrics.get('accuracy', 0.0):.4f}")
            print(f"    Brier: {metrics.get('brier', 0.0):.4f}")
            print(f"    MCW: {metrics.get('mcw', 0.0):.4f}")

        print("\n=== Dialect-wise Metrics ===")
        for dialect, metrics in sorted(dialect_wise.items()):
            n = metrics.get('n', 0)
            acc = metrics.get('accuracy', 0.0)
            overlap = metrics.get('avg_token_overlap', 0.0)
            print(f"\n  [{dialect}] (n={n})")
            if acc > 0:
                print(f"    Accuracy: {acc:.4f}")
            if overlap > 0:
                print(f"    Token Overlap: {overlap:.4f}")

        print(f"\n{'='*60}")
        print(f"Report saved: {result.get('report_path')}")
        print(f"{'='*60}\n")

        # Print metadata
        print("Metadata:")
        print(json.dumps(report.get("metadata", {}), ensure_ascii=False, indent=2))

        return 0

    except FileNotFoundError as e:
        print(f"\nError: File not found - {e}")
        print("\nMake sure you have processed the BanglaVerse dataset:")
        print("  python banglaverse_processing/process_dataset.py")
        return 1

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
