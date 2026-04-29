from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.evaluator import evaluate


def load_environment(env_file: str) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True, verbose=True)
    else:
        # Falls back to default .env lookup behavior if present in cwd/parents.
        load_dotenv(override=True, verbose=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BanglaBayanno benchmark evaluation")
    parser.add_argument("--model", required=True, help="Ollama model name, for example ollama/qwen2.5vl:7b")
    parser.add_argument("--data_dir", default="dataset", help="Dataset directory containing qa.json and images/")
    parser.add_argument("--results_dir", default="results", help="Directory for benchmark reports")
    parser.add_argument("--preprocessed_dir", default="preprocessed_dataset", help="Directory for preprocessed split JSON files")
    parser.add_argument("--skip_preprocess", action="store_true", help="Skip preprocessing and use existing split files")
    parser.add_argument("--sample", type=int, default=None, help="Optional sample limit")
    parser.add_argument("--load_in_4bit", action="store_true", help="Legacy flag, ignored in Ollama-only mode")
    parser.add_argument("--env_file", default=".env", help="Path to .env file for runtime config (for example OLLAMA_HOST)")
    parser.add_argument("--ollama_host", default=None, help="Override Ollama host, for example http://69.30.85.131:22054")
    parser.add_argument("--ece_bins", type=int, default=10, help="Number of bins used for ECE")
    parser.add_argument("--request_delay", type=float, default=0.0, help="Delay in seconds between API requests")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum retries for retryable API errors")
    parser.add_argument("--retry_backoff", type=float, default=2.0, help="Base backoff in seconds for retries")
    parser.add_argument("--save-raw", action="store_true", help="Save raw model outputs for each sample to results/raw_outputs/")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint file (JSONL path) - skips already processed samples")
    parser.add_argument("--validate", action="store_true", help="Run post-evaluation validation agent (qwen3.5:35b as coach) after benchmark")
    parser.add_argument("--judge_model", type=str, default="qwen3.5:35b", help="Ollama model to use as validation judge (default: qwen3.5:35b)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    load_environment(args.env_file)

    # Determine actual Ollama host
    actual_host = args.ollama_host or os.getenv("OLLAMA_HOST")
    if actual_host:
        print(f"Using Ollama host: {actual_host}")
    else:
        print("Using default Ollama host (localhost)")

    result = evaluate(
        model_name=args.model,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        preprocessed_dir=args.preprocessed_dir,
        skip_preprocess=args.skip_preprocess,
        sample=args.sample,
        load_in_4bit=args.load_in_4bit,
        ollama_host=actual_host,
        ece_bins=args.ece_bins,
        request_delay=args.request_delay,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        save_raw=args.save_raw,
        resume_from=args.resume_from,
        validate=args.validate,
        judge_model=args.judge_model,
    )
    report = result["report"]
    overall = report.get("overall", {})
    print("=== Overall ===")
    print(f"  total_samples   : {overall.get('total_samples', 0)}")
    print(f"  attempted       : {overall.get('attempted', 0)}")
    print(f"  errors          : {overall.get('errors', 0)}")
    print(f"  parse_success   : {overall.get('parse_success_rate', 0.0):.4f}")
    print(f"  accuracy        : {overall.get('accuracy', 0.0):.4f}")
    print(f"  error_rate      : {overall.get('error_rate', 0.0):.4f}")
    print(f"  mean_confidence : {overall.get('mean_confidence', 0.0):.4f}")
    print(f"  overconf_gap    : {overall.get('overconfidence_gap', 0.0):.4f}")
    print(f"  ece             : {overall.get('ece', 0.0):.4f}")
    print(f"  high_conf_err   : {overall.get('high_conf_error_rate', 0.0):.4f}")
    print(f"  high_conf_wrong : {overall.get('high_conf_wrong_count', 0)}")
    print("\n=== Segment-wise ===")
    for segment in ("polar", "numeric", "descriptive"):
        seg = report.get("segment_wise", {}).get(segment, {})
        print(f"\n  [{segment}] (n={seg.get('total_samples', 0)})")
        print(f"    accuracy        : {seg.get('accuracy', 0.0):.4f}")
        print(f"    error_rate      : {seg.get('error_rate', 0.0):.4f}")
        print(f"    mean_confidence : {seg.get('mean_confidence', 0.0):.4f}")
        print(f"    overconf_gap    : {seg.get('overconfidence_gap', 0.0):.4f}")
        print(f"    ece             : {seg.get('ece', 0.0):.4f}")
        print(f"    high_conf_err   : {seg.get('high_conf_error_rate', 0.0):.4f}")
        print(f"    high_conf_wrong : {seg.get('high_conf_wrong_count', 0)}")
    print(f"\nSaved: {result.get('report_path')}")
    print(json.dumps(report.get("metadata", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
