from __future__ import annotations

import argparse
import json
from pathlib import Path

from banglabench.evaluator import evaluate


def load_environment(env_file: str) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        # Falls back to default .env lookup behavior if present in cwd/parents.
        load_dotenv(override=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BanglaBayanno benchmark evaluation")
    parser.add_argument("--model", required=True, help="Ollama model name, for example ollama/qwen2.5vl:7b")
    parser.add_argument("--data_dir", default="data", help="Dataset directory containing qa.json and images/")
    parser.add_argument("--results_dir", default="results", help="Directory for prediction and metrics files")
    parser.add_argument("--sample", type=int, default=None, help="Optional sample limit")
    parser.add_argument("--load_in_4bit", action="store_true", help="Legacy flag, ignored in Ollama-only mode")
    parser.add_argument("--env_file", default=".env", help="Path to .env file for runtime config (for example OLLAMA_HOST)")
    parser.add_argument("--request_delay", type=float, default=0.0, help="Delay in seconds between API requests")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum retries for retryable API errors")
    parser.add_argument("--retry_backoff", type=float, default=2.0, help="Base backoff in seconds for retries")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    load_environment(args.env_file)
    result = evaluate(
        model_name=args.model,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        sample=args.sample,
        load_in_4bit=args.load_in_4bit,
        request_delay=args.request_delay,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
    )
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
