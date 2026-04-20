from __future__ import annotations

import argparse

from banglabench.visuals import generate_all_figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate benchmark figures")
    parser.add_argument("--results_dir", default="results", help="Directory containing *_metrics.json and *_predictions.json")
    parser.add_argument("--output", default="figures", help="Directory for generated figures")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    paths = generate_all_figures(args.results_dir, args.output)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
