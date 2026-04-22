from __future__ import annotations

import argparse
import json

from app.preprocess import prepare_segment_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-click preprocessing for Bangla-Bayanno dataset.")
    parser.add_argument(
        "--data_dir",
        default="dataset/Bangla-Bayanno-full",
        help="Directory containing qa.json and images/",
    )
    parser.add_argument(
        "--output_dir",
        default="preprocessed_dataset",
        help="Directory where preprocessed files will be written",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit for quick checks",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = prepare_segment_files(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
