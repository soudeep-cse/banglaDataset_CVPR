from __future__ import annotations

import argparse
import json
from pathlib import Path


def infer_taxonomy(predictions_path: str | Path) -> dict[str, int]:
    with Path(predictions_path).open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    counts = {
        "correct": 0,
        "refusal": 0,
        "hallucination": 0,
        "other_error": 0,
    }
    for row in rows:
        if row.get("correct"):
            counts["correct"] += 1
        elif row.get("refused"):
            counts["refusal"] += 1
        elif row.get("confidence") is not None and float(row.get("confidence")) >= 70 and not row.get("correct"):
            counts["hallucination"] += 1
        else:
            counts["other_error"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple failure taxonomy analysis")
    parser.add_argument("predictions_path", help="Path to *_predictions.json")
    args = parser.parse_args()
    print(json.dumps(infer_taxonomy(args.predictions_path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
