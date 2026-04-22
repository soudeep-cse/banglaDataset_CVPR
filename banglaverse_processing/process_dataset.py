from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TASK_FOLDERS = {
    "VQA": "vqa",
    "CSU": "csu",
    "Captions": "captions",
}

DIALECT_DIR = "data/dialectual_data"
IMAGE_DIR = "data/images"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_posix(path: Path) -> str:
    return path.as_posix()


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    return cleaned.rstrip(",.;:!?।")


def _answer_index(answer: str, options: list[str]) -> int | None:
    if not options:
        return None

    raw_answer = str(answer or "").strip()
    for idx, option in enumerate(options):
        if raw_answer == str(option or "").strip():
            return idx

    normalized_answer = _normalize_text(raw_answer)
    for idx, option in enumerate(options):
        if normalized_answer == _normalize_text(option):
            return idx

    return None


def _build_image_index(images_root: Path) -> dict[str, str]:
    image_index: dict[str, str] = {}
    if not images_root.exists():
        return image_index

    for category_dir in images_root.iterdir():
        images_dir = category_dir / "images"
        if not images_dir.is_dir():
            continue
        for image_path in images_dir.iterdir():
            if not image_path.is_file():
                continue
            image_index[image_path.stem] = _to_posix(image_path)

    return image_index


def _infer_category(file_path: Path, task_name: str, dialect: str) -> str:
    suffix = f"_{task_name}_{dialect}.json"
    name = file_path.name
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return file_path.stem


def _parse_rows(
    file_path: Path,
    dialect: str,
    task_name: str,
    task_label: str,
    image_index: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    payload = _load_json(file_path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list JSON in {file_path}")

    category = _infer_category(file_path, task_name, dialect)
    records: list[dict[str, Any]] = []
    stats: dict[str, int] = {
        "rows": 0,
        "missing_image_path": 0,
        "missing_answer_index": 0,
    }

    for row_idx, row in enumerate(payload):
        if not isinstance(row, dict):
            continue

        image_id = str(row.get("image_id", "")).strip()
        image_path = image_index.get(image_id)

        record_id = f"{dialect}:{task_label}:{category}:{image_id or row_idx}"
        base_record: dict[str, Any] = {
            "record_id": record_id,
            "dialect": dialect,
            "task": task_label,
            "domain": category,
            "image_id": image_id,
            "image_path": image_path,
            "source_file": _to_posix(file_path),
        }

        if image_path is None:
            stats["missing_image_path"] += 1

        if task_label in {"vqa", "csu"}:
            options = [str(item) for item in row.get("options", []) if isinstance(item, str)]
            original_options = [
                str(item) for item in row.get("original_options", []) if isinstance(item, str)
            ]
            answer = str(row.get("answer", "")).strip()
            answer_index = _answer_index(answer, options)

            if answer_index is None:
                stats["missing_answer_index"] += 1

            base_record.update(
                {
                    "question": str(row.get("question", "")).strip(),
                    "original_question": str(row.get("original_question", "")).strip(),
                    "options": options,
                    "original_options": original_options,
                    "answer": answer,
                    "original_answer": str(row.get("original_answer", "")).strip(),
                    "answer_index": answer_index,
                    "correct_option": options[answer_index] if answer_index is not None else None,
                }
            )
        else:
            base_record.update(
                {
                    "caption": str(row.get("caption", "")).strip(),
                    "original_caption": str(row.get("original_caption", "")).strip(),
                }
            )

        records.append(base_record)
        stats["rows"] += 1

    return records, stats


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_dataset(root: Path, output_dir: Path) -> dict[str, Any]:
    dialect_root = root / DIALECT_DIR
    image_root = root / IMAGE_DIR

    if not dialect_root.exists():
        raise FileNotFoundError(f"Could not find dialectual_data directory: {dialect_root}")

    image_index = _build_image_index(image_root)

    all_records: list[dict[str, Any]] = []
    per_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_dialect: dict[str, list[dict[str, Any]]] = defaultdict(list)

    file_level_stats: list[dict[str, Any]] = []
    domain_counts: Counter[tuple[str, str]] = Counter()

    dialect_dirs = sorted([path for path in dialect_root.iterdir() if path.is_dir()])

    for dialect_dir in dialect_dirs:
        dialect = dialect_dir.name
        results_dir = dialect_dir / "Results"
        if not results_dir.is_dir():
            continue

        for folder_name, task_label in TASK_FOLDERS.items():
            task_dir = results_dir / folder_name
            if not task_dir.is_dir():
                continue

            for json_file in sorted(task_dir.glob("*.json")):
                task_name = task_label
                records, stats = _parse_rows(
                    file_path=json_file,
                    dialect=dialect,
                    task_name=task_name,
                    task_label=task_label,
                    image_index=image_index,
                )

                all_records.extend(records)
                per_task[task_label].extend(records)
                per_dialect[dialect].extend(records)

                for record in records:
                    domain_counts[(record["dialect"], record["domain"])] += 1

                file_level_stats.append(
                    {
                        "dialect": dialect,
                        "task": task_label,
                        "source_file": _to_posix(json_file),
                        **stats,
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)

    missing_image_rows = [row for row in all_records if not row.get("image_path")]
    missing_answer_index_rows = [
        row
        for row in all_records
        if row.get("task") in {"vqa", "csu"} and row.get("answer_index") is None
    ]

    _write_jsonl(output_dir / "all_records.jsonl", all_records)
    for task_name, rows in per_task.items():
        _write_jsonl(output_dir / f"{task_name}_records.jsonl", rows)
    for dialect, rows in per_dialect.items():
        _write_jsonl(output_dir / f"{dialect}_records.jsonl", rows)
    _write_jsonl(output_dir / "records_missing_image_path.jsonl", missing_image_rows)
    _write_jsonl(output_dir / "records_missing_answer_index.jsonl", missing_answer_index_rows)

    domain_count_output = [
        {"dialect": dialect, "domain": domain, "count": count}
        for (dialect, domain), count in sorted(domain_counts.items())
    ]

    summary = {
        "root": _to_posix(root),
        "output_dir": _to_posix(output_dir),
        "total_records": len(all_records),
        "missing_image_path_records": len(missing_image_rows),
        "missing_answer_index_records": len(missing_answer_index_rows),
        "task_counts": {task: len(rows) for task, rows in sorted(per_task.items())},
        "dialect_counts": {dialect: len(rows) for dialect, rows in sorted(per_dialect.items())},
        "image_index_size": len(image_index),
        "files_processed": len(file_level_stats),
        "file_level_stats": file_level_stats,
        "dialect_domain_counts": domain_count_output,
    }

    _write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process BanglaVerse dialectual_data into reusable JSONL files.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("BanglaVerse"),
        help="Path to BanglaVerse dataset root (default: BanglaVerse)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("banglaverse_processing/output"),
        help="Output directory for processed files",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = process_dataset(root=args.root, output_dir=args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
