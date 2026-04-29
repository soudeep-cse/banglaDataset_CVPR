from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import BanglaVerseSample, QuestionSample


QUESTION_KEYS = ("question_bn", "question", "query", "q", "text", "prompt")
ANSWER_KEYS = ("answer_bn", "answer", "gt_answer", "label", "target", "gold")
IMAGE_KEYS = (
    "image",
    "image_file",
    "image_path",
    "image_name",
    "file_name",
    "filename",
    "img",
    "path",
)
QUESTION_TYPE_KEYS = ("question_type", "qtype", "question_category", "q_category")
ANSWER_TYPE_KEYS = ("answer_type", "answer_category", "atype", "category", "type")
PAIR_KEYS = ("pair_id", "group_id", "logical_pair", "consistency_id")
ID_KEYS = ("id", "qa_id", "uid", "sample_id", "qid")


def _first_present(row: dict[str, Any], keys: tuple[str, ...], default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "samples", "items", "examples", "qa"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    raise ValueError("Unsupported QA JSON structure")


def _resolve_image_path(images_dir: Path, raw_value: Any) -> Path:
    if raw_value is None:
        raise ValueError("Missing image reference in dataset row")
    candidate = Path(str(raw_value))
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    if candidate.suffix:
        return (images_dir / candidate).resolve()
    for suffix in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        extended = (images_dir / f"{candidate}{suffix}").resolve()
        if extended.exists():
            return extended
    return (images_dir / candidate).resolve()


def _rows_to_samples(rows: list[dict[str, Any]], images_dir: Path, limit: int | None = None) -> list[QuestionSample]:
    if limit is not None:
        rows = rows[:limit]

    samples: list[QuestionSample] = []
    for index, row in enumerate(rows):
        sample_id = str(_first_present(row, ID_KEYS, f"sample-{index}"))
        question = str(_first_present(row, QUESTION_KEYS, "")).strip()
        answer = str(_first_present(row, ANSWER_KEYS, "")).strip()
        image_ref = _first_present(row, IMAGE_KEYS, None)
        if not question:
            continue  # skip rows with missing question
        if image_ref is None:
            raise ValueError(f"Missing image reference for sample {sample_id}")
        samples.append(
            QuestionSample(
                sample_id=sample_id,
                question=question,
                answer=answer,
                image_path=_resolve_image_path(images_dir, image_ref),
                question_type=str(_first_present(row, QUESTION_TYPE_KEYS, "unknown")),
                answer_type=str(_first_present(row, ANSWER_TYPE_KEYS, _first_present(row, QUESTION_TYPE_KEYS, "unknown"))),
                pair_id=str(_first_present(row, PAIR_KEYS, "")) or None,
                metadata=row,
            )
        )
    return samples


def load_dataset_from_file(qa_file: str | Path, images_dir: str | Path, limit: int | None = None) -> list[QuestionSample]:
    qa_file = Path(qa_file)
    payload = _read_json(qa_file)
    rows = _extract_rows(payload)
    return _rows_to_samples(rows, Path(images_dir), limit=limit)


def load_dataset(data_dir: str | Path, limit: int | None = None) -> list[QuestionSample]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        alternate = Path("dataset") if data_dir.name == "data" else Path("data")
        if alternate.exists():
            data_dir = alternate
    qa_path = data_dir / "qa.json"
    images_dir = data_dir / "images"
    payload = _read_json(qa_path)
    rows = _extract_rows(payload)
    return _rows_to_samples(rows, images_dir, limit=limit)


def _resolve_banglaverse_image_path(raw_value: Any) -> Path:
    """Resolve image path from BanglaVerse dataset."""
    if raw_value is None:
        raise ValueError("Missing image reference in BanglaVerse record")
    candidate = Path(str(raw_value))
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    # Try relative to project root
    project_root = Path.cwd()
    full_path = project_root / candidate
    if full_path.exists():
        return full_path.resolve()
    return candidate


def load_banglaverse_from_jsonl(
    jsonl_path: str | Path,
    limit: int | None = None,
    task_filter: str | None = None,
    dialect_filter: str | None = None,
) -> list[BanglaVerseSample]:
    """Load BanglaVerse samples from JSONL file.

    Args:
        jsonl_path: Path to JSONL file (e.g., vqa_records.jsonl)
        limit: Maximum number of samples to load
        task_filter: Filter by task type ("vqa", "csu", "captions")
        dialect_filter: Filter by dialect ("barishal", "chittagong", etc.)
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    samples: list[BanglaVerseSample] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            task = row.get("task", "").lower()
            dialect = row.get("dialect", "").lower()

            # Apply filters
            if task_filter and task != task_filter.lower():
                continue
            if dialect_filter and dialect != dialect_filter.lower():
                continue

            # Build sample
            sample_id = row.get("record_id", f"{task}-{len(samples)}")
            image_path = _resolve_banglaverse_image_path(row.get("image_path"))

            if task in {"vqa", "csu"}:
                # MCQ task
                options = row.get("options", [])
                answer = row.get("answer", "")
                answer_index = row.get("answer_index")
                sample = BanglaVerseSample(
                    sample_id=sample_id,
                    task=task,
                    dialect=dialect,
                    domain=row.get("domain", ""),
                    image_path=image_path,
                    question=row.get("question", ""),
                    options=options if isinstance(options, list) else [],
                    answer=answer,
                    answer_index=answer_index if answer_index is not None else None,
                    metadata=row,
                )
            elif task == "captions":
                # Caption task
                sample = BanglaVerseSample(
                    sample_id=sample_id,
                    task=task,
                    dialect=dialect,
                    domain=row.get("domain", ""),
                    image_path=image_path,
                    caption=row.get("caption", ""),
                    answer=row.get("caption", ""),  # Reference caption
                    metadata=row,
                )
            else:
                continue

            samples.append(sample)

            if limit is not None and len(samples) >= limit:
                break

    return samples


def load_banglaverse_dataset(
    data_dir: str | Path = "banglaverse_processing/output",
    tasks: list[str] | None = None,
    limit: int | None = None,
) -> list[BanglaVerseSample]:
    """Load BanglaVerse dataset from output directory.

    Args:
        data_dir: Directory containing JSONL files
        tasks: List of tasks to load ("vqa", "csu", "captions"). Default: all.
        limit: Per-task sample limit.
    """
    data_dir = Path(data_dir)
    if tasks is None:
        tasks = ["vqa", "csu", "captions"]

    all_samples: list[BanglaVerseSample] = []
    for task in tasks:
        jsonl_file = data_dir / f"{task}_records.jsonl"
        if jsonl_file.exists():
            samples = load_banglaverse_from_jsonl(jsonl_file, limit=limit, task_filter=task)
            all_samples.extend(samples)

    return all_samples
