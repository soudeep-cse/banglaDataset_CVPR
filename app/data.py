from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import QuestionSample


QUESTION_KEYS = ("question", "question_bn", "query", "q", "text", "prompt")
ANSWER_KEYS = ("answer", "answer_bn", "gt_answer", "label", "target", "gold")
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
            raise ValueError(f"Missing question text for sample {sample_id}")
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
