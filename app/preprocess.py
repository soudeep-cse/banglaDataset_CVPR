from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any


QUESTION_KEYS = ("question_bn", "question", "query", "q", "text", "prompt")
ANSWER_KEYS = ("answer_bn", "answer", "gt_answer", "label", "target", "gold")
IMAGE_KEYS = ("image_file", "image", "image_path", "image_name", "file_name", "filename", "img", "path")
ANSWER_TYPE_KEYS = ("answer_type", "answer_category", "atype", "category", "type")
ID_KEYS = ("qa_id", "id", "uid", "sample_id", "qid")

YES_VARIANTS = {"yes", "y", "haan", "ha", "হ্যাঁ", "হ্যা", "হঁ্যা", "জি", "yes."}
NO_VARIANTS = {"no", "n", "na", "nah", "না", "no."}
TRAILING_PUNCT_RE = re.compile(r"[।?!]+$")


def _first_present(row: dict[str, Any], keys: tuple[str, ...], default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def _extract_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "samples", "items", "examples", "qa"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    raise ValueError("Unsupported QA JSON structure")


def _normalize_digits(text: str) -> str:
    return text.translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789"))


def normalize_text(text: str, lowercase: bool = True, remove_trailing_punct: bool = True) -> str:
    normalized = unicodedata.normalize("NFC", text or "")
    normalized = normalized.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "…": "..."}))
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if lowercase:
        normalized = normalized.lower()
    if remove_trailing_punct:
        normalized = TRAILING_PUNCT_RE.sub("", normalized).strip()
    return normalized


def infer_segment(answer_type: str, question: str, answer: str) -> str:
    normalized_type = normalize_text(answer_type, lowercase=True, remove_trailing_punct=True)
    if normalized_type in {"polar", "yes/no", "yesno", "boolean", "bool"}:
        return "polar"
    if normalized_type in {"numeric", "number", "count"}:
        return "numeric"
    if normalized_type in {"descriptive", "description", "open", "freeform"}:
        return "descriptive"

    ans = normalize_text(answer)
    if ans in YES_VARIANTS or ans in NO_VARIANTS:
        return "polar"
    if re.search(r"-?\d+(?:\.\d+)?", _normalize_digits(ans)):
        return "numeric"

    q = normalize_text(question)
    if any(token in q for token in ("কত", "সংখ্যা", "how many", "count")):
        return "numeric"
    return "descriptive"


def preprocess_answer(answer: str, segment: str) -> str:
    cleaned = normalize_text(answer, lowercase=True, remove_trailing_punct=True)
    if segment == "polar":
        if cleaned in YES_VARIANTS:
            return "হ্যাঁ"
        if cleaned in NO_VARIANTS:
            return "না"
        return cleaned
    if segment == "numeric":
        return _normalize_digits(cleaned)
    return cleaned


def prepare_segment_files(data_dir: str | Path, output_dir: str | Path = "preprocessed", limit: int | None = None) -> dict[str, object]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        alternate = Path("dataset") if data_dir.name == "data" else Path("data")
        if alternate.exists():
            data_dir = alternate

    qa_path = data_dir / "qa.json"
    payload = json.loads(qa_path.read_text(encoding="utf-8"))
    rows = _extract_rows(payload)
    if limit is not None:
        rows = rows[:limit]

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    split_rows: dict[str, list[dict[str, Any]]] = {"polar": [], "numeric": [], "descriptive": []}
    for index, row in enumerate(rows):
        sample_id = str(_first_present(row, ID_KEYS, f"sample-{index}"))
        question = str(_first_present(row, QUESTION_KEYS, "")).strip()
        answer = str(_first_present(row, ANSWER_KEYS, "")).strip()
        image_file = str(_first_present(row, IMAGE_KEYS, "")).strip()
        answer_type = str(_first_present(row, ANSWER_TYPE_KEYS, "unknown"))

        segment = infer_segment(answer_type, question, answer)
        cleaned_question = normalize_text(question, lowercase=False, remove_trailing_punct=False)
        cleaned_answer = preprocess_answer(answer, segment)

        out_row = dict(row)
        out_row["qa_id"] = sample_id
        out_row["question_bn"] = cleaned_question
        out_row["answer_bn"] = cleaned_answer
        out_row["answer"] = cleaned_answer
        if image_file and "image_file" not in out_row:
            out_row["image_file"] = image_file
        out_row["type"] = segment
        split_rows[segment].append(out_row)

    output_paths: dict[str, str] = {}
    for segment, segment_rows in split_rows.items():
        out_path = output_root / f"qa_{segment}.json"
        out_path.write_text(json.dumps(segment_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        output_paths[segment] = str(out_path)

    return {
        "data_dir": str(data_dir),
        "output_dir": str(output_root),
        "counts": {key: len(value) for key, value in split_rows.items()},
        "files": output_paths,
        "total": sum(len(value) for value in split_rows.values()),
    }
