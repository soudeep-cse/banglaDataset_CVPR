"""Post-run validation agent using qwen2.5vl as a coach to catch matching errors."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests


def _get_ollama_host(host: str | None = None) -> str:
    return host or os.getenv("OLLAMA_HOST", "http://localhost:11434")


def _llm_verify(
    question: str,
    ground_truth: str,
    predicted: str,
    marked_correct: bool,
    host: str,
    model: str = "qwen2.5vl:latest",
) -> dict[str, Any]:
    """Ask LLM: is the marked_correct label actually right?"""
    marked_label = 'CORRECT' if marked_correct else 'INCORRECT'
    marked_str = 'correct' if marked_correct else 'incorrect'
    prompt = (
        f"You are a Bangla VQA evaluation judge. A model answered a question and was marked as {marked_label}.\n"
        f"Your job: verify if this marking is right.\n\n"
        f"Question: {question}\n"
        f"Ground Truth: {ground_truth}\n"
        f"Predicted: {predicted}\n"
        f"Marked as: {marked_str}\n\n"
        "Rules:\n"
        "- Synonyms count as correct. Common Bangla synonyms:\n"
        "  * জল = পানি (water)\n"
        "  * জলের = পানির (of water)\n"
        "  * বাটি = বালি = বালিকা = পাত্র (bowl/container)\n"
        "  * জলের বাটি = পানির বালিকা = পানির বাটি = জলের বালি (water bowl)\n"
        "  * বিড়াল = ক্যাট = cat (same animal)\n"
        "  * কুকুর = ডগ = dog (same animal)\n"
        "  * পাখি = বার্ড = bird (same animal)\n"
        "  * নৌকা = বোট = boat (same)\n"
        "  * গাড়ি = কার = car (same)\n"
        "  * হাতি = elephant, ঘোড়া = horse, ভালুক = bear, বানর = monkey\n"
        "  * লাল = red, নীল = blue, সবুজ = green, হলুদ = yellow, সাদা = white, কালো = black\n"
        "- Partial answers containing the key concept count as correct:\n"
        "  * 'পানির দিকে' when truth is 'জল' -> correct (refers to water)\n"
        "  * 'কাঠের বেড়া' when truth is 'কাঠ' -> correct (contains key concept)\n"
        "- Completely different answers are incorrect:\n"
        "  * 'ছাঁচ' when truth is 'কাঠ' -> incorrect (different materials)\n"
        "  * 'লাল' when truth is 'নীল' -> incorrect (different colors)\n"
        "  * 'বিড়াল' when truth is 'কুকুর' -> incorrect (different animals)\n"
        "- Be LENIENT: if predicted and ground truth refer to the same real-world thing, mark as correct.\n\n"
        "Respond with JSON only:\n"
        '{"verdict": "correct" or "incorrect", "reason": "one line explanation", "is_mislabeled": true or false}'
    )

    try:
        response = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.0, "num_predict": 150}},
            timeout=30,
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip()

        import re as _re
        # Try JSON parse first
        match_json = _re.search(r"\{.*?\}", raw, _re.S)
        if match_json:
            try:
                payload = json.loads(match_json.group(0))
                verdict = str(payload.get("verdict", "")).strip().lower()
                is_mislabeled = bool(payload.get("is_mislabeled", False))
                return {
                    "verdict": verdict or "unknown",
                    "reason": payload.get("reason", ""),
                    "is_mislabeled": is_mislabeled,
                    "raw": raw,
                }
            except json.JSONDecodeError:
                pass

        # Fallback: extract verdict and is_mislabeled from raw text
        raw_lower = raw.lower()
        verdict_match = _re.search(r'"?verdict"?\s*:\s*"?(correct|incorrect)"?', raw_lower)
        mislabeled_match = _re.search(r'"?is_mislabeled"?\s*:\s*(true|false)', raw_lower)
        if verdict_match:
            verdict = verdict_match.group(1)
            is_mislabeled = mislabeled_match.group(1) == "true" if mislabeled_match else False
            return {"verdict": verdict, "reason": "parsed from malformed JSON", "is_mislabeled": is_mislabeled, "raw": raw}

    except Exception as e:
        return {"verdict": "unknown", "reason": str(e), "is_mislabeled": False, "raw": ""}

    return {"verdict": "unknown", "reason": "parse failed", "is_mislabeled": False, "raw": raw}


def validate_results(
    results: list[dict[str, Any]],
    ollama_host: str | None = None,
    judge_model: str = "qwen2.5vl:latest",
    sample_limit: int | None = None,
) -> dict[str, Any]:
    """
    Run post-evaluation validation on all results using LLM as coach.

    Args:
        results: List of result dicts from evaluator (each has question, ground_truth, predicted, correct)
        ollama_host: Ollama host URL
        judge_model: Model to use as judge
        sample_limit: Only validate first N results (for large datasets)

    Returns:
        Validation report with mislabeled samples and summary
    """
    host = _get_ollama_host(ollama_host)
    to_check = results[:sample_limit] if sample_limit else results

    mislabeled: list[dict[str, Any]] = []
    confirmed: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    print(f"\n=== Validation Agent ({judge_model}) ===")
    print(f"Checking {len(to_check)} results...\n")

    for i, item in enumerate(to_check):
        qa_id = item.get("qa_id", i)
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        predicted = item.get("predicted", "")
        marked_correct = bool(item.get("correct", False))
        segment = item.get("type", "descriptive")

        # Only validate descriptive — polar/numeric exact match is reliable
        if segment != "descriptive":
            confirmed.append({"qa_id": qa_id, "segment": segment, "skipped": True})
            continue

        result = _llm_verify(
            question=question,
            ground_truth=ground_truth,
            predicted=predicted,
            marked_correct=marked_correct,
            host=host,
            model=judge_model,
        )

        entry = {
            "qa_id": qa_id,
            "segment": segment,
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "marked_correct": marked_correct,
            "judge_verdict": result["verdict"],
            "judge_reason": result["reason"],
            "is_mislabeled": result["is_mislabeled"],
        }

        if result["verdict"] == "unknown":
            errors.append(entry)
        elif result["is_mislabeled"]:
            mislabeled.append(entry)
            print(f"  [MISLABELED] qa_id={qa_id}")
            print(f"    Q: {question}")
            print(f"    GT: {ground_truth} | Pred: {predicted}")
            print(f"    Marked: {'correct' if marked_correct else 'incorrect'} → Judge: {result['verdict']}")
            print(f"    Reason: {result['reason']}\n")
        else:
            confirmed.append(entry)

    total_descriptive = sum(1 for r in to_check if r.get("type") == "descriptive")
    summary = {
        "total_checked": len(to_check),
        "total_descriptive": total_descriptive,
        "mislabeled_count": len(mislabeled),
        "confirmed_count": len(confirmed),
        "error_count": len(errors),
        "mislabeled_rate": len(mislabeled) / total_descriptive if total_descriptive else 0.0,
    }

    print(f"=== Validation Summary ===")
    print(f"  Total checked   : {summary['total_checked']}")
    print(f"  Descriptive     : {summary['total_descriptive']}")
    print(f"  Mislabeled      : {summary['mislabeled_count']} ({summary['mislabeled_rate']:.1%})")
    print(f"  Confirmed OK    : {summary['confirmed_count']}")
    print(f"  Judge errors    : {summary['error_count']}")

    return {
        "summary": summary,
        "mislabeled": mislabeled,
        "confirmed": confirmed,
        "errors": errors,
    }


def validate_report_file(
    report_path: str | Path,
    output_path: str | Path | None = None,
    ollama_host: str | None = None,
    judge_model: str = "qwen2.5vl:latest",
    sample_limit: int | None = None,
) -> dict[str, Any]:
    """Load a saved report JSON and validate its results."""
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    results = report.get("results", [])

    validation = validate_results(
        results=results,
        ollama_host=ollama_host,
        judge_model=judge_model,
        sample_limit=sample_limit,
    )

    if output_path is None:
        output_path = report_path.parent / f"validation_{report_path.stem}.json"

    Path(output_path).write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nValidation report saved: {output_path}")
    return validation
