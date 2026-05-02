"""Run per-segment eval using HuggingFace transformers (e.g. Phi-3.5-vision).

- Balanced sampling per segment
- Saves incrementally to results/run_hf_*.jsonl after every sample
- Real-time summary JSON updated after each sample
- Continues on any failure
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from app.data import load_dataset_from_file
from app.metrics import compare_answers, compute_metrics
from app.preprocess import prepare_segment_files

def _safe_stem(name: str) -> str:
    import re
    safe = re.sub(r"[\\/:*?\"<>|]+", "_", name.strip())
    safe = re.sub(r"\s+", "_", safe)
    safe = re.sub(r"_+", "_", safe)
    safe = safe.strip("._")
    return safe or "model"

def _confidence_to_unit(value: float | None) -> float:
    if value is None:
        return 0.5
    confidence = float(value)
    if confidence > 1.0:
        confidence /= 100.0
    return max(0.0, min(1.0, confidence))

def _write_realtime_summary(
    out_summary: Path,
    all_results: list[dict[str, Any]],
    args: argparse.Namespace,
    timestamp: str,
    balanced_n: int,
    segments: tuple,
) -> None:
    from app.metrics import compute_metrics
    overall = compute_metrics(all_results)
    seg_wise: dict[str, Any] = {}
    for seg in segments:
        rows = [r for r in all_results if r.get("answer_type") == seg]
        if rows:
            seg_wise[seg] = compute_metrics(rows)
    summary = {
        "model": args.model_id,
        "timestamp": timestamp,
        "balanced_samples_per_segment": balanced_n,
        "total_processed_so_far": len(all_results),
        "status": "in_progress" if overall["total_samples"] < balanced_n * 3 else "completed",
        "overall": overall,
        "segment_wise": seg_wise,
    }
    tmp = out_summary.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    tmp.replace(out_summary)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="microsoft/Phi-3.5-vision-instruct", help="HF model id")
    parser.add_argument("--device", default="cuda", help="torch device (cuda/cpu)")
    parser.add_argument("--num_crops", type=int, default=16, help="Processor num_crops (16 for single-frame, 4 for multi)")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Generation max_new_tokens")
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--preprocessed_dir", default="preprocessed_dataset")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--per_segment", type=int, default=None, help="Max per segment (default: all, balanced)")
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir = Path(args.preprocessed_dir)

    if not args.skip_preprocess:
        prepare_segment_files(data_dir=data_dir, output_dir=preprocessed_dir, limit=None)

    print(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=args.device,
        _attn_implementation="eager",
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        num_crops=args.num_crops,
    )
    print("Model loaded.\n")

    safe_name = _safe_stem(args.model_id.replace("/", "_"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_jsonl = results_dir / f"run_hf_{safe_name}_{timestamp}.jsonl"
    out_summary = results_dir / f"run_hf_{safe_name}_{timestamp}_summary.json"

    images_dir = data_dir / "images"
    all_results: list[dict[str, Any]] = []
    segments = ("polar", "numeric", "descriptive")

    # Load balanced samples
    seg_samples: dict[str, list] = {}
    for seg in segments:
        qa_file = preprocessed_dir / f"qa_{seg}.json"
        if not qa_file.exists():
            print(f"[SKIP] {seg}: not found")
            continue
        loaded = load_dataset_from_file(qa_file, images_dir=images_dir)
        if loaded:
            seg_samples[seg] = loaded

    if not seg_samples:
        print("No segments found.")
        return

    min_avail = min(len(s) for s in seg_samples.values())
    balanced_n = min(args.per_segment, min_avail) if args.per_segment else min_avail
    print(f"Balanced sampling: {balanced_n} per segment")
    for seg, samps in seg_samples.items():
        print(f"  {seg}: {len(samps)} available -> using {balanced_n}")

    for seg, all_samps in seg_samples.items():
        samples = all_samps[:balanced_n]
        print(f"\n=== {seg.upper()} ({len(samples)} samples) ===")
        iterator = tqdm(samples, desc=seg) if HAS_TQDM else samples

        for i, sample_row in enumerate(iterator):
            t0 = time.time()
            record: dict[str, Any] = {
                "qa_id": sample_row.sample_id,
                "image_file": Path(sample_row.image_path).name,
                "answer_type": seg,
                "question_bn": sample_row.question,
                "ground_truth": sample_row.answer,
                "predicted_answer": None,
                "confidence": None,
                "correct": False,
                "parse_success": False,
                "error": None,
                "latency_sec": 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            try:
                image = Image.open(sample_row.image_path).convert("RGB")
                messages = [
                    {
                        "role": "user",
                        "content": f"<|image_1|>\n{sample_row.question}",
                    }
                ]
                prompt = processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(prompt, [image], return_tensors="pt").to(args.device)

                with torch.no_grad():
                    gen_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        temperature=0.0,
                        do_sample=False,
                        eos_token_id=processor.tokenizer.eos_token_id,
                    )
                # Remove input tokens
                gen_ids = gen_ids[:, inputs["input_ids"].shape[1]:]
                raw_text = processor.batch_decode(
                    gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

                predicted = raw_text.strip()
                # Phi-3.5-vision-instruct: try to extract JSON answer/confidence
                confidence = None
                try:
                    j = json.loads(predicted)
                    predicted = str(j.get("answer", j.get("prediction", predicted))).strip()
                    confidence = float(j.get("confidence", j.get("score", 50)))
                except Exception:
                    # Fallback: simple regex for confidence
                    import re
                    m = re.search(r"confidence\s*[:=]\s*(\d+(?:\.\d+)?)", predicted, re.I)
                    if m:
                        confidence = float(m.group(1))
                    m2 = re.search(r"answer\s*[:=]\s*(.+)", predicted, re.I)
                    if m2:
                        predicted = m2.group(1).strip()

                confidence = _confidence_to_unit(confidence)
                correct = compare_answers(
                    predicted=predicted,
                    ground_truth=sample_row.answer,
                    segment=seg,
                    question=sample_row.question,
                )
                record["predicted_answer"] = predicted
                record["confidence"] = confidence
                record["correct"] = correct
                record["parse_success"] = True
                record["raw_output"] = raw_text

            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["error_trace"] = traceback.format_exc()

            record["latency_sec"] = round(time.time() - t0, 3)

            with out_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            all_results.append(record)
            _write_realtime_summary(out_summary, all_results, args, timestamp, balanced_n, segments)

            seg_done = i + 1
            seg_ok = sum(1 for r in all_results if r.get("answer_type") == seg and r["parse_success"])
            seg_err = seg_done - seg_ok
            seg_correct = sum(1 for r in all_results if r.get("answer_type") == seg and r.get("correct"))
            if not HAS_TQDM:
                print(f"  [{seg}] {seg_done}/{len(samples)} | ok={seg_ok} err={seg_err} correct={seg_correct} lat={record['latency_sec']:.1f}s")
            elif hasattr(iterator, "set_postfix"):
                iterator.set_postfix(ok=seg_ok, err=seg_err, acc=f"{seg_correct / seg_done:.1%}")

            if args.delay > 0:
                time.sleep(args.delay)

    # Final summary
    from app.metrics import compute_metrics
    overall = compute_metrics(all_results)
    seg_wise: dict[str, Any] = {}
    for seg in segments:
        rows = [r for r in all_results if r.get("answer_type") == seg]
        if rows:
            seg_wise[seg] = compute_metrics(rows)

    summary = {
        "model": args.model_id,
        "timestamp": timestamp,
        "balanced_samples_per_segment": balanced_n,
        "total_processed": len(all_results),
        "status": "completed",
        "overall": overall,
        "segment_wise": seg_wise,
        "output_jsonl": str(out_jsonl),
    }
    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== DONE ===")
    print(f"Total: {overall['total_samples']} | Errors: {overall['errors']} | Acc: {overall['accuracy']:.4f}")
    for seg, m in seg_wise.items():
        print(f"  [{seg}] n={m['total_samples']} acc={m['accuracy']:.4f}")
    print(f"\nFiles:\n  JSONL : {out_jsonl}\n  Summary: {out_summary}")


if __name__ == "__main__":
    main()
