# BanglaVerse Processing

This folder contains a reusable processor for the BanglaVerse dataset.

## What It Does

- Reads all JSON files from `BanglaVerse/data/dialectual_data/*/Results/{VQA,CSU,Captions}`
- Resolves image paths from `BanglaVerse/data/images/*/images`
- For MCQ tasks (`VQA`, `CSU`), finds the correct answer index from options
- Exports merged and split JSONL files for repeatable runs
- Writes a `summary.json` with counts and diagnostics

## Run

From repository root:

```bash
python banglaverse_processing/process_dataset.py
```

Custom paths:

```bash
python banglaverse_processing/process_dataset.py --root BanglaVerse --output banglaverse_processing/output
```

## Output Files

Default output directory: `banglaverse_processing/output`

- `all_records.jsonl`: all tasks combined
- `vqa_records.jsonl`: only VQA rows
- `csu_records.jsonl`: only CSU rows
- `captions_records.jsonl`: only caption rows
- `<dialect>_records.jsonl`: one file per dialect (barishal, chittagong, noakhali, rangpur, sylhet)
- `records_missing_image_path.jsonl`: rows where image path could not be resolved
- `records_missing_answer_index.jsonl`: MCQ rows where answer did not match any option
- `summary.json`: run summary and quality stats

## Main Fields

Common fields:
- `record_id`, `dialect`, `task`, `domain`, `image_id`, `image_path`, `source_file`

MCQ fields (`vqa`, `csu`):
- `question`, `original_question`, `options`, `original_options`
- `answer`, `original_answer`, `answer_index`, `correct_option`

Caption fields:
- `caption`, `original_caption`
