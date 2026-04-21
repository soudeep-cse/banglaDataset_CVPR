# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BanglaBayanno** is a VLM benchmarking framework for Bangla Visual Question Answering. It evaluates vision-language models using reliability metrics beyond simple accuracy (ECE, hallucination rate, logical consistency). There is **no model training** — only inference and evaluation.

## Environment Setup

```bash
# Install with local (GPU) model support
pip install -e ".[local]"

# Install with API model support
pip install -e ".[api]"

# Install both
pip install -e ".[local,api]"

# Copy and fill in API keys
cp .env.example .env
```

Required API keys (in `.env`): `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `DASHSCOPE_API_KEY`, `HF_TOKEN`

## Dataset

```bash
# Download dataset via git-lfs (puts qa.json + images/ into data/)
bash scripts/download_dataset.sh
```

Dataset lives in `data/qa.json` + `data/images/`. The loader in `banglabench/data.py` normalizes many variant key names (e.g. `question_bn`, `query`, `q` → question field).

## Running Evaluation

```bash
# API model
python evaluate.py --model gpt-4o --sample 500

# Local model with 4-bit quantization
python evaluate.py --model qwen2-vl --load_in_4bit --sample 100

# Any HuggingFace model ID directly
python evaluate.py --model Qwen/Qwen2-VL-7B-Instruct --load_in_4bit

# Full options
python evaluate.py \
  --model <model_name> \
  --data_dir data/ \
  --results_dir results/ \
  --sample 500 \
  --load_in_4bit \
  --request_delay 1.0 \
  --max_retries 5 \
  --retry_backoff 2.0
```

Outputs written to `results/<model>_predictions.json` and `results/<model>_metrics.json`.

## Visualization & Analysis

```bash
# Generate paper-ready figures from results/
python visualize.py --results_dir results/ --output figures/

# Classify prediction errors into taxonomy categories
python taxonomy_inference.py results/<model>_predictions.json
```

## Supported Models

| Name (--model) | Backend |
|---|---|
| `gpt-4o` | OpenAI API |
| `claude-3-5-sonnet` | Anthropic API |
| `gemini-1.5-pro`, `gemini-2.5-flash-lite` | Google GenAI |
| `qwen-vl-max` | DashScope API |
| `idefics2`, `idefics3` | HuggingFace (local) |
| `llava-1.5`, `llava-next` | HuggingFace (local) |
| `qwen-vl-chat`, `qwen2-vl` | HuggingFace (local) |
| Any HF model ID with `/` | HuggingFace (local) |

## Architecture

```
evaluate.py / visualize.py / taxonomy_inference.py   ← entry points
banglabench/
  data.py       ← loads qa.json, normalizes schema → QuestionSample[]
  types.py      ← QuestionSample, ModelResponse, PredictionRecord dataclasses
  evaluator.py  ← main loop: iterate samples, retry on rate-limit, save results
  metrics.py    ← computes 9 reliability metrics from PredictionRecord[]
  visuals.py    ← matplotlib figure generation
  models/
    __init__.py     ← build_model() factory, name→class mapping
    base.py         ← BaseVLM abstract class, prompt builder, JSON parser
    local.py        ← TransformersVLM (AutoProcessor + vision model loaders)
    proprietary.py  ← OpenAI, Anthropic, Gemini, DashScope wrappers
    idefics.py / llava.py / qwen_vl.py  ← HF model ID mappings
```

**Key design decisions:**
- All models implement `BaseVLM.generate(sample) → ModelResponse` — uniform interface regardless of backend.
- The prompt always requests a JSON response `{"answer": "...", "confidence": 0–100}`; `base.py` handles extraction/fallback.
- `evaluator.py` uses exponential backoff (`--max_retries`, `--retry_backoff`) to handle API rate limits gracefully.
- `data.py` intentionally accepts many schema variants so results from different dataset versions remain compatible.

## Adding a New Model

1. Add a class inheriting `BaseVLM` in `banglabench/models/` (or `proprietary.py` for APIs).
2. Implement `generate(self, sample: QuestionSample) -> ModelResponse`.
3. Register the name in `banglabench/models/__init__.py`'s `build_model()` factory.

## Metrics Reference

Computed by `banglabench/metrics.py`:

| Metric | Description |
|---|---|
| Accuracy | Correct / Total |
| Attempt Rate | Non-refusal answers / Total |
| CGA | Correctness Given Attempted |
| F-score | Harmonic mean of Attempt Rate and CGA |
| ECE | Expected Calibration Error (confidence vs accuracy) |
| Hallucination Rate | Wrong confident answers |
| Logical Consistency | Agreement on paired questions |
| Polar/Numeric/Descriptive Accuracy | Per question-type breakdown |
