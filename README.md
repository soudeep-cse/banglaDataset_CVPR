# BanglaBayanno Benchmark

This repository provides a runnable benchmark scaffold for Bangla VQA reliability evaluation.

## What it does

- Loads a Bangla VQA dataset from `data/qa.json` and `data/images/`
- Runs open-source or API-backed vision-language models through one shared interface
- Computes benchmark metrics:
  - Accuracy
  - Polar Accuracy
  - Numeric Accuracy
  - Descriptive Accuracy
  - ECE
  - Hallucination Rate
  - Logical Consistency
  - CGA
  - F-score
- Saves per-sample predictions and summary metrics in `results/`
- Generates paper-style figures in `figures/`

## Install

```bash
pip install -e .
pip install -e .[local]
pip install -e .[api]
```

## API keys with .env

Create a `.env` file in the project root:

```dotenv
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
ANTHROPIC_API_KEY=sk-ant-...
```

## Download dataset

```bash
bash scripts/download_dataset.sh
```

## Run evaluation

```bash
python evaluate.py --model qwen2-vl --sample 500
python evaluate.py --model idefics2 --sample 500 --load_in_4bit
python evaluate.py --model gemini-flash --sample 50
```

## Generate figures

```bash
python visualize.py --results_dir results --output figures
```

## Paper Pipeline (Methodology)

Use this exact pipeline for your paper's experiment section.

### Phase 1: Data Preparation

1. Download Bangla-Bayanno dataset with `scripts/download_dataset.sh`.
2. Benchmark loader reads:
  - `data/qa.json`
  - `data/images/`
3. Loader normalizes common schema variants (for example `question_bn`, `answer_bn`, `image_file`) into one internal sample format.

### Phase 2: Model Inference Layer

1. Entry point: `evaluate.py`.
2. It calls `banglabench/models/__init__.py` to construct one selected model adapter.
3. Adapters support two groups:
  - Open-source local VLMs (Idefics, LLaVA, Qwen)
  - API VLMs (GPT, Gemini, Claude, Qwen-VL-Max)
4. Unified prompt + response parser is applied through `banglabench/models/base.py`.

### Phase 3: Prediction Logging

For each sample, the evaluator stores:

- sample id
- question
- ground-truth answer
- model prediction
- confidence (if available)
- correctness flag
- refusal flag
- answer/question type
- pair id (for consistency)

Saved at:

- `results/<model>_predictions.json`

### Phase 4: Reliability Metrics

Metrics are computed in `banglabench/metrics.py`:

1. Accuracy
2. Polar Accuracy
3. Numeric Accuracy
4. Descriptive Accuracy
5. ECE (10-bin calibration)
6. Hallucination Rate (confident and wrong)
7. Logical Consistency (pair-based)
8. CGA (correctness on answered items)
9. F-score (attempt-rate and CGA harmonic mean)

Saved at:

- `results/<model>_metrics.json`

### Phase 5: Visualization and Reporting

Run `visualize.py` to generate:

1. `figures/reliability_diagram.png`
2. `figures/benchmark_table.png`
3. `figures/hallucination_rates.png`
4. `figures/category_heatmap.png`

### Phase 6: Error Taxonomy (Optional)

Run `taxonomy_inference.py` on a predictions file to get coarse buckets:

- correct
- refusal
- hallucination
- other_error

### Reproducibility Checklist

1. Fix sample size (for example 500) for all models.
2. Keep identical prompt and parsing rules across models.
3. Use same data split/order for all runs.
4. Store model name, timestamp, and metrics JSON for each run.
5. Report both performance and reliability metrics together.

## Experiment Runbook (Quick)

```bash
# 1) Setup
uv venv
.venv\Scripts\activate
uv pip install -e .
uv pip install -e ".[local]"
uv pip install -e ".[api]"

# 2) Data
bash scripts/download_dataset.sh

# 3) Benchmark (examples)
python evaluate.py --model llava-1.5 --sample 500 --load_in_4bit
python evaluate.py --model qwen2-vl --sample 500
python evaluate.py --model gemini-2.5-flash-lite --sample 500 --request_delay 2.5 --max_retries 6 --retry_backoff 3

# 4) Figures
python visualize.py --results_dir results --output figures

# 5) Error taxonomy
python taxonomy_inference.py results/gemini-2.5-flash-lite_predictions.json
```

## Notes

- Open-source models do not need API keys.
- Proprietary models need their provider keys in the environment.
- The benchmark expects the dataset rows to contain a question, answer, and image reference.
