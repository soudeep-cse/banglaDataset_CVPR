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
GOOGLE_API_KEY=AIza...
GEMINI_API_KEY=AIza...
ANTHROPIC_API_KEY=sk-ant-...
DASHSCOPE_API_KEY=sk-...
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

## Notes

- Open-source models do not need API keys.
- Proprietary models need their provider keys in the environment.
- The benchmark expects the dataset rows to contain a question, answer, and image reference.
