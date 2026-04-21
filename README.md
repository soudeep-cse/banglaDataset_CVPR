# BanglaBayanno (Ollama-Only)

Bangla VQA benchmark runner focused on Ollama-based vision models.

## Features

- Ollama benchmarking pipeline with preprocessing + segment split
- Dataset loader for `dataset/qa.json` + `dataset/images/`
- Metrics: Accuracy, OER, ECE, Brier, MCW (overall + segment-wise)
- Final single report output: `results/evaluation_report.json`
- CLI evaluation flow
- FastAPI service for evaluate and generate APIs

## Setup (uv)

```bash
uv pip install -e .
```

## Environment

Create `.env` in repo root:

```dotenv
OLLAMA_HOST=http://69.30.85.131:22054
```

If not set, default host is `http://localhost:11434`.
You can also override it per run with `--ollama_host http://69.30.85.131:22054`.

## Dataset

Place dataset files manually in:

- `dataset/qa.json`
- `dataset/images/`

If you prefer the old layout, `data/` also works as a fallback.

The evaluator and API both read from this location by default.

## Preprocessing

Applied before benchmarking:

- Whitespace + punctuation cleanup
- Unicode normalization (NFC)
- Polar canonicalization (`yes/no` variants -> `হ্যাঁ/না`)
- Numeric digit normalization (Bangla -> English digits)
- Lowercase normalize
- Trailing punctuation removal
- Metadata unchanged (`qa_id`, `image_file`, `answer_type`)

Segment files are saved in `preprocessed/`:

- `preprocessed/qa_polar.json`
- `preprocessed/qa_numeric.json`
- `preprocessed/qa_descriptive.json`

## Run Benchmark (CLI)

```bash
python evaluate.py --model ollama/qwen2.5vl:7b --sample 5
```

Useful options:

```bash
python evaluate.py --model qwen2.5vl --sample 100 --preprocessed_dir preprocessed
python evaluate.py --model llava --sample 100 --oer_threshold 0.7 --ece_bins 10
python evaluate.py --model bakllava --skip_preprocess
```

If your RunPod host is not in `.env`, pass it explicitly:

```bash
python evaluate.py --model ollama/qwen2.5vl:7b --sample 5 --ollama_host http://69.30.85.131:22054
```

Outputs:

- `results/evaluation_report.json`

Console output format:

```text
=== Overall ===
  total: ...
  accuracy: ...
  oer: ...
  ece: ...
  brier: ...
  mcw: ...

=== Segment-wise ===
  [polar] (n=...)
  [numeric] (n=...)
  [descriptive] (n=...)
```

Supported model names (Ollama aliases):

- `qwen2.5vl`
- `llava`
- `bakllava`
- `moondream`
- `minicpm-v`
- `llama3.2-vision`
- or explicit `ollama/<model_id>`

## FastAPI Service

Run server:

```bash
python serve.py
```

Server base: `http://127.0.0.1:8000`

Endpoints:

- `GET /health`
- `POST /v1/evaluate` (JSON body)
- `POST /v1/generate` (JSON body, image path based)
- `POST /v1/generate-upload` (multipart form-data, image file upload)

### Evaluate API Example

```bash
curl -X POST http://127.0.0.1:8000/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{"model":"ollama/qwen2.5vl:7b","sample":3,"ollama_host":"http://69.30.85.131:22054"}'
```

### Generate API (Path-Based) Example

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"ollama/qwen2.5vl:7b","question":"ছবিতে কী আছে?","image_path":"dataset/images/1.jpg","ollama_host":"http://69.30.85.131:22054"}'
```

### Generate API (Upload) Example

```bash
curl -X POST http://127.0.0.1:8000/v1/generate-upload \
  -F "model=ollama/qwen2.5vl:7b" \
  -F "ollama_host=http://69.30.85.131:22054" \
  -F "question=ছবিতে কী আছে?" \
  -F "image=@dataset/images/1.jpg"
```
