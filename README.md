# BanglaBayanno (Ollama-Only)

Bangla VQA benchmark runner focused on Ollama-based vision models.

## Features

- Ollama-only model pipeline
- Dataset loader for `data/qa.json` + `data/images/`
- Reliability metrics: accuracy, ECE, hallucination rate, logical consistency, CGA, F-score
- CLI evaluation flow
- FastAPI service for evaluate and generate APIs

## Setup (uv)

```bash
uv pip install -e ".[local,api]"
```

## Environment

Create `.env` in repo root:

```dotenv
OLLAMA_HOST=http://69.30.85.131:22054
```

If not set, default host is `http://localhost:11434`.

## Dataset Download

Linux/macOS:

```bash
bash scripts/download_dataset.sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_dataset.ps1
```

## Run Benchmark (CLI)

```bash
python evaluate.py --model ollama/qwen2.5vl:7b --sample 5
```

Outputs:

- `results/<model>_predictions.json`
- `results/<model>_metrics.json`

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
  -d '{"model":"ollama/qwen2.5vl:7b","sample":3}'
```

### Generate API (Path-Based) Example

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"ollama/qwen2.5vl:7b","question":"ছবিতে কী আছে?","image_path":"data/images/1.jpg"}'
```

### Generate API (Upload) Example

```bash
curl -X POST http://127.0.0.1:8000/v1/generate-upload \
  -F "model=ollama/qwen2.5vl:7b" \
  -F "question=ছবিতে কী আছে?" \
  -F "image=@data/images/1.jpg"
```
