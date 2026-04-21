from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .schemas import EvaluateRequest, GenerateRequest, GenerateResponse, HealthResponse
from .service import run_evaluation, run_single_generation, run_single_generation_from_upload

app = FastAPI(title="BanglaBayanno Ollama API", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/v1/evaluate")
def evaluate_endpoint(payload: EvaluateRequest) -> dict[str, object]:
    try:
        return run_evaluation(
            model=payload.model,
            data_dir=payload.data_dir,
            results_dir=payload.results_dir,
            sample=payload.sample,
            request_delay=payload.request_delay,
            max_retries=payload.max_retries,
            retry_backoff=payload.retry_backoff,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/generate", response_model=GenerateResponse)
def generate_endpoint(payload: GenerateRequest) -> GenerateResponse:
    try:
        response = run_single_generation(
            model=payload.model,
            question=payload.question,
            image_path=payload.image_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return GenerateResponse(
        answer=response.answer,
        confidence=response.confidence,
        refused=response.refused,
        raw_text=response.raw_text,
    )


@app.post("/v1/generate-upload", response_model=GenerateResponse)
async def generate_upload_endpoint(
    question: str = Form(...),
    image: UploadFile = File(...),
    model: str = Form("ollama/qwen2.5vl:7b"),
) -> GenerateResponse:
    try:
        image_bytes = await image.read()
        response = run_single_generation_from_upload(
            model=model,
            question=question,
            image_bytes=image_bytes,
            filename=image.filename or "upload.jpg",
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return GenerateResponse(
        answer=response.answer,
        confidence=response.confidence,
        refused=response.refused,
        raw_text=response.raw_text,
    )
