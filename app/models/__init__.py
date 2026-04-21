from __future__ import annotations

from .ollama import build_ollama_model


def build_model(name: str, load_in_4bit: bool = False, host: str | None = None):
    del load_in_4bit
    normalized = name.strip().lower()

    if normalized in {"ollama-llava", "llava-ollama", "llava"}:
        return build_ollama_model(name=normalized, model_id="llava:7b", host=host)
    if normalized in {"ollama-qwen2-vl", "qwen2-vl-ollama", "ollama-qwen2.5vl", "qwen2.5vl-ollama", "qwen2.5vl"}:
        return build_ollama_model(name=normalized, model_id="qwen2.5vl:7b", host=host)
    if normalized in {"bakllava", "ollama-bakllava", "bakllava-ollama"}:
        return build_ollama_model(name=normalized, model_id="bakllava:latest", host=host)
    if normalized in {"moondream", "ollama-moondream", "moondream-ollama"}:
        return build_ollama_model(name=normalized, model_id="moondream:latest", host=host)
    if normalized in {"minicpm-v", "minicpmv", "ollama-minicpm-v", "minicpm-v-ollama"}:
        return build_ollama_model(name=normalized, model_id="minicpm-v:latest", host=host)
    if normalized in {"llama3.2-vision", "llama3.2-visionollama", "ollama-llama3.2-vision", "llama3.2-vision-ollama"}:
        return build_ollama_model(name=normalized, model_id="llama3.2-vision:latest", host=host)
    if normalized.startswith("ollama/"):
        model_id = name[len("ollama/"):]
        return build_ollama_model(name=normalized, model_id=model_id, host=host)

    raise KeyError(
        "Only Ollama models are enabled in this build. "
        "Use one of: ollama/qwen2.5vl:7b, ollama-llava, ollama-qwen2.5vl"
    )


__all__ = ["build_model"]
