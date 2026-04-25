from __future__ import annotations

from .ollama import build_ollama_model
from .ollama_http import build_ollama_http_model


def build_model(name: str, load_in_4bit: bool = False, host: str | None = None):
    del load_in_4bit
    normalized = name.strip().lower()

    # RunPod server models (exact names from ollama list)
    if normalized in {"ollama-llava", "llava-ollama", "llava", "llava:7b"}:
        return build_ollama_model(name=normalized, model_id="llava:7b", host=host)
    if normalized in {"bakllava", "ollama-bakllava", "bakllava-ollama", "bakllava:7b"}:
        return build_ollama_model(name=normalized, model_id="bakllava:7b", host=host)
    if normalized in {"moondream", "ollama-moondream", "moondream-ollama", "moondream:1.8b"}:
        return build_ollama_model(name=normalized, model_id="moondream:1.8b", host=host)
    if normalized in {"qwen3.5", "ollama-qwen3.5", "qwen3.5-ollama", "qwen3.5:9b"}:
        # Note: qwen3.5:9b is text-only, not a vision model
        return build_ollama_model(name=normalized, model_id="qwen3.5:9b", host=host)
    # Legacy aliases for compatibility (qwen2.5vl maps to qwen3.5:9b)
    if normalized in {"ollama-qwen2-vl", "qwen2-vl-ollama", "ollama-qwen2.5vl", "qwen2.5vl-ollama", "qwen2.5vl"}:
        return build_ollama_model(name=normalized, model_id="qwen3.5:9b", host=host)
    if normalized in {"minicpm-v", "minicpmv", "ollama-minicpm-v", "minicpm-v-ollama"}:
        return build_ollama_model(name=normalized, model_id="minicpm-v:latest", host=host)
    if normalized in {"llama3.2-vision", "llama3.2-visionollama", "ollama-llama3.2-vision", "llama3.2-vision-ollama"}:
        return build_ollama_model(name=normalized, model_id="llama3.2-vision:latest", host=host)
    if normalized.startswith("ollama/"):
        model_id = name[len("ollama/"):]
        return build_ollama_model(name=normalized, model_id=model_id, host=host)

    # HTTP-based variants (using requests instead of ollama Python client)
    # RunPod server models via HTTP
    if normalized in {"http-llava", "httpllava", "llava-http", "http-llava:7b"}:
        return build_ollama_http_model(name=normalized, model_id="llava:7b", host=host)
    if normalized in {"http-bakllava", "httpbakllava", "bakllava-http", "http-bakllava:7b"}:
        return build_ollama_http_model(name=normalized, model_id="bakllava:7b", host=host)
    if normalized in {"http-moondream", "httpmoondream", "moondream-http", "http-moondream:1.8b"}:
        return build_ollama_http_model(name=normalized, model_id="moondream:1.8b", host=host)
    if normalized in {"http-qwen3.5", "httpqwen3.5", "qwen3.5-http", "http-qwen3.5:9b", "qwen2.5vl:7b"}:
        return build_ollama_http_model(name=normalized, model_id="qwen2.5vl:7b", host=host)
    # Generic HTTP prefix
    if normalized.startswith("http-ollama/") or normalized.startswith("httpllama/"):
        model_id = name.split("/", 1)[1] if "/" in name else "llava:7b"
        return build_ollama_http_model(name=normalized, model_id=model_id, host=host)

    raise KeyError(
        "Only Ollama models are enabled. Available models from RunPod: "
        "llava, llava:7b, bakllava, bakllava:7b, moondream, moondream:1.8b, "
        "qwen3.5, qwen3.5:9b. "
        "Use 'http-' prefix for direct HTTP API (e.g., http-llava)"
    )


__all__ = ["build_model"]
