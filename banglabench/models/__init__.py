from __future__ import annotations

from .ollama import build_ollama_model


def build_model(name: str, load_in_4bit: bool = False):
    del load_in_4bit
    normalized = name.strip().lower()

    if normalized in {"ollama-llava", "llava-ollama"}:
        return build_ollama_model(name=normalized, model_id="llava:7b")
    if normalized in {"ollama-qwen2-vl", "qwen2-vl-ollama", "ollama-qwen2.5vl", "qwen2.5vl-ollama"}:
        return build_ollama_model(name=normalized, model_id="qwen2.5vl:7b")
    if normalized.startswith("ollama/"):
        model_id = name[len("ollama/"):]
        return build_ollama_model(name=normalized, model_id=model_id)

    raise KeyError(
        "Only Ollama models are enabled in this build. "
        "Use one of: ollama/qwen2.5vl:7b, ollama-llava, ollama-qwen2.5vl"
    )


__all__ = ["build_model"]
