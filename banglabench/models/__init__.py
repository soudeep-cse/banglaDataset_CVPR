from __future__ import annotations

from .idefics import build_idefics_model
from .llava import build_llava_model
from .proprietary import AnthropicModel, DashScopeQwenModel, GeminiModel, OpenAIModel
from .qwen_vl import build_qwen_model


OPEN_SOURCE_MODELS = {
    "idefics2": lambda load_in_4bit=False: build_idefics_model("idefics2", load_in_4bit=load_in_4bit),
    "idefics3": lambda load_in_4bit=False: build_idefics_model("idefics3", load_in_4bit=load_in_4bit),
    "llava-next": lambda load_in_4bit=False: build_llava_model("llava-next", load_in_4bit=load_in_4bit),
    "llava-1.5": lambda load_in_4bit=False: build_llava_model("llava-1.5", load_in_4bit=load_in_4bit),
    "qwen-vl-chat": lambda load_in_4bit=False: build_qwen_model("qwen-vl-chat", load_in_4bit=load_in_4bit),
    "qwen2-vl": lambda load_in_4bit=False: build_qwen_model("qwen2-vl", load_in_4bit=load_in_4bit),
}


def build_model(name: str, load_in_4bit: bool = False):
    normalized = name.strip().lower()
    if normalized in OPEN_SOURCE_MODELS:
        return OPEN_SOURCE_MODELS[normalized](load_in_4bit=load_in_4bit)
    if normalized == "gpt-4o":
        return OpenAIModel(name="gpt-4o")
    if normalized in {"claude", "claude-3.5-sonnet", "claude-3-5-sonnet", "claude-3-5-sonnet-latest"}:
        return AnthropicModel(name="claude-3-5-sonnet-latest")
    if normalized in {"gemini", "gemini-1.5-pro"}:
        return GeminiModel(name="gemini-1.5-pro")
    if normalized in {"gemini-flash", "gemini-flash-lite", "gemini-1.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-flash"}:
        if normalized in {"gemini-1.5-flash"}:
            return GeminiModel(name="gemini-1.5-flash")
        if normalized in {"gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-flash-lite", "gemini-flash"}:
            return GeminiModel(name="gemini-2.5-flash-lite")
    if normalized == "qwen-vl-max":
        return DashScopeQwenModel(name="qwen-vl-max")
    
    # Allow arbitrary HuggingFace model IDs (e.g., "unsloth/gemma-3-12b-it-bnb-4bit")
    if "/" in name:
        from .local import TransformersVLM
        return TransformersVLM(name=name, model_id=name, load_in_4bit=load_in_4bit)
    
    raise KeyError(f"Unknown model name: {name}")


__all__ = ["build_model"]
