from __future__ import annotations

from .local import build_local_model


MODEL_MAP = {
    "llava-next": "llava-hf/llava-v1.6-vicuna-7b-hf",
    "llava-1.5": "llava-hf/llava-1.5-7b-hf",
}


def build_llava_model(name: str, load_in_4bit: bool = False):
    return build_local_model(name=name, model_id=MODEL_MAP[name], load_in_4bit=load_in_4bit)
