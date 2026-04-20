from __future__ import annotations

from .local import build_local_model


MODEL_MAP = {
    "qwen-vl-chat": "Qwen/Qwen-VL-Chat",
    "qwen2-vl": "Qwen/Qwen2-VL-7B-Instruct",
}


def build_qwen_model(name: str, load_in_4bit: bool = False):
    return build_local_model(name=name, model_id=MODEL_MAP[name], load_in_4bit=load_in_4bit)
