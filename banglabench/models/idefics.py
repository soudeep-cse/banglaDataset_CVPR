from __future__ import annotations

from .local import build_local_model


MODEL_MAP = {
    "idefics2": "HuggingFaceM4/idefics2-8b",
    "idefics3": "HuggingFaceM4/Idefics3-8B-Llama3",
}


def build_idefics_model(name: str, load_in_4bit: bool = False):
    return build_local_model(name=name, model_id=MODEL_MAP[name], load_in_4bit=load_in_4bit)
