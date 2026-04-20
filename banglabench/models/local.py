from __future__ import annotations

from typing import Any

from .base import BaseVLM, load_image, parse_model_output
from ..types import ModelResponse, QuestionSample


class TransformersVLM(BaseVLM):
    def __init__(self, name: str, model_id: str, load_in_4bit: bool = False):
        super().__init__(name=name)
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        self._processor = None
        self._model = None

    def _lazy_load(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        try:
            import transformers
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers to run local models") from exc

        processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        if self.load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        vision_candidates = (
            getattr(transformers, "AutoModelForVision2Seq", None),
            getattr(transformers, "AutoModelForImageTextToText", None),
        )
        model = None
        last_error: Exception | None = None
        for model_cls in vision_candidates:
            if model_cls is None:
                continue
            try:
                model = model_cls.from_pretrained(self.model_id, **model_kwargs)
                break
            except Exception as exc:
                last_error = exc

        if model is None:
            try:
                model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
            except Exception as exc:
                if last_error is not None:
                    raise RuntimeError(
                        f"Failed to load {self.model_id} with available vision/casual model loaders. "
                        f"Last vision loader error: {last_error}"
                    ) from exc
                raise
        self._processor = processor
        self._model = model

    def _input_device(self):
        if self._model is None:
            return None
        try:
            return next(self._model.parameters()).device
        except StopIteration:
            return None

    def _prepare_inputs(self, question: str, image) -> dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.build_prompt(question)},
                ],
            }
        ]
        if hasattr(self._processor, "apply_chat_template"):
            prompt = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=prompt, images=image, return_tensors="pt")
        else:
            prompt = "<image>\n" + self.build_prompt(question)
            inputs = self._processor(text=prompt, images=image, return_tensors="pt")
        device = self._input_device()
        if device is not None:
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        return inputs

    def generate(self, sample: QuestionSample) -> ModelResponse:
        self._lazy_load()
        import torch

        image = load_image(sample.image_path)
        inputs = self._prepare_inputs(sample.question, image)
        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, max_new_tokens=64, do_sample=False)
        decoded = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return parse_model_output(decoded)


def build_local_model(name: str, model_id: str, load_in_4bit: bool = False) -> TransformersVLM:
    return TransformersVLM(name=name, model_id=model_id, load_in_4bit=load_in_4bit)
