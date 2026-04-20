from __future__ import annotations

import os

from .base import BaseVLM, load_image, parse_model_output
from ..types import ModelResponse, QuestionSample


def _require_env(var_name: str, *fallback_names: str) -> str:
    for name in (var_name, *fallback_names):
        value = os.getenv(name, "").strip()
        if value:
            return value
    expected = ", ".join((var_name, *fallback_names))
    raise RuntimeError(
        f"Missing API key environment variable. Provide one of: {expected}."
    )


class OpenAIModel(BaseVLM):
    def __init__(self, name: str = "gpt-4o"):
        super().__init__(name=name)
        self._client = None

    def _lazy_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))
        return self._client

    def generate(self, sample: QuestionSample) -> ModelResponse:
        client = self._lazy_client()
        image = load_image(sample.image_path)
        import base64
        from io import BytesIO

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        response = client.chat.completions.create(
            model=self.name,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": self.build_prompt(sample.question)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                ]},
            ],
        )
        text = response.choices[0].message.content or ""
        return parse_model_output(text)


class AnthropicModel(BaseVLM):
    def __init__(self, name: str = "claude-3-5-sonnet-latest"):
        super().__init__(name=name)
        self._client = None

    def _lazy_client(self):
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=_require_env("ANTHROPIC_API_KEY"))
        return self._client

    def generate(self, sample: QuestionSample) -> ModelResponse:
        client = self._lazy_client()
        image = load_image(sample.image_path)
        import base64
        from io import BytesIO

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        response = client.messages.create(
            model=self.name,
            max_tokens=128,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": self.build_prompt(sample.question)},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encoded}},
                ]},
            ],
        )
        text = "".join(block.text for block in response.content if hasattr(block, "text"))
        return parse_model_output(text)


class GeminiModel(BaseVLM):
    def __init__(self, name: str = "gemini-2.5-flash-lite"):
        super().__init__(name=name)
        self._client = None

    def _lazy_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(
                api_key=_require_env("GOOGLE_API_KEY", "GEMINI_API_KEY")
            )
        return self._client

    def generate(self, sample: QuestionSample) -> ModelResponse:
        client = self._lazy_client()
        image = load_image(sample.image_path)
        response = client.models.generate_content(
            model=self.name,
            contents=[self.build_prompt(sample.question), image],
        )
        return parse_model_output(getattr(response, "text", ""))


class DashScopeQwenModel(BaseVLM):
    def __init__(self, name: str = "qwen-vl-max"):
        super().__init__(name=name)

    def generate(self, sample: QuestionSample) -> ModelResponse:
        try:
            import dashscope
            from dashscope import MultiModalConversation
        except ImportError as exc:
            raise RuntimeError("Install dashscope to use Qwen-VL-Max") from exc
        dashscope.api_key = _require_env("DASHSCOPE_API_KEY")
        image = load_image(sample.image_path)
        import base64
        from io import BytesIO

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        response = MultiModalConversation.call(
            model="qwen-vl-max",
            messages=[
                {"role": "user", "content": [
                    {"image": f"data:image/png;base64,{encoded}"},
                    {"text": self.build_prompt(sample.question)},
                ]}
            ],
        )
        text = ""
        if hasattr(response, "output") and response.output:
            text = str(response.output)
        return parse_model_output(text)
