import os
import math
from typing import Optional

from openai import OpenAI

from instaoptima.config import ExperimentConfig


class LLMClient:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        prompt_text = self._sanitize_text(prompt)
        resolved_model = self._sanitize_model(model or self.config.model)
        resolved_temperature = self._sanitize_temperature(
            temperature if temperature is not None else self.config.temperature
        )

        response = self.client.chat.completions.create(
            model=resolved_model,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=resolved_temperature,
            max_tokens=self.config.max_generation_tokens,
        )
        content = response.choices[0].message.content
        return (content or "").strip()

    @staticmethod
    def _sanitize_text(value: str) -> str:
        text = str(value)
        # Remove invalid surrogate code points that can break JSON encoding.
        return text.encode("utf-8", errors="ignore").decode("utf-8")

    @staticmethod
    def _sanitize_model(value: str) -> str:
        model = str(value).strip()
        if not model:
            raise ValueError("OpenAI model name must be a non-empty string.")
        return model

    @staticmethod
    def _sanitize_temperature(value: float) -> float:
        temperature = float(value)
        if not math.isfinite(temperature):
            raise ValueError("OpenAI temperature must be a finite number.")
        return temperature
