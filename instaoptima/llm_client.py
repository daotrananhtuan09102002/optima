import os

import openai

from instaoptima.config import ExperimentConfig


class LLMClient:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if api_key:
            openai.api_key = api_key

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        response = openai.ChatCompletion.create(
            model=model or self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=self.config.max_generation_tokens,
        )
        return response["choices"][0]["message"]["content"].strip()
