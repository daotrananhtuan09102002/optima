import math

from instaoptima.config import ExperimentConfig
from instaoptima.instruction import Instruction

try:
    import torch
    from transformers import RobertaForMaskedLM, RobertaTokenizerFast
except ImportError:  # pragma: no cover
    torch = None
    RobertaForMaskedLM = None
    RobertaTokenizerFast = None


class PromptPerplexityScorer:
    """Pseudo-perplexity scorer using a RoBERTa masked language model."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._tokenizer = None
        self._model = None

    def score(self, instruction: Instruction) -> float:
        prompt_text = instruction.definition
        if (
            RobertaTokenizerFast is None
            or RobertaForMaskedLM is None
            or torch is None
        ):
            # Fallback heuristic when the local environment does not have
            # the dependencies for RoBERTa pseudo-perplexity scoring.
            return float(len(prompt_text.split()))

        tokenizer = self._get_tokenizer()
        model = self._get_model()
        encoded = tokenizer(prompt_text, return_tensors="pt", truncation=True)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        total_loss = 0.0
        token_count = input_ids.size(1)
        if token_count <= 2:
            return 0.0

        for index in range(1, token_count - 1):
            masked_ids = input_ids.clone()
            masked_ids[0, index] = tokenizer.mask_token_id
            with torch.no_grad():
                outputs = model(
                    input_ids=masked_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
            total_loss += outputs.loss.item()

        return math.exp(total_loss / (token_count - 2))

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = RobertaTokenizerFast.from_pretrained(
                self.config.perplexity_model_name
            )
        return self._tokenizer

    def _get_model(self):
        if self._model is None:
            self._model = RobertaForMaskedLM.from_pretrained(
                self.config.perplexity_model_name
            )
            self._model.eval()
        return self._model
