import math

from instaoptima.config import ExperimentConfig
from instaoptima.instruction import Instruction

try:
    import torch
    from transformers import AutoConfig, AutoTokenizer, RobertaForMaskedLM
except ImportError:  # pragma: no cover
    torch = None
    AutoConfig = None
    AutoTokenizer = None
    RobertaForMaskedLM = None


class PromptPerplexityScorer:
    """Pseudo-perplexity scorer using a RoBERTa masked language model."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._tokenizer = None
        self._model = None
        self._backend_unavailable = False

    def score(self, instruction: Instruction) -> float:
        prompt_text = instruction.full_instruction_text
        if (
            AutoTokenizer is None
            or AutoConfig is None
            or RobertaForMaskedLM is None
            or torch is None
        ):
            # Fallback heuristic when the local environment does not have
            # the dependencies for RoBERTa pseudo-perplexity scoring.
            return float(len(prompt_text.split()))

        tokenizer = self._get_tokenizer()
        model = self._get_model()
        if tokenizer is None or model is None:
            return float(len(prompt_text.split()))
        encoded = tokenizer(prompt_text, return_tensors="pt", truncation=True)
        input_ids = encoded["input_ids"]
        token_count = input_ids.size(1)
        if token_count == 0:
            return 0.0

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=encoded.get("attention_mask"),
                labels=input_ids,
            )

        # Match the paper's implementation: exp(mlm_loss / token_count).
        return math.exp(outputs.loss.item() / token_count)

    def _get_tokenizer(self):
        if self._backend_unavailable:
            return None
        if self._tokenizer is None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.config.perplexity_model_name
                )
            except (ImportError, OSError):  # pragma: no cover
                self._backend_unavailable = True
                return None
        return self._tokenizer

    def _get_model(self):
        if self._backend_unavailable:
            return None
        if self._model is None:
            try:
                model_config = AutoConfig.from_pretrained(
                    self.config.perplexity_model_name
                )
                self._model = RobertaForMaskedLM.from_pretrained(
                    self.config.perplexity_model_name,
                    config=model_config,
                )
            except (ImportError, OSError):  # pragma: no cover
                self._backend_unavailable = True
                return None
            self._model.eval()
            print(
                "Loaded perplexity model "
                f"'{self.config.perplexity_model_name}' successfully; "
                "using RoBERTa MLM scoring (not pseudo/fallback)."
            )
        return self._model
