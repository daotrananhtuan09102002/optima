from collections import Counter
from dataclasses import dataclass
import math

from instaoptima.config import ExperimentConfig
from instaoptima.flan_t5_evaluator import FlanT5ObjectiveEvaluator
from instaoptima.instruction import Instruction
from instaoptima.instruction import TaskExample
from instaoptima.llm_client import LLMClient
from instaoptima.perplexity import PromptPerplexityScorer


@dataclass(frozen=True)
class CachedInstructionEvaluation:
    metrics: dict[str, float]
    performance_objective: float
    perplexity: float


class InstructionEvaluator:
    def __init__(
        self,
        llm_client: LLMClient | None,
        config: ExperimentConfig,
    ) -> None:
        self.llm_client = llm_client
        self.config = config
        self.perplexity_scorer = PromptPerplexityScorer(config)
        self.task_model_evaluator = FlanT5ObjectiveEvaluator(config)
        self._evaluation_cache: dict[tuple[str, tuple[tuple[str, str, str | None], ...]], CachedInstructionEvaluation] = {}

    def evaluate(
        self,
        instruction: Instruction,
        train_dataset: list[TaskExample],
        test_dataset: list[TaskExample],
    ) -> dict[str, float]:
        cache_key = instruction.cache_key()
        cached_result = self._evaluation_cache.get(cache_key)

        if cached_result is None:
            predictions = self.task_model_evaluator.evaluate(
                instruction=instruction,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
            )
            gold_labels = [example.label for example in test_dataset]
            normalized_predictions = [
                self._normalize_label(prediction)
                for prediction in predictions
            ]
            metrics = self._compute_metrics(normalized_predictions, gold_labels)
            performance_objective = self._performance_objective(metrics)
            perplexity = self.perplexity_scorer.score(instruction)
            cached_result = CachedInstructionEvaluation(
                metrics=metrics,
                performance_objective=performance_objective,
                perplexity=perplexity,
            )
            self._evaluation_cache[cache_key] = cached_result

        instruction.update_evaluation(
            dict(cached_result.metrics),
            cached_result.performance_objective,
            cached_result.perplexity,
        )
        return instruction.metrics

    def _normalize_label(self, raw_prediction: str) -> str:
        prediction = raw_prediction.lower()
        if self.config.label_space:
            for label in self.config.label_space:
                if label.lower() in prediction:
                    return label.lower()
        if "positive" in prediction:
            return "positive"
        if "negative" in prediction:
            return "negative"
        return prediction.strip()

    def _compute_metrics(
        self,
        predictions: list[str],
        gold_labels: list[str],
    ) -> dict[str, float]:
        if not gold_labels:
            return {
                "accuracy": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "prediction_entropy": 0.0,
            }

        labels = self.config.label_space or sorted(set(gold_labels))
        correct = sum(pred == gold for pred, gold in zip(predictions, gold_labels))
        accuracy = correct / len(gold_labels)

        per_label_metrics = []
        for label in labels:
            tp = sum(
                pred == label and gold == label
                for pred, gold in zip(predictions, gold_labels)
            )
            fp = sum(
                pred == label and gold != label
                for pred, gold in zip(predictions, gold_labels)
            )
            fn = sum(
                pred != label and gold == label
                for pred, gold in zip(predictions, gold_labels)
            )
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            per_label_metrics.append((precision, recall, f1))

        macro_precision = sum(metric[0] for metric in per_label_metrics) / len(labels)
        macro_recall = sum(metric[1] for metric in per_label_metrics) / len(labels)
        macro_f1 = sum(metric[2] for metric in per_label_metrics) / len(labels)

        return {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "prediction_entropy": self._prediction_entropy(predictions),
        }

    def _performance_objective(self, metrics: dict[str, float]) -> float:
        metric_sum = sum(
            metrics.get(metric_name, 0.0)
            for metric_name in self.config.performance_metric_names
        )
        if metric_sum == 0:
            return float("inf")
        return 1.0 / metric_sum

    @staticmethod
    def _prediction_entropy(predictions: list[str]) -> float:
        if not predictions:
            return 0.0

        counts = Counter(predictions)
        total = len(predictions)
        entropy = 0.0
        for count in counts.values():
            probability = count / total
            entropy -= probability * math.log(probability + 1e-12)
        return entropy
