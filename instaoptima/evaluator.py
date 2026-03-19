from collections import Counter
import math

from instaoptima.config import ExperimentConfig
from instaoptima.instruction import Instruction
from instaoptima.instruction import TaskExample
from instaoptima.llm_client import LLMClient
from instaoptima.perplexity import PromptPerplexityScorer


class InstructionEvaluator:
    def __init__(self, llm_client: LLMClient, config: ExperimentConfig) -> None:
        self.llm_client = llm_client
        self.config = config
        self.perplexity_scorer = PromptPerplexityScorer(config)

    def evaluate(
        self, instruction: Instruction, dataset: list[TaskExample]
    ) -> dict[str, float]:
        predictions: list[str] = []
        gold_labels: list[str] = []

        for example in dataset:
            prompt = instruction.build_prompt(example, self.config)
            prediction = self._normalize_label(self.llm_client.generate(prompt))
            predictions.append(prediction)
            gold_labels.append(example.label)

        metrics = self._compute_metrics(predictions, gold_labels)
        performance_objective = self._performance_objective(metrics)
        perplexity = self.perplexity_scorer.score(instruction)
        instruction.update_evaluation(metrics, performance_objective, perplexity)
        return metrics

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
        metric_sum = sum(metrics.get(metric_name, 0.0) for metric_name in self.config.performance_metric_names)
        if metric_sum == 0:
            return float("inf")
        return 1.0 / metric_sum

    @staticmethod
    def _prediction_entropy(predictions: list[str]) -> float:
        counts = Counter(predictions)
        total = len(predictions)
        entropy = 0.0
        for count in counts.values():
            probability = count / total
            entropy -= probability * math.log(probability + 1e-12)
        return entropy
