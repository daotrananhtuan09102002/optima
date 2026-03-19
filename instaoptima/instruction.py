from dataclasses import dataclass, field

from instaoptima.config import ExperimentConfig


@dataclass(frozen=True)
class TaskExample:
    text: str
    label: str
    aspect: str | None = None


@dataclass(frozen=True)
class ObjectiveVector:
    performance: float = float("inf")
    length: float = float("inf")
    perplexity: float = float("inf")

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.performance, self.length, self.perplexity)


@dataclass
class Instruction:
    definition: str
    examples: list[TaskExample]
    objectives: ObjectiveVector = field(default_factory=ObjectiveVector)
    metrics: dict[str, float] = field(default_factory=dict)
    crowding_distance: float = 0.0
    rank: int = 0

    def build_prompt(self, example: TaskExample, config: ExperimentConfig) -> str:
        example_block = "\n\n".join(
            self._format_example(example_item, include_label=True, config=config)
            for example_item in self.examples
        )
        label_hint = self._label_hint(config)
        query_block = self._format_example(example, include_label=False, config=config)
        return (
            f"{self.definition}\n\n"
            f"{example_block}\n\n"
            f"{query_block}\n"
            f"Label (only output one of: {label_hint}):"
        )

    def update_evaluation(
        self,
        metrics: dict[str, float],
        performance_objective: float,
        perplexity: float,
    ) -> None:
        self.metrics = metrics
        self.objectives = ObjectiveVector(
            performance=performance_objective,
            length=float(self.character_length),
            perplexity=perplexity,
        )

    @property
    def character_length(self) -> int:
        return len(self.definition)

    def objective_summary(self) -> str:
        return (
            f"performance={self.objectives.performance:.6f}, "
            f"length={self.objectives.length:.1f}, "
            f"perplexity={self.objectives.perplexity:.6f}"
        )

    @staticmethod
    def _format_example(
        example: TaskExample,
        include_label: bool,
        config: ExperimentConfig,
    ) -> str:
        lines = [f"Sentence: {example.text}"]
        if config.task_type == "absa" and example.aspect:
            lines.append(f"Aspect: {example.aspect}")
        if include_label:
            lines.append(f"Label: {example.label}")
        return "\n".join(lines)

    @staticmethod
    def _label_hint(config: ExperimentConfig) -> str:
        if config.label_space:
            return ", ".join(config.label_space)
        return "positive, negative"
