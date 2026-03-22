from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import random
import statistics
from typing import Literal

from tqdm import tqdm

from instaoptima.config import ExperimentConfig
from instaoptima.data_loader import DatasetBundle
from instaoptima.data_loader import ExperimentDatasetLoader
from instaoptima.evaluator import InstructionEvaluator
from instaoptima.instruction import Instruction
from instaoptima.llm_client import LLMClient
from instaoptima.operators import EvolutionOperators
from instaoptima.pareto import pareto_front
from instaoptima.pareto import select_next_population
from instaoptima.population import PopulationFactory


OperatorName = Literal[
    "definition_mutation",
    "definition_crossover",
    "example_mutation",
    "example_crossover",
]


class InstaOptimaExperiment:
    def __init__(
        self,
        config: ExperimentConfig | None = None,
        artifact_root: str | Path | None = None,
    ) -> None:
        self.config = config or ExperimentConfig()
        self.llm_client = LLMClient(self.config)
        self.dataset_loader = ExperimentDatasetLoader(self.config)
        self.evaluator = InstructionEvaluator(self.llm_client, self.config)
        self.operators = EvolutionOperators(self.llm_client, self.config)
        self.population_factory = PopulationFactory(self.config)
        self.artifact_root = self._initialize_artifact_root(artifact_root)
        self._validate_or_write_config()

    def run(self, num_runs: int | None = None) -> None:
        completed_run_indices = self._existing_run_indices()
        run_scores = self._load_existing_run_scores(completed_run_indices)
        completed_runs = len(completed_run_indices)
        target_runs = self.config.num_runs

        if completed_runs >= target_runs:
            print(
                "All configured runs are already present in "
                f"{self.artifact_root} ({completed_runs}/{target_runs})."
            )
            self._log_run_summary(run_scores, target_runs=target_runs)
            self._write_summary_artifacts(run_scores, target_runs=target_runs)
            return

        requested_runs = target_runs - completed_runs if num_runs is None else num_runs
        if requested_runs <= 0:
            raise ValueError("Number of runs to execute must be greater than zero.")
        runs_to_execute = min(requested_runs, target_runs - completed_runs)
        if runs_to_execute < requested_runs:
            print(
                f"Requested {requested_runs} runs, but only {runs_to_execute} runs "
                f"remain before reaching num_runs={target_runs}."
            )

        dataset_bundle = self.dataset_loader.load()
        for run_index in range(completed_runs, completed_runs + runs_to_execute):
            print(f"\n========== Run {run_index + 1}/{target_runs} ==========")
            final_population, final_pareto = self._run_single_experiment(
                dataset_bundle,
                run_index,
            )
            self._log_pareto_front(final_pareto, title="Final Pareto Front")

            representative = max(
                final_pareto or final_population,
                key=lambda item: item.metrics.get("accuracy", 0.0),
            )
            run_scores.append(representative.metrics.get("accuracy", 0.0))
            print(
                "Representative metrics: "
                f"acc={representative.metrics.get('accuracy', 0.0):.4f}, "
                f"macro_f1={representative.metrics.get('macro_f1', 0.0):.4f}, "
                f"macro_precision={representative.metrics.get('macro_precision', 0.0):.4f}, "
                f"macro_recall={representative.metrics.get('macro_recall', 0.0):.4f}"
            )
            self._write_run_artifacts(
                run_index=run_index,
                final_population=final_population,
                final_pareto=final_pareto,
                representative=representative,
            )

        self._log_run_summary(run_scores, target_runs=target_runs)
        self._write_summary_artifacts(run_scores, target_runs=target_runs)

    def _run_single_experiment(
        self,
        dataset_bundle: DatasetBundle,
        run_index: int,
    ) -> tuple[list[Instruction], list[Instruction]]:
        random.seed(self.config.shuffle_seed + run_index)
        train_dataset = dataset_bundle.train
        test_dataset = dataset_bundle.test

        population = self.population_factory.create_initial_population(train_dataset)
        self._evaluate_population(
            population,
            train_dataset,
            test_dataset,
            desc="Evaluate initial population",
        )
        self._write_initial_population_artifacts(run_index, population)
        self._log_population(population, title="Initial Population")

        for generation in range(1, self.config.generations + 1):
            print(f"\n===== Generation {generation}/{self.config.generations} =====")
            offspring = self._generate_and_evaluate_offspring(
                population,
                train_dataset,
                test_dataset,
            )
            population, generation_pareto = select_next_population(
                population=population + offspring,
                population_size=self.config.population_size,
            )
            self._log_population(population, title="Selected Population")
            self._log_pareto_front(generation_pareto)

        return population, pareto_front(population)

    def _evaluate_population(
        self,
        population: list[Instruction],
        train_dataset,
        test_dataset,
        desc: str,
    ) -> None:
        for instruction in tqdm(population, desc=desc):
            self.evaluator.evaluate(
                instruction=instruction,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
            )

    def _generate_and_evaluate_offspring(
        self,
        population: list[Instruction],
        train_dataset,
        test_dataset,
    ) -> list[Instruction]:
        offspring: list[Instruction] = []
        for _ in tqdm(
            range(self.config.population_size),
            desc="Generate and evaluate offspring",
        ):
            first_parent = random.choice(population)
            second_parent = random.choice(population)
            operator = random.choice(self._available_operators())
            child = self._apply_operator(operator, first_parent, second_parent)
            self.evaluator.evaluate(
                instruction=child,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
            )
            offspring.append(child)

        return offspring

    @staticmethod
    def _available_operators() -> tuple[OperatorName, ...]:
        return (
            "definition_mutation",
            "definition_crossover",
            "example_mutation",
            "example_crossover",
        )

    def _apply_operator(
        self,
        operator: OperatorName,
        first_parent: Instruction,
        second_parent: Instruction,
    ) -> Instruction:
        if operator == "definition_mutation":
            return self.operators.mutate_definition(first_parent)
        if operator == "definition_crossover":
            return self.operators.crossover_definition(first_parent, second_parent)
        if operator == "example_mutation":
            return self.operators.mutate_example(first_parent)
        return self.operators.crossover_example(first_parent, second_parent)

    @staticmethod
    def _log_population(population: list[Instruction], title: str) -> None:
        print(f"\n{title}:")
        for instruction in population:
            print(
                "PerfObj: "
                f"{instruction.objectives.performance:.6f}, "
                f"Len: {instruction.objectives.length:.1f}, "
                f"PPL: {instruction.objectives.perplexity:.6f}, "
                f"Acc: {instruction.metrics.get('accuracy', 0.0):.4f}, "
                f"Def: {instruction.definition[:60]}"
            )

    @staticmethod
    def _log_pareto_front(
        pareto_population: list[Instruction],
        title: str = "Pareto Front",
    ) -> None:
        print(f"\n{title}:")
        for instruction in pareto_population:
            print(
                "PerfObj: "
                f"{instruction.objectives.performance:.6f}, "
                f"Len: {instruction.objectives.length:.1f}, "
                f"PPL: {instruction.objectives.perplexity:.6f}, "
                f"Acc: {instruction.metrics.get('accuracy', 0.0):.4f}"
            )

    @staticmethod
    def _log_run_summary(run_scores: list[float], target_runs: int | None = None) -> None:
        if not run_scores:
            return

        mean_score = statistics.mean(run_scores)
        std_score = statistics.stdev(run_scores) if len(run_scores) > 1 else 0.0
        total_label = (
            f"{len(run_scores)}/{target_runs}"
            if target_runs is not None
            else str(len(run_scores))
        )
        print(
            "\n===== Summary =====\n"
            f"Representative accuracy over {total_label} runs: "
            f"{mean_score * 100:.2f} +- {std_score * 100:.2f}"
        )

    def _create_artifact_root(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_slug = self._slugify(self.config.dataset_name)
        task_slug = self._slugify(self.config.task_type)
        artifact_root = Path("artifacts") / f"{dataset_slug}_{task_slug}_{timestamp}"
        artifact_root.mkdir(parents=True, exist_ok=True)
        return artifact_root

    def _initialize_artifact_root(self, artifact_root: str | Path | None) -> Path:
        if artifact_root is None:
            return self._create_artifact_root()
        root = Path(artifact_root)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _validate_or_write_config(self) -> None:
        config_path = self.artifact_root / "config.json"
        current_config = asdict(self.config)
        if not config_path.exists():
            self._write_json(config_path, current_config)
            return

        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if self._normalized_config(existing_config) != self._normalized_config(
            current_config
        ):
            raise ValueError(
                "The provided artifact root was created with a different config. "
                f"Artifact root: {self.artifact_root}"
            )

    def _existing_run_indices(self) -> list[int]:
        indices: list[int] = []
        for run_dir in self.artifact_root.glob("run_*"):
            if not run_dir.is_dir():
                continue
            try:
                run_index = int(run_dir.name.split("_", maxsplit=1)[1])
            except (IndexError, ValueError):
                continue
            if (run_dir / "metrics.json").exists():
                indices.append(run_index)

        indices.sort()
        expected = list(range(1, len(indices) + 1))
        if indices != expected:
            raise ValueError(
                "Artifact root contains missing or non-contiguous runs. "
                f"Found run indices: {indices}"
            )
        return indices

    def _load_existing_run_scores(self, run_indices: list[int]) -> list[float]:
        run_scores: list[float] = []
        for run_index in run_indices:
            metrics_path = self.artifact_root / f"run_{run_index}" / "metrics.json"
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            representative = payload.get("representative", {})
            metrics = representative.get("metrics", {})
            accuracy = metrics.get("accuracy")
            if accuracy is None:
                raise ValueError(
                    f"Missing representative accuracy in {metrics_path}."
                )
            run_scores.append(float(accuracy))
        return run_scores

    def _write_run_artifacts(
        self,
        run_index: int,
        final_population: list[Instruction],
        final_pareto: list[Instruction],
        representative: Instruction,
    ) -> None:
        run_dir = self.artifact_root / f"run_{run_index + 1}"
        run_dir.mkdir(parents=True, exist_ok=True)

        run_summary = {
            "run_index": run_index + 1,
            "population_size": len(final_population),
            "pareto_size": len(final_pareto),
            "representative": self._serialize_instruction(representative),
        }
        self._write_json(run_dir / "metrics.json", run_summary)
        self._write_json(
            run_dir / "pareto_front.json",
            [self._serialize_instruction(instruction) for instruction in final_pareto],
        )
        self._write_json(
            run_dir / "final_population.json",
            [self._serialize_instruction(instruction) for instruction in final_population],
        )
        (run_dir / "best_instruction.txt").write_text(
            self._format_instruction_text(representative),
            encoding="utf-8",
        )

    def _write_initial_population_artifacts(
        self,
        run_index: int,
        population: list[Instruction],
    ) -> None:
        run_dir = self.artifact_root / f"run_{run_index + 1}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(
            run_dir / "initial_population.json",
            [self._serialize_instruction(instruction) for instruction in population],
        )

    def _write_summary_artifacts(
        self,
        run_scores: list[float],
        target_runs: int | None = None,
    ) -> None:
        if not run_scores:
            return

        mean_score = statistics.mean(run_scores)
        std_score = statistics.stdev(run_scores) if len(run_scores) > 1 else 0.0
        summary = {
            "num_runs": len(run_scores),
            "target_num_runs": target_runs if target_runs is not None else len(run_scores),
            "representative_accuracy_scores": run_scores,
            "mean_accuracy": mean_score,
            "std_accuracy": std_score,
        }
        self._write_json(self.artifact_root / "summary.json", summary)

    @staticmethod
    def _normalized_config(payload: dict[str, object]) -> dict[str, object]:
        normalized = dict(payload)
        normalized.pop("openai_api_key", None)
        return normalized

    def _serialize_instruction(self, instruction: Instruction) -> dict[str, object]:
        return {
            "definition": instruction.definition,
            "examples": [
                {
                    "text": example.text,
                    "label": example.label,
                    "aspect": example.aspect,
                }
                for example in instruction.examples
            ],
            "objectives": {
                "performance": self._normalize_json_float(
                    instruction.objectives.performance
                ),
                "length": self._normalize_json_float(instruction.objectives.length),
                "perplexity": self._normalize_json_float(
                    instruction.objectives.perplexity
                ),
            },
            "metrics": {
                key: self._normalize_json_float(value)
                for key, value in instruction.metrics.items()
            },
            "crowding_distance": self._normalize_json_float(
                instruction.crowding_distance
            ),
            "rank": instruction.rank,
        }

    def _format_instruction_text(self, instruction: Instruction) -> str:
        lines = [
            "Definition:",
            instruction.definition,
            "",
            "Examples:",
        ]
        for index, example in enumerate(instruction.examples, start=1):
            lines.append(f"{index}. Sentence: {example.text}")
            if example.aspect:
                lines.append(f"   Aspect: {example.aspect}")
            lines.append(f"   Label: {example.label}")
        lines.extend(
            [
                "",
                "Objectives:",
                f"- performance: {instruction.objectives.performance}",
                f"- length: {instruction.objectives.length}",
                f"- perplexity: {instruction.objectives.perplexity}",
                "",
                "Metrics:",
            ]
        )
        for key, value in sorted(instruction.metrics.items()):
            lines.append(f"- {key}: {value}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _normalize_json_float(value: float) -> float | None:
        return value if math.isfinite(value) else None

    @staticmethod
    def _slugify(raw_value: str) -> str:
        normalized = "".join(
            character if character.isalnum() else "_"
            for character in raw_value.strip().lower()
        )
        return normalized.strip("_") or "experiment"
