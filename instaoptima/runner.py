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
        run_stats = self._load_existing_run_stats(completed_run_indices)
        completed_runs = len(completed_run_indices)
        target_runs = self.config.num_runs

        if completed_runs >= target_runs:
            print(
                "All configured runs are already present in "
                f"{self.artifact_root} ({completed_runs}/{target_runs})."
            )
            self._log_run_summary(run_stats, target_runs=target_runs)
            self._write_summary_artifacts(run_stats, target_runs=target_runs)
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
            run_stats.append(
                {
                    "accuracy": float(representative.metrics.get("accuracy", 0.0)),
                    "length": float(representative.objectives.length),
                    "perplexity": float(representative.objectives.perplexity),
                }
            )
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

        self._log_run_summary(run_stats, target_runs=target_runs)
        self._write_summary_artifacts(run_stats, target_runs=target_runs)

    def _run_single_experiment(
        self,
        dataset_bundle: DatasetBundle,
        run_index: int,
    ) -> tuple[list[Instruction], list[Instruction]]:
        random.seed(self.config.shuffle_seed + run_index)
        train_dataset = dataset_bundle.train
        test_dataset = dataset_bundle.test

        population = self.population_factory.create_initial_population(
            train_dataset,
            run_seed=self.config.shuffle_seed + run_index,
        )
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
            combined_population = population + offspring
            self._write_generation_union_artifacts(
                run_index=run_index,
                generation=generation,
                combined_population=combined_population,
            )
            population, generation_pareto = select_next_population(
                population=combined_population,
                population_size=self.config.population_size,
                random_replacement_ratio=self.config.random_replacement_ratio,
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
    def _log_run_summary(
        run_stats: list[dict[str, float]],
        target_runs: int | None = None,
    ) -> None:
        if not run_stats:
            return

        accuracy_values = [
            stat["accuracy"]
            for stat in run_stats
            if math.isfinite(stat.get("accuracy", float("nan")))
        ]
        length_values = [
            stat["length"]
            for stat in run_stats
            if math.isfinite(stat.get("length", float("nan")))
        ]
        perplexity_values = [
            stat["perplexity"]
            for stat in run_stats
            if math.isfinite(stat.get("perplexity", float("nan")))
        ]

        mean_accuracy = statistics.mean(accuracy_values)
        std_accuracy = (
            statistics.stdev(accuracy_values) if len(accuracy_values) > 1 else 0.0
        )
        mean_length = statistics.mean(length_values)
        std_length = statistics.stdev(length_values) if len(length_values) > 1 else 0.0
        mean_perplexity = statistics.mean(perplexity_values)
        std_perplexity = (
            statistics.stdev(perplexity_values) if len(perplexity_values) > 1 else 0.0
        )

        total_label = (
            f"{len(run_stats)}/{target_runs}"
            if target_runs is not None
            else str(len(run_stats))
        )
        print(
            "\n===== Summary =====\n"
            f"Representative accuracy over {total_label} runs: "
            f"{mean_accuracy * 100:.2f} +- {std_accuracy * 100:.2f}\n"
            f"Representative length over {total_label} runs: "
            f"{mean_length:.2f} +- {std_length:.2f}\n"
            f"Representative perplexity over {total_label} runs: "
            f"{mean_perplexity:.4f} +- {std_perplexity:.4f}"
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

    def _load_existing_run_stats(self, run_indices: list[int]) -> list[dict[str, float]]:
        run_stats: list[dict[str, float]] = []
        for run_index in run_indices:
            metrics_path = self.artifact_root / f"run_{run_index}" / "metrics.json"
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            representative = payload.get("representative", {})
            metrics = representative.get("metrics", {})
            objectives = representative.get("objectives", {})
            accuracy = metrics.get("accuracy")
            if accuracy is None:
                raise ValueError(
                    f"Missing representative accuracy in {metrics_path}."
                )
            length = objectives.get("length")
            if length is None:
                raise ValueError(
                    f"Missing representative length in {metrics_path}."
                )
            perplexity = objectives.get("perplexity")
            if perplexity is None:
                raise ValueError(
                    f"Missing representative perplexity in {metrics_path}."
                )

            run_stats.append(
                {
                    "accuracy": float(accuracy),
                    "length": float(length),
                    "perplexity": float(perplexity),
                }
            )
        return run_stats

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

    def _write_generation_union_artifacts(
        self,
        run_index: int,
        generation: int,
        combined_population: list[Instruction],
    ) -> None:
        run_dir = self.artifact_root / f"run_{run_index + 1}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(
            run_dir / f"generation_{generation}_union.json",
            [self._serialize_instruction(instruction) for instruction in combined_population],
        )

    def _write_summary_artifacts(
        self,
        run_stats: list[dict[str, float]],
        target_runs: int | None = None,
    ) -> None:
        if not run_stats:
            return

        accuracy_values = [
            stat["accuracy"]
            for stat in run_stats
            if math.isfinite(stat.get("accuracy", float("nan")))
        ]
        length_values = [
            stat["length"]
            for stat in run_stats
            if math.isfinite(stat.get("length", float("nan")))
        ]
        perplexity_values = [
            stat["perplexity"]
            for stat in run_stats
            if math.isfinite(stat.get("perplexity", float("nan")))
        ]

        mean_accuracy = statistics.mean(accuracy_values)
        std_accuracy = (
            statistics.stdev(accuracy_values) if len(accuracy_values) > 1 else 0.0
        )
        mean_length = statistics.mean(length_values)
        std_length = statistics.stdev(length_values) if len(length_values) > 1 else 0.0
        mean_perplexity = statistics.mean(perplexity_values)
        std_perplexity = (
            statistics.stdev(perplexity_values) if len(perplexity_values) > 1 else 0.0
        )
        summary = {
            "num_runs": len(run_stats),
            "target_num_runs": target_runs if target_runs is not None else len(run_stats),
            "representative_accuracy_scores": accuracy_values,
            "representative_length_scores": length_values,
            "representative_perplexity_scores": perplexity_values,
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "mean_length": mean_length,
            "std_length": std_length,
            "mean_perplexity": mean_perplexity,
            "std_perplexity": std_perplexity,
        }

        pareto_points = self._collect_pareto_points()
        figure_path = self._write_tradeoff_figure(pareto_points)
        representative_run_index = self._select_representative_run_index()
        generation_union_points = self._collect_generation_union_points(
            run_index=representative_run_index
        )
        final_population_points = self._collect_final_population_points(
            run_index=representative_run_index
        )
        front_points = generation_union_points or final_population_points
        front_point_source = (
            "generation_union"
            if generation_union_points
            else "final_population"
        )
        front_figure_path = self._write_fronts_tradeoff_figure(
            front_points,
            max_fronts=3,
        )
        report_path = self._write_tradeoff_report(
            pareto_points=pareto_points,
            summary=summary,
            target_runs=target_runs,
            figure_path=front_figure_path,
        )
        summary["pareto_point_count"] = len(pareto_points)
        summary["generation_union_point_count"] = len(generation_union_points)
        summary["final_population_point_count"] = len(final_population_points)
        summary["front_figure_point_count"] = len(front_points)
        summary["front_figure_point_source"] = front_point_source
        summary["front_figure_run_index"] = representative_run_index
        summary["fronts_plotted"] = 3
        summary["tradeoff_figure"] = figure_path.name if figure_path else None
        summary["tradeoff_fronts_figure"] = (
            front_figure_path.name if front_figure_path else None
        )
        summary["tradeoff_report"] = report_path.name if report_path else None

        self._write_json(self.artifact_root / "summary.json", summary)

    def _collect_pareto_points(self) -> list[dict[str, float | int]]:
        points: list[dict[str, float | int]] = []
        for run_dir in sorted(self.artifact_root.glob("run_*")):
            if not run_dir.is_dir():
                continue
            try:
                run_index = int(run_dir.name.split("_", maxsplit=1)[1])
            except (IndexError, ValueError):
                continue

            pareto_path = run_dir / "pareto_front.json"
            if not pareto_path.exists():
                continue

            payload = json.loads(pareto_path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                continue

            for item in payload:
                if not isinstance(item, dict):
                    continue
                metrics = item.get("metrics", {})
                objectives = item.get("objectives", {})
                examples = item.get("examples", [])
                if not isinstance(metrics, dict) or not isinstance(objectives, dict):
                    continue

                accuracy = metrics.get("accuracy")
                length = objectives.get("length")
                perplexity = objectives.get("perplexity")
                if accuracy is None or length is None or perplexity is None:
                    continue
                if not (
                    math.isfinite(float(accuracy))
                    and math.isfinite(float(length))
                    and math.isfinite(float(perplexity))
                ):
                    continue

                points.append(
                    {
                        "run_index": run_index,
                        "accuracy": float(accuracy),
                        "length": float(length),
                        "perplexity": float(perplexity),
                        "num_examples": int(len(examples)) if isinstance(examples, list) else 0,
                    }
                )
        return points

    def _write_tradeoff_figure(
        self,
        pareto_points: list[dict[str, float | int]],
    ) -> Path | None:
        if not pareto_points:
            return None

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not available; skip trade-off figure generation.")
            return None

        run_indices = sorted(
            {
                int(point["run_index"])
                for point in pareto_points
            }
        )
        cmap = plt.cm.get_cmap("tab10", max(1, len(run_indices)))
        color_by_run = {
            run_idx: cmap(position)
            for position, run_idx in enumerate(run_indices)
        }

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        for point in pareto_points:
            run_idx = int(point["run_index"])
            acc = float(point["accuracy"])
            length = float(point["length"])
            ppl = float(point["perplexity"])
            color = color_by_run[run_idx]
            label = f"run_{run_idx}"

            axes[0].scatter(acc, length, color=color, alpha=0.85, s=48, label=label)
            axes[1].scatter(acc, ppl, color=color, alpha=0.85, s=48, label=label)
            axes[2].scatter(length, ppl, color=color, alpha=0.85, s=48, label=label)

        axes[0].set_xlabel("Accuracy (higher is better)")
        axes[0].set_ylabel("Length (lower is better)")
        axes[0].set_title("Accuracy vs Length")
        axes[0].grid(alpha=0.25)

        axes[1].set_xlabel("Accuracy (higher is better)")
        axes[1].set_ylabel("Perplexity (lower is better)")
        axes[1].set_title("Accuracy vs PPL")
        axes[1].grid(alpha=0.25)

        axes[2].set_xlabel("Length (lower is better)")
        axes[2].set_ylabel("Perplexity (lower is better)")
        axes[2].set_title("Length vs PPL")
        axes[2].grid(alpha=0.25)

        handles, labels = axes[0].get_legend_handles_labels()
        unique_handles = {}
        for handle, label in zip(handles, labels):
            unique_handles[label] = handle
        if unique_handles:
            fig.legend(
                unique_handles.values(),
                unique_handles.keys(),
                loc="upper center",
                ncol=min(len(unique_handles), 6),
            )

        fig.suptitle(
            "Pareto Trade-off (Fig.3-style): Accuracy, Length, Perplexity",
            y=1.02,
        )

        output_path = self.artifact_root / "tradeoff_fig3_like.png"
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def _collect_final_population_points(
        self,
        run_index: int | None = None,
    ) -> list[dict[str, float | int]]:
        points: list[dict[str, float | int]] = []
        for run_dir in sorted(self.artifact_root.glob("run_*")):
            if not run_dir.is_dir():
                continue
            try:
                parsed_run_index = int(run_dir.name.split("_", maxsplit=1)[1])
            except (IndexError, ValueError):
                continue

            if run_index is not None and parsed_run_index != run_index:
                continue

            population_path = run_dir / "final_population.json"
            if not population_path.exists():
                continue

            payload = json.loads(population_path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                continue

            for item in payload:
                if not isinstance(item, dict):
                    continue
                metrics = item.get("metrics", {})
                objectives = item.get("objectives", {})
                examples = item.get("examples", [])
                if not isinstance(metrics, dict) or not isinstance(objectives, dict):
                    continue

                performance = objectives.get("performance")
                length = objectives.get("length")
                perplexity = objectives.get("perplexity")
                accuracy = metrics.get("accuracy")
                if (
                    performance is None
                    or length is None
                    or perplexity is None
                    or accuracy is None
                ):
                    continue
                if not (
                    math.isfinite(float(performance))
                    and math.isfinite(float(length))
                    and math.isfinite(float(perplexity))
                    and math.isfinite(float(accuracy))
                ):
                    continue

                points.append(
                    {
                        "run_index": parsed_run_index,
                        "performance": float(performance),
                        "length": float(length),
                        "perplexity": float(perplexity),
                        "accuracy": float(accuracy),
                        "num_examples": int(len(examples)) if isinstance(examples, list) else 0,
                    }
                )
        return points

    def _collect_generation_union_points(
        self,
        run_index: int | None = None,
    ) -> list[dict[str, float | int]]:
        points: list[dict[str, float | int]] = []
        for run_dir in sorted(self.artifact_root.glob("run_*")):
            if not run_dir.is_dir():
                continue
            try:
                parsed_run_index = int(run_dir.name.split("_", maxsplit=1)[1])
            except (IndexError, ValueError):
                continue

            if run_index is not None and parsed_run_index != run_index:
                continue

            for union_path in sorted(run_dir.glob("generation_*_union.json")):
                payload = json.loads(union_path.read_text(encoding="utf-8"))
                if not isinstance(payload, list):
                    continue
                for item in payload:
                    if not isinstance(item, dict):
                        continue
                    metrics = item.get("metrics", {})
                    objectives = item.get("objectives", {})
                    examples = item.get("examples", [])
                    if not isinstance(metrics, dict) or not isinstance(objectives, dict):
                        continue

                    performance = objectives.get("performance")
                    length = objectives.get("length")
                    perplexity = objectives.get("perplexity")
                    accuracy = metrics.get("accuracy")
                    if (
                        performance is None
                        or length is None
                        or perplexity is None
                        or accuracy is None
                    ):
                        continue
                    if not (
                        math.isfinite(float(performance))
                        and math.isfinite(float(length))
                        and math.isfinite(float(perplexity))
                        and math.isfinite(float(accuracy))
                    ):
                        continue

                    points.append(
                        {
                            "run_index": parsed_run_index,
                            "performance": float(performance),
                            "length": float(length),
                            "perplexity": float(perplexity),
                            "accuracy": float(accuracy),
                            "num_examples": int(len(examples)) if isinstance(examples, list) else 0,
                        }
                    )
        return points

    def _select_representative_run_index(self) -> int | None:
        run_candidates: list[tuple[int, float]] = []
        for run_dir in sorted(self.artifact_root.glob("run_*")):
            if not run_dir.is_dir():
                continue
            try:
                run_index = int(run_dir.name.split("_", maxsplit=1)[1])
            except (IndexError, ValueError):
                continue

            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                continue

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            representative = payload.get("representative", {})
            metrics = representative.get("metrics", {})
            accuracy = metrics.get("accuracy")
            if accuracy is None:
                continue
            run_candidates.append((run_index, float(accuracy)))

        if not run_candidates:
            return None

        run_candidates.sort(key=lambda item: (-item[1], item[0]))
        return run_candidates[0][0]

    @staticmethod
    def _point_dominates(
        candidate: dict[str, float | int],
        competitor: dict[str, float | int],
    ) -> bool:
        candidate_values = (
            float(candidate["performance"]),
            float(candidate["length"]),
            float(candidate["perplexity"]),
        )
        competitor_values = (
            float(competitor["performance"]),
            float(competitor["length"]),
            float(competitor["perplexity"]),
        )
        return (
            all(c <= o for c, o in zip(candidate_values, competitor_values))
            and any(c < o for c, o in zip(candidate_values, competitor_values))
        )

    def _non_dominated_fronts_from_points(
        self,
        points: list[dict[str, float | int]],
        max_fronts: int = 3,
    ) -> list[list[dict[str, float | int]]]:
        if not points:
            return []

        remaining = list(points)
        fronts: list[list[dict[str, float | int]]] = []
        while remaining and len(fronts) < max_fronts:
            current_front: list[dict[str, float | int]] = []
            for point in remaining:
                is_dominated = False
                for other in remaining:
                    if point is other:
                        continue
                    if self._point_dominates(other, point):
                        is_dominated = True
                        break
                if not is_dominated:
                    current_front.append(point)

            if not current_front:
                break

            fronts.append(current_front)
            remaining_ids = {id(point) for point in current_front}
            remaining = [point for point in remaining if id(point) not in remaining_ids]

        return fronts

    def _write_fronts_tradeoff_figure(
        self,
        final_population_points: list[dict[str, float | int]],
        max_fronts: int = 2,
    ) -> Path | None:
        if not final_population_points:
            return None

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not available; skip fronts-based trade-off figure generation.")
            return None

        fronts = self._non_dominated_fronts_from_points(
            final_population_points,
            max_fronts=max_fronts,
        )
        if not fronts:
            return None

        colors = ["red", "green", "blue"]
        labels = ["Front 1", "Front 2", "Front 3"]

        fig, axes = plt.subplots(3, 1, figsize=(6, 12), constrained_layout=True)
        for front_index, front in enumerate(fronts):
            color = colors[front_index]
            label = labels[front_index]
            for point in front:
                performance = float(point["performance"])
                length = float(point["length"])
                perplexity = float(point["perplexity"])

                axes[0].scatter(
                    performance,
                    perplexity,
                    color=color,
                    s=28,
                    alpha=0.9,
                    label=label,
                )
                axes[1].scatter(
                    performance,
                    length,
                    color=color,
                    s=28,
                    alpha=0.9,
                    label=label,
                )
                axes[2].scatter(
                    perplexity,
                    length,
                    color=color,
                    s=28,
                    alpha=0.9,
                    label=label,
                )

        axes[0].set_xlabel("Performance")
        axes[0].set_ylabel("Perplexity")
        axes[0].grid(alpha=0.25)

        axes[1].set_xlabel("Performance")
        axes[1].set_ylabel("Length")
        axes[1].grid(alpha=0.25)

        axes[2].set_xlabel("Perplexity")
        axes[2].set_ylabel("Length")
        axes[2].grid(alpha=0.25)

        for axis in axes:
            handles, handle_labels = axis.get_legend_handles_labels()
            unique_handles: dict[str, object] = {}
            for handle, handle_label in zip(handles, handle_labels):
                if handle_label not in unique_handles:
                    unique_handles[handle_label] = handle
            if unique_handles:
                axis.legend(unique_handles.values(), unique_handles.keys(), loc="best")

        front_label = "Front 1-2" if max_fronts == 2 else f"Front 1-{max_fronts}"
        fig.suptitle(f"Fig.3-style 2D Pareto Fronts ({front_label})", y=1.01)
        output_path = self.artifact_root / "tradeoff_fig3_fronts.png"
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def _write_tradeoff_report(
        self,
        pareto_points: list[dict[str, float | int]],
        summary: dict[str, object],
        target_runs: int | None,
        figure_path: Path | None,
    ) -> Path | None:
        if not pareto_points:
            return None

        acc_values = [float(point["accuracy"]) for point in pareto_points]
        length_values = [float(point["length"]) for point in pareto_points]
        ppl_values = [float(point["perplexity"]) for point in pareto_points]
        example_counts = [int(point.get("num_examples", 0)) for point in pareto_points]

        def _pearson(x: list[float], y: list[float]) -> float:
            if len(x) != len(y) or len(x) < 2:
                return 0.0
            mean_x = statistics.mean(x)
            mean_y = statistics.mean(y)
            numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
            denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
            denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
            denominator = denom_x * denom_y
            if denominator == 0:
                return 0.0
            return numerator / denominator

        acc_len_corr = _pearson(acc_values, length_values)
        acc_ppl_corr = _pearson(acc_values, ppl_values)
        len_ppl_corr = _pearson(length_values, ppl_values)

        shot_distribution: dict[int, int] = {}
        length_by_shot: dict[int, list[float]] = {}
        for point in pareto_points:
            num_examples = int(point.get("num_examples", 0))
            shot_distribution[num_examples] = shot_distribution.get(num_examples, 0) + 1
            length_by_shot.setdefault(num_examples, []).append(float(point["length"]))

        run_count = int(summary.get("num_runs", len({int(p["run_index"]) for p in pareto_points})))
        target = target_runs if target_runs is not None else run_count

        lines = [
            "# Trade-off Report",
            "",
            f"Runs aggregated: {run_count}/{target}",
            f"Pareto points used: {len(pareto_points)}",
            "",
            "## Representative Metrics (across runs)",
            (
                "- Accuracy: "
                f"{float(summary['mean_accuracy']) * 100:.2f} +- {float(summary['std_accuracy']) * 100:.2f}"
            ),
            (
                "- Length: "
                f"{float(summary['mean_length']):.2f} +- {float(summary['std_length']):.2f}"
            ),
            (
                "- Perplexity: "
                f"{float(summary['mean_perplexity']):.4f} +- {float(summary['std_perplexity']):.4f}"
            ),
            "",
            "## Pareto Point Ranges",
            f"- Accuracy min/max: {min(acc_values):.4f} / {max(acc_values):.4f}",
            f"- Length min/max: {min(length_values):.2f} / {max(length_values):.2f}",
            f"- Perplexity min/max: {min(ppl_values):.4f} / {max(ppl_values):.4f}",
            "",
            "## Pairwise Trade-off Correlation (Pearson)",
            f"- Accuracy vs Length: {acc_len_corr:.4f}",
            f"- Accuracy vs Perplexity: {acc_ppl_corr:.4f}",
            f"- Length vs Perplexity: {len_ppl_corr:.4f}",
            "",
            "## Example Count Distribution on Pareto Points",
            f"- Mean #examples: {statistics.mean(example_counts):.2f}",
        ]

        for num_examples in sorted(shot_distribution):
            point_count = shot_distribution[num_examples]
            share = point_count / len(pareto_points) * 100.0
            lines.append(
                f"- {num_examples} example(s): {point_count} point(s) ({share:.1f}%)"
            )

        lines.extend(
            [
                "",
                "## Length by Example Count",
            ]
        )
        for num_examples in sorted(length_by_shot):
            lengths = length_by_shot[num_examples]
            mean_len = statistics.mean(lengths)
            std_len = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
            lines.append(
                f"- {num_examples} example(s): {mean_len:.2f} +- {std_len:.2f}"
            )

        lines.extend(
            [
                "",
            "## Artifacts",
            (
                f"- Figure (paper-style fronts): {figure_path.name}"
                if figure_path is not None
                else "- Figure (paper-style fronts): not generated"
            ),
            ]
        )

        output_path = self.artifact_root / "tradeoff_report.md"
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

    @staticmethod
    def _normalized_config(payload: dict[str, object]) -> dict[str, object]:
        normalized = dict(payload)
        normalized.pop("openai_api_key", None)
        return json.loads(json.dumps(normalized, ensure_ascii=False, sort_keys=True))

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
