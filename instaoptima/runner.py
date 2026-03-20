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
    def __init__(self, config: ExperimentConfig | None = None) -> None:
        self.config = config or ExperimentConfig()
        self.llm_client = LLMClient(self.config)
        self.dataset_loader = ExperimentDatasetLoader(self.config)
        self.evaluator = InstructionEvaluator(self.llm_client, self.config)
        self.operators = EvolutionOperators(self.llm_client, self.config)
        self.population_factory = PopulationFactory(self.config)

    def run(self) -> None:
        dataset_bundle = self.dataset_loader.load()
        run_scores: list[float] = []

        for run_index in range(self.config.num_runs):
            print(f"\n========== Run {run_index + 1}/{self.config.num_runs} ==========")
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

        self._log_run_summary(run_scores)

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
    def _log_run_summary(run_scores: list[float]) -> None:
        if not run_scores:
            return

        mean_score = statistics.mean(run_scores)
        std_score = statistics.stdev(run_scores) if len(run_scores) > 1 else 0.0
        print(
            "\n===== Summary =====\n"
            f"Representative accuracy over {len(run_scores)} runs: "
            f"{mean_score * 100:.2f} +- {std_score * 100:.2f}"
        )
