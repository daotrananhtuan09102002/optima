import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from tqdm import tqdm

from instaoptima.config import ExperimentConfig
from instaoptima.data_loader import DatasetBundle
from instaoptima.data_loader import ExperimentDatasetLoader
from instaoptima.evaluator import InstructionEvaluator
from instaoptima.instruction import Instruction
from instaoptima.instruction import TaskExample
from instaoptima.llm_client import LLMClient
from instaoptima.operators import EvolutionOperators
from instaoptima.pareto import pareto_front
from instaoptima.pareto import select_next_population
from instaoptima.population import PopulationFactory


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
        run_scores = []

        for run_index in range(self.config.num_runs):
            print(f"\n========== Run {run_index + 1}/{self.config.num_runs} ==========")
            best_instruction = self._run_single_experiment(dataset_bundle, run_index)
            test_metrics = self.evaluator.evaluate(
                best_instruction,
                dataset_bundle.get_split(self.config.report_split),
            )
            run_scores.append(test_metrics["accuracy"])
            print(
                "Final test metrics: "
                f"acc={test_metrics['accuracy']:.4f}, "
                f"macro_f1={test_metrics['macro_f1']:.4f}, "
                f"macro_precision={test_metrics['macro_precision']:.4f}, "
                f"macro_recall={test_metrics['macro_recall']:.4f}"
            )

        self._log_run_summary(run_scores)

    def _run_single_experiment(
        self,
        dataset_bundle: DatasetBundle,
        run_index: int,
    ) -> Instruction:
        random.seed(self.config.shuffle_seed + run_index)
        population = self.population_factory.create_initial_population(dataset_bundle.train)
        optimization_dataset = dataset_bundle.get_split(self.config.optimization_split)

        for generation in range(self.config.generations):
            print(f"\n===== Generation {generation} =====")
            generation_dataset = self._sample_optimization_dataset(optimization_dataset)

            for instruction in tqdm(population):
                self.evaluator.evaluate(instruction, generation_dataset)

            self._log_population(population)

            pareto_population = pareto_front(population)
            self._log_pareto_front(pareto_population)

            offspring = self._generate_offspring(pareto_population)
            for child in tqdm(offspring, desc="Evaluate offspring"):
                self.evaluator.evaluate(child, generation_dataset)

            population, _ = select_next_population(
                population=population + offspring,
                population_size=self.config.population_size,
                random_replacement_ratio=self.config.random_replacement_ratio,
            )

        best_instruction = min(
            population,
            key=lambda item: (
                item.objectives.performance,
                item.objectives.perplexity,
                item.objectives.length,
            ),
        )
        return best_instruction

    def _sample_optimization_dataset(
        self,
        optimization_dataset: list[TaskExample],
    ) -> list[TaskExample]:
        sample_size = self.config.optimization_eval_sample_size
        if sample_size is None:
            return optimization_dataset

        bounded_size = max(1, min(int(sample_size), len(optimization_dataset)))
        if bounded_size >= len(optimization_dataset):
            return optimization_dataset
        return random.sample(optimization_dataset, bounded_size)

    def _generate_offspring(self, pareto_population: list[Instruction]) -> list[Instruction]:
        jobs = self._sample_offspring_jobs(pareto_population)
        if not jobs:
            return []

        worker_count = max(1, int(self.config.operator_parallel_workers))
        batch_size = max(1, int(self.config.operator_batch_size))

        # Preserve compatibility with deterministic, sequential generation when requested.
        if worker_count == 1:
            return [
                self._apply_offspring_job(job)
                for job in tqdm(jobs, desc="Generate offspring")
            ]

        offspring: list[Instruction] = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for start_idx in range(0, len(jobs), batch_size):
                batch = jobs[start_idx : start_idx + batch_size]
                for child in tqdm(
                    executor.map(self._apply_offspring_job, batch),
                    total=len(batch),
                    desc="Generate offspring",
                ):
                    offspring.append(child)

        return offspring

    def _sample_offspring_jobs(
        self,
        pareto_population: list[Instruction],
    ) -> list[tuple[Literal["definition_mutation", "definition_crossover", "example_mutation", "example_crossover"], Instruction, Instruction | None]]:
        jobs = []
        operators: tuple[
            Literal[
                "definition_mutation",
                "definition_crossover",
                "example_mutation",
                "example_crossover",
            ],
            ...,
        ] = (
            "definition_mutation",
            "definition_crossover",
            "example_mutation",
            "example_crossover",
        )

        while len(jobs) < self.config.population_size:
            parent = random.choice(pareto_population)
            operator = random.choice(operators)
            partner = (
                random.choice(pareto_population)
                if operator in {"definition_crossover", "example_crossover"}
                else None
            )
            jobs.append((operator, parent, partner))

        return jobs

    def _apply_offspring_job(
        self,
        job: tuple[
            Literal[
                "definition_mutation",
                "definition_crossover",
                "example_mutation",
                "example_crossover",
            ],
            Instruction,
            Instruction | None,
        ],
    ) -> Instruction:
        operator, parent, partner = job
        if operator == "definition_mutation":
            return self.operators.mutate_definition(parent)
        if operator == "definition_crossover":
            return self.operators.crossover_definition(parent, partner or parent)
        if operator == "example_mutation":
            return self.operators.mutate_example(parent)
        return self.operators.crossover_example(parent, partner or parent)

    @staticmethod
    def _log_population(population: list[Instruction]) -> None:
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
    def _log_pareto_front(pareto_population: list[Instruction]) -> None:
        print("\nPareto Front:")
        for instruction in pareto_population:
            print(
                "PerfObj: "
                f"{instruction.objectives.performance:.6f}, "
                f"Len: {instruction.objectives.length:.1f}, "
                f"PPL: {instruction.objectives.perplexity:.6f}"
            )

    @staticmethod
    def _log_run_summary(run_scores: list[float]) -> None:
        if not run_scores:
            return

        mean_score = statistics.mean(run_scores)
        std_score = statistics.stdev(run_scores) if len(run_scores) > 1 else 0.0
        print(
            "\n===== Summary =====\n"
            f"Accuracy over {len(run_scores)} runs: {mean_score * 100:.2f} +- {std_score * 100:.2f}"
        )
