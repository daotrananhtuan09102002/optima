from instaoptima.config import ExperimentConfig
from instaoptima.instruction import Instruction
from instaoptima.instruction import TaskExample


class PopulationFactory:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def create_initial_population(
        self,
        seed_examples: list[TaskExample],
    ) -> list[Instruction]:
        base_definitions = self._base_definitions()
        if not seed_examples:
            raise ValueError("At least one seed example is required to initialize population.")

        population: list[Instruction] = []
        for index in range(self.config.population_size):
            definition = base_definitions[index % len(base_definitions)]
            start = index % len(seed_examples)
            examples = [
                seed_examples[(start + offset) % len(seed_examples)]
                for offset in range(min(self.config.max_examples, len(seed_examples)))
            ]
            population.append(Instruction(definition=definition, examples=examples))
        return population

    def _base_definitions(self) -> list[str]:
        if self.config.task_type == "absa":
            return [
                "Given a sentence and an aspect term, classify the sentiment toward the aspect.",
                "Decide whether the sentiment expressed about the given aspect is positive, negative, neutral, or conflict.",
                "Analyze the aspect-level sentiment in the sentence and output the correct polarity label.",
            ]

        return [
            "Classify the sentiment of the sentence.",
            "Determine if the sentiment is positive or negative.",
            "Analyze whether the sentence expresses positive or negative emotion.",
        ]
