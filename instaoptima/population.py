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
                "Determine the sentiment polarity for the specified aspect in the sentence.",
                "Focus on the provided aspect and classify its sentiment as positive, negative, neutral, or conflict.",
                "Read the sentence with its aspect term and predict the aspect sentiment label.",
                "Identify how the sentence feels about the given aspect term.",
                "For the named aspect, return one sentiment label: positive, negative, neutral, or conflict.",
                "Judge sentiment only for the provided aspect, not for the whole sentence.",
                "Infer the sentiment orientation toward the target aspect from the sentence.",
                "Classify sentiment at aspect level using the given sentence and aspect.",
                "Based on the sentence context, decide the sentiment for the specified aspect.",
                "Output the sentiment label associated with the highlighted aspect term.",
                "Evaluate sentiment toward the given aspect and return the correct class.",
                "Given an aspect in a sentence, predict whether sentiment is positive, negative, neutral, or conflict.",
                "Perform aspect-based sentiment classification for the provided aspect term.",
                "Classify only the sentiment about the target aspect in the sentence.",
                "Determine the aspect-specific sentiment from the sentence context.",
                "Return the polarity label that best matches sentiment toward the provided aspect.",
                "Analyze opinion toward the given aspect and output one valid sentiment class.",
                "Predict sentiment for the specified aspect term using the sentence evidence.",
            ]

        return [
            "Classify the sentiment of the sentence.",
            "Determine if the sentiment is positive or negative.",
            "Analyze whether the sentence expresses positive or negative emotion.",
            "Read the sentence and output its overall sentiment label.",
            "Identify the emotional polarity conveyed by the sentence.",
            "Decide whether the sentence sentiment is positive or negative.",
            "Infer the sentiment orientation of the given sentence.",
            "Predict the sentiment category for this sentence.",
            "Assess if the sentence expresses favorable or unfavorable sentiment.",
            "Determine the sentiment tone of the sentence text.",
            "Classify the sentence into the correct sentiment class.",
            "Evaluate sentiment expressed in the sentence and return one label.",
            "Judge whether the sentence sentiment is positive or negative.",
            "Analyze the sentence and assign the appropriate sentiment label.",
            "Estimate the overall sentiment polarity of the sentence.",
            "From the sentence content, choose the correct sentiment class.",
            "Identify if the sentence communicates positive or negative sentiment.",
            "Detect sentiment in the sentence and output the matching label.",
            "Determine overall emotional valence expressed by the sentence.",
            "Perform sentence-level sentiment classification for this text.",
            "Predict whether the sentence sentiment is positive or negative.",
        ]
