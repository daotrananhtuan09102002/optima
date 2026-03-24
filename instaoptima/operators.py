import re

from instaoptima.config import ExperimentConfig
from instaoptima.instruction import Instruction
from instaoptima.instruction import TaskExample
from instaoptima.llm_client import LLMClient


class EvolutionOperators:
    def __init__(
        self, llm_client: LLMClient, config: ExperimentConfig | None = None
    ) -> None:
        self.llm_client = llm_client
        self.config = config or ExperimentConfig()

    def mutate_definition(self, instruction: Instruction) -> Instruction:
        prompt = self._build_definition_mutation_prompt(instruction)
        new_definition = self._clean_definition_output(
            self._generate_operator_output(prompt)
        )
        return Instruction(new_definition, instruction.examples)

    def mutate_example(self, instruction: Instruction) -> Instruction:
        prompt = self._build_example_mutation_prompt(
            instruction.examples,
            instruction.objective_summary(),
        )
        rewritten = self._generate_operator_output(prompt)
        new_examples = self._parse_examples(rewritten, instruction.examples)
        return Instruction(instruction.definition, new_examples)

    def crossover_definition(
        self, first: Instruction, second: Instruction
    ) -> Instruction:
        prompt = self._build_definition_crossover_prompt(
            first,
            second,
        )
        new_definition = self._clean_definition_output(
            self._generate_operator_output(prompt)
        )
        return Instruction(new_definition, first.examples)

    def crossover_example(self, first: Instruction, second: Instruction) -> Instruction:
        prompt = self._build_example_crossover_prompt(
            first.examples,
            second.examples,
            first.objective_summary(),
            second.objective_summary(),
        )
        rewritten = self._generate_operator_output(prompt)
        new_examples = self._parse_examples(rewritten, first.examples)
        return Instruction(first.definition, new_examples)

    def crossover(self, first: Instruction, second: Instruction) -> Instruction:
        return self.crossover_definition(first, second)

    def _build_definition_mutation_prompt(self, instruction: Instruction) -> str:
        return (
            "You are optimizing a task instruction for evolutionary search. "
            "Rewrite the template prompt to preserve the same task intent while "
            "improving performance under the objectives.\n\n"
            "Output requirements:\n"
            "- Output only the rewritten prompt text.\n"
            "- Do not include explanations, prefixes, markdown, or quotes.\n"
            "- Do not include phrases like 'Certainly', 'Here is', 'paraphrased', "
            "or 'optimized version'.\n"
            "- Return exactly one instruction paragraph.\n\n"
            f"Minimization objectives:\n{self.config.minimization_objectives}\n\n"
            f"Objective values:\n{instruction.objective_summary()}\n\n"
            f"Template prompt:\n{instruction.definition}"
        )

    def _build_definition_crossover_prompt(
        self,
        first_instruction: Instruction,
        second_instruction: Instruction,
    ) -> str:
        return (
            "You are optimizing a task instruction for evolutionary search. "
            "Combine the strengths of two template prompts into one improved "
            "instruction that keeps the original task intent.\n\n"
            "Output requirements:\n"
            "- Output only the final merged prompt text.\n"
            "- Do not include explanations, prefixes, markdown, or quotes.\n"
            "- Do not include phrases like 'Certainly', 'Here is', 'rephrased', "
            "or 'optimized version'.\n"
            "- Return exactly one instruction paragraph.\n\n"
            f"Minimization objectives:\n{self.config.minimization_objectives}\n\n"
            f"Template prompt 1:\n{first_instruction.definition}\n"
            f"Objective values 1:\n{first_instruction.objective_summary()}\n\n"
            f"Template prompt 2:\n{second_instruction.definition}\n"
            f"Objective values 2:\n{second_instruction.objective_summary()}"
        )

    def _build_example_mutation_prompt(
        self,
        examples: list[TaskExample],
        objective_summary: str,
    ) -> str:
        return (
            "I want you to be a professional prompt engineer. Now I am working "
            "on the multi-objective evolutionary prompt optimization for "
            "sentiment analysis, and I need your help to design and optimize "
            "the template prompt. Here I give you two groups of examples for "
            "completing the prompt, please generate new examples to substitute "
            f"the following examples and there are no more than {self.config.max_examples} "
            "examples in the new prompt. Given the minimization objectives, "
            "please be creative and output the generated example in the same "
            "format. Please remove Minimization objectives in the output.\n\n"
            f"Minimization objectives:\n{self.config.minimization_objectives}\n\n"
            f"Objective values:\n{objective_summary}\n\n"
            f"Examples:\n{self._format_examples(examples)}"
        )

    def _build_example_crossover_prompt(
        self,
        first_examples: list[TaskExample],
        second_examples: list[TaskExample],
        first_objectives: str,
        second_objectives: str,
    ) -> str:
        return (
            "I want you to be a professional prompt engineer. Now I am working "
            "on the multi-objective evolutionary prompt optimization for "
            "sentiment analysis, and I need your help to design and optimize "
            "the template prompt. Here I give you two groups of examples for "
            "completing the prompt, please read the examples of the two groups "
            "of examples and crossover the examples into a new example group "
            f"and there are no more than {self.config.max_examples} examples in "
            "the new examples. Given the minimization objectives, please be "
            "creative and output the crossovered the examples. Please remove "
            "Minimization objectives in the output.\n\n"
            f"Minimization objectives:\n{self.config.minimization_objectives}\n\n"
            f"Example group 1:\n{self._format_examples(first_examples)}\n\n"
            f"Objective values 1:\n{first_objectives}\n\n"
            f"Example group 2:\n{self._format_examples(second_examples)}"
            f"\n\nObjective values 2:\n{second_objectives}"
        )

    def _generate_operator_output(self, prompt: str) -> str:
        return self.llm_client.generate(
            prompt,
            model=self.config.operator_model or self.config.model,
            temperature=(
                self.config.operator_temperature
                if self.config.operator_temperature is not None
                else self.config.temperature
            ),
        )

    def _format_examples(self, examples: list[TaskExample]) -> str:
        return "\n".join(
            self._format_example(example) for example in examples
        )

    def _format_example(self, example: TaskExample) -> str:
        lines = [f"Sentence: {example.text}"]
        if self.config.task_type == "absa" and example.aspect:
            lines.append(f"Aspect: {example.aspect}")
        lines.append(f"Label: {example.label}")
        return "\n".join(lines)

    def _parse_examples(
        self,
        raw_text: str,
        fallback_examples: list[TaskExample],
    ) -> list[TaskExample]:
        pattern = (
            r"Sentence:\s*(.*?)"
            r"(?:\s*Aspect:\s*(.*?))?"
            r"\s*Label:\s*([A-Za-z_ -]+)"
        )
        matches = re.findall(pattern, raw_text, flags=re.IGNORECASE | re.DOTALL)
        parsed_examples = []
        for sentence, aspect, label in matches:
            normalized_label = label.strip().lower()
            if self.config.label_space and normalized_label not in self.config.label_space:
                continue
            parsed_examples.append(
                TaskExample(
                    text=sentence.strip(),
                    aspect=aspect.strip() if aspect else None,
                    label=normalized_label,
                )
            )
        if not parsed_examples:
            return fallback_examples
        return parsed_examples[: self.config.max_examples]

    def _clean_definition_output(self, raw_text: str) -> str:
        cleaned = raw_text.strip()

        cleaned = re.sub(r"^\s*```(?:\w+)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        cleaned = cleaned.strip()

        chatty_prefix_pattern = re.compile(
            r"^(?:"
            r"(?:certainly|sure|of course|absolutely|great)\W+"
            r"|(?:here(?:'s| is)?\s+(?:a\s+)?)"
            r"|(?:the\s+(?:rephrased|paraphrased|optimized|improved)\s+version\s*(?:is)?\W*)"
            r"|(?:rephrased\s+prompt\s*[:\-])"
            r"|(?:optimized\s+prompt\s*[:\-])"
            r")",
            flags=re.IGNORECASE,
        )
        cleaned = chatty_prefix_pattern.sub("", cleaned).strip()

        # If the model wraps the instruction in quotes after an intro, keep only quoted content.
        quoted_match = re.search(r'"([^"\n]{20,})"', cleaned)
        if quoted_match:
            cleaned = quoted_match.group(1).strip()

        cleaned = cleaned.strip(" \n\t\"'`")

        if not cleaned:
            return raw_text.strip()
        return cleaned
