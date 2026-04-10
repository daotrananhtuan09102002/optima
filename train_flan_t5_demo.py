from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from instaoptima.config import ExperimentConfig
from instaoptima.demo import load_demo_instruction
from instaoptima.fine_tune import fine_tune_and_save
from instaoptima.instruction import Instruction
from instaoptima.instruction import TaskExample


DEFAULT_INSTRUCTION_PATH = Path(
    "artifacts/laptop14_absa_20260409_045620/run_1/best_instruction.txt"
)
DEFAULT_RANDOM_POOL_PATH = Path(
    "artifacts/laptop14_absa_20260409_045620/run_1/final_population.json"
)
DEFAULT_OUTPUT_ROOT = Path("models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="config_absa_debug.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--instruction-path",
        "-i",
        type=Path,
        default=DEFAULT_INSTRUCTION_PATH,
        help="Path to best_instruction.txt used when --prompt-mode=best.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Directory to save the fine-tuned model. Defaults to models/flan_t5_absa_<prompt-mode>.",
    )
    parser.add_argument(
        "--model-source",
        "-m",
        default=None,
        help="Base model name or local path. Defaults to config.task_model_name.",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=float,
        default=10.0,
        help="Number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("best", "random", "none"),
        default="best",
        help="Prompt condition used to build training inputs.",
    )
    parser.add_argument(
        "--random-pool-path",
        type=Path,
        default=DEFAULT_RANDOM_POOL_PATH,
        help="JSON population file used to sample a random instruction when --prompt-mode=random.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed used when sampling a random instruction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    instruction = resolve_instruction(
        prompt_mode=args.prompt_mode,
        instruction_path=args.instruction_path,
        random_pool_path=args.random_pool_path,
        random_seed=args.random_seed,
    )
    output_dir = args.output_dir or default_output_dir(args.prompt_mode)
    summary = fine_tune_and_save(
        config=config,
        instruction=instruction,
        output_dir=output_dir,
        model_source=args.model_source,
        num_train_epochs=args.epochs,
        prompt_mode=args.prompt_mode,
    )

    print("Fine-tuning completed.")
    print(f"Prompt mode: {summary['prompt_mode']}")
    print(f"Model saved to: {summary['output_dir']}")
    print(f"Train size: {summary['train_size']}")
    print(f"Validation size: {summary['validation_size']}")
    print(f"Test size: {summary['test_size']}")
    print(f"Epochs: {summary['num_train_epochs']}")
    print("Test metrics:")
    for key, value in summary["test_metrics"].items():
        print(f"  - {key}: {value}")


def resolve_instruction(
    *,
    prompt_mode: str,
    instruction_path: Path,
    random_pool_path: Path,
    random_seed: int,
) -> Instruction | None:
    if prompt_mode == "best":
        return load_demo_instruction(instruction_path)
    if prompt_mode == "random":
        return load_random_instruction(random_pool_path, random_seed)
    return None


def load_random_instruction(pool_path: Path, random_seed: int) -> Instruction:
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Random instruction pool is empty or invalid: {pool_path}")

    instructions: list[Instruction] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        definition = str(item.get("definition", "")).strip()
        examples_payload = item.get("examples", [])
        if not definition or not isinstance(examples_payload, list):
            continue
        examples = []
        for example in examples_payload:
            if not isinstance(example, dict):
                continue
            text = str(example.get("text", "")).strip()
            label = str(example.get("label", "")).strip().lower()
            aspect = example.get("aspect")
            aspect_text = str(aspect).strip() if aspect is not None else None
            if text and label:
                examples.append(TaskExample(text=text, label=label, aspect=aspect_text))
        instructions.append(Instruction(definition=definition, examples=examples))

    if not instructions:
        raise ValueError(f"No valid instructions found in random pool: {pool_path}")
    return random.Random(random_seed).choice(instructions)


def default_output_dir(prompt_mode: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"flan_t5_absa_{prompt_mode}_prompt"


if __name__ == "__main__":
    main()
