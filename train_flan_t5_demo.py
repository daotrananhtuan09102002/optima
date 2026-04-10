from __future__ import annotations

import argparse
from pathlib import Path

from instaoptima.config import ExperimentConfig
from instaoptima.demo import load_demo_instruction
from instaoptima.fine_tune import fine_tune_and_save


DEFAULT_INSTRUCTION_PATH = Path(
    "artifacts/laptop14_absa_20260409_045620/run_1/best_instruction.txt"
)
DEFAULT_OUTPUT_DIR = Path("models/flan_t5_absa_best_prompt")


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
        help="Path to best_instruction.txt used to build training prompts.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the fine-tuned model.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    instruction = load_demo_instruction(args.instruction_path)
    summary = fine_tune_and_save(
        config=config,
        instruction=instruction,
        output_dir=args.output_dir,
        model_source=args.model_source,
        num_train_epochs=args.epochs,
    )

    print("Fine-tuning completed.")
    print(f"Model saved to: {summary['output_dir']}")
    print(f"Train size: {summary['train_size']}")
    print(f"Validation size: {summary['validation_size']}")
    print(f"Test size: {summary['test_size']}")
    print(f"Epochs: {summary['num_train_epochs']}")
    print("Test metrics:")
    for key, value in summary["test_metrics"].items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
