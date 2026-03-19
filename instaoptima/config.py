from dataclasses import asdict, dataclass
import os
from pathlib import Path

from dotenv import load_dotenv
import yaml

load_dotenv()


@dataclass(frozen=True)
class ExperimentConfig:
    model: str = "gpt-4o-mini"
    operator_model: str | None = None
    population_size: int = 100
    generations: int = 10
    num_runs: int = 5
    task_type: str = "sentence_classification"
    dataset_source: str = "huggingface"
    dataset_name: str = "glue"
    dataset_subset: str = "sst2"
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    train_sample_size: int = 1000
    validation_sample_size: int = 1000
    test_sample_size: int = 1000
    local_train_path: str | None = None
    local_validation_path: str | None = None
    local_test_path: str | None = None
    auto_download_local_dataset: bool = True
    local_validation_ratio: float = 0.1
    text_field: str = "sentence"
    aspect_field: str | None = None
    label_field: str = "label"
    label_space: list[str] | None = None
    shuffle_seed: int = 42
    temperature: float = 1.0
    operator_temperature: float | None = None
    max_generation_tokens: int = 500
    max_examples: int = 2
    minimization_objectives: str = (
        "- Minimize the instruction performance objective value.\n"
        "- Minimize the instruction length.\n"
        "- Minimize the instruction perplexity."
    )
    optimization_split: str = "validation"
    report_split: str = "test"
    performance_metric_names: tuple[str, ...] = (
        "accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
    )
    perplexity_model_name: str = "roberta-base"
    random_replacement_ratio: float = 0.1
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ExperimentConfig":
        path = Path(yaml_path)
        with path.open("r", encoding="utf-8") as yaml_file:
            raw_config = yaml.safe_load(yaml_file) or {}

        if not isinstance(raw_config, dict):
            raise ValueError("YAML config must contain a top-level mapping.")

        default_values = asdict(cls())
        merged_config = {**default_values, **raw_config}
        if merged_config.get("label_space"):
            merged_config["label_space"] = [
                str(label).strip().lower() for label in merged_config["label_space"]
            ]
        if merged_config.get("performance_metric_names"):
            merged_config["performance_metric_names"] = tuple(
                merged_config["performance_metric_names"]
            )
        return cls(**merged_config)
