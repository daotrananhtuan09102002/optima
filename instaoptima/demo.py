from __future__ import annotations

import os
import re
from pathlib import Path

from instaoptima.config import ExperimentConfig
from instaoptima.instruction import Instruction
from instaoptima.instruction import TaskExample


DEFAULT_DEMO_CONFIG_PATH = Path("config_absa_debug.yaml")
DEFAULT_DEMO_INSTRUCTION_PATH = Path(
    "artifacts/laptop14_absa_20260409_045620/run_1/best_instruction.txt"
)
DEFAULT_DEMO_TRAINED_MODEL_PATH = Path("models/flan_t5_absa_best_prompt")
DEFAULT_DEMO_MODEL_SOURCE = os.getenv(
    "INSTOPTIMA_DEMO_MODEL",
    str(DEFAULT_DEMO_TRAINED_MODEL_PATH)
    if DEFAULT_DEMO_TRAINED_MODEL_PATH.exists()
    else "google/flan-t5-base",
)


def load_demo_config(config_path: str | os.PathLike[str] | None = None) -> ExperimentConfig:
    path = Path(config_path) if config_path else DEFAULT_DEMO_CONFIG_PATH
    return ExperimentConfig.from_yaml(path)


def load_demo_instruction(
    instruction_path: str | os.PathLike[str] | None = None,
) -> Instruction:
    path = Path(instruction_path) if instruction_path else DEFAULT_DEMO_INSTRUCTION_PATH
    return parse_instruction_text(path.read_text(encoding="utf-8"))


def parse_instruction_text(raw_text: str) -> Instruction:
    definition = _extract_section(raw_text, "Definition", "Examples").strip()
    examples_block = _extract_section(raw_text, "Examples", "Objectives").strip()
    examples = _parse_examples(examples_block)
    if not definition:
        raise ValueError("Instruction file is missing a definition section.")
    return Instruction(definition=definition, examples=examples)


def build_demo_prompt(
    sentence: str,
    aspect: str | None,
    config: ExperimentConfig,
    instruction: Instruction,
) -> str:
    query = TaskExample(text=sentence.strip(), aspect=(aspect or "").strip() or None, label="")
    return instruction.build_prompt(query, config)


def generate_prediction(
    sentence: str,
    aspect: str | None,
    *,
    model,
    tokenizer,
    config: ExperimentConfig,
    instruction: Instruction,
) -> dict[str, str]:
    torch, _, _ = _load_inference_backend()
    prompt = build_demo_prompt(sentence, aspect, config, instruction)
    device = next(model.parameters()).device
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.task_model_max_source_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=config.task_model_generation_max_new_tokens,
        )
    raw_output = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    normalized_label = normalize_prediction(raw_output, config)
    return {
        "prompt": prompt,
        "raw_output": raw_output,
        "normalized_label": normalized_label,
    }


def load_model_and_tokenizer(
    model_source: str,
    config: ExperimentConfig,
):
    _, AutoModelForSeq2SeqLM, AutoTokenizer = _load_inference_backend()
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        cache_dir=config.task_model_cache_dir,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_source,
        cache_dir=config.task_model_cache_dir,
    )
    model.to(resolve_device(config.task_model_device))
    model.eval()
    return model, tokenizer


def normalize_prediction(raw_prediction: str, config: ExperimentConfig) -> str:
    prediction = raw_prediction.lower().strip()
    if config.label_space:
        for label in config.label_space:
            if label.lower() in prediction:
                return label.lower()
    return prediction


def resolve_device(device_preference: str) -> str:
    torch, _, _ = _load_inference_backend()
    if device_preference == "cpu":
        return "cpu"
    if device_preference == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_inference_backend():
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing inference dependencies. Install requirements.txt before running the demo."
        ) from exc
    return torch, AutoModelForSeq2SeqLM, AutoTokenizer


def _extract_section(raw_text: str, start_heading: str, end_heading: str) -> str:
    pattern = rf"{start_heading}:\s*(.*?)(?:\n{end_heading}:|\Z)"
    match = re.search(pattern, raw_text, flags=re.DOTALL)
    return match.group(1) if match else ""


def _parse_examples(block: str) -> list[TaskExample]:
    examples: list[TaskExample] = []
    chunks = re.split(r"\n(?=\d+\.\s+Sentence:)", block.strip()) if block.strip() else []
    for chunk in chunks:
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        if not lines:
            continue
        sentence = ""
        aspect = None
        label = ""
        for index, line in enumerate(lines):
            if index == 0:
                line = re.sub(r"^\d+\.\s*", "", line)
            if line.startswith("Sentence:"):
                sentence = line.removeprefix("Sentence:").strip()
            elif line.startswith("Aspect:"):
                aspect = line.removeprefix("Aspect:").strip()
            elif line.startswith("Label:"):
                label = line.removeprefix("Label:").strip().lower()
        if sentence:
            examples.append(TaskExample(text=sentence, aspect=aspect, label=label))
    return examples
