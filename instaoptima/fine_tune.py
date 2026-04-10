from __future__ import annotations

import inspect
import json
from collections import Counter
from dataclasses import asdict
import math
from pathlib import Path

from instaoptima.config import ExperimentConfig
from instaoptima.data_loader import ExperimentDatasetLoader
from instaoptima.instruction import Instruction
from instaoptima.instruction import TaskExample


def fine_tune_and_save(
    *,
    config: ExperimentConfig,
    instruction: Instruction | None,
    output_dir: str | Path,
    model_source: str | None = None,
    num_train_epochs: float | None = None,
    prompt_mode: str = "best",
) -> dict[str, object]:
    backend = _load_backend()
    torch = backend["torch"]
    AutoModelForSeq2SeqLM = backend["AutoModelForSeq2SeqLM"]
    AutoTokenizer = backend["AutoTokenizer"]
    DataCollatorForSeq2Seq = backend["DataCollatorForSeq2Seq"]
    Dataset = backend["Dataset"]
    Seq2SeqTrainer = backend["Seq2SeqTrainer"]
    Seq2SeqTrainingArguments = backend["Seq2SeqTrainingArguments"]

    dataset_bundle = ExperimentDatasetLoader(config).load()
    model_name_or_path = model_source or config.task_model_name
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=config.task_model_cache_dir,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path,
        cache_dir=config.task_model_cache_dir,
    )

    train_dataset = Dataset.from_list(
        _build_records(dataset_bundle.train, instruction, config)
    )
    validation_examples = list(dataset_bundle.validation)
    test_examples = list(dataset_bundle.test)
    validation_dataset = Dataset.from_list(
        _build_records(validation_examples, instruction, config)
    )
    test_dataset = Dataset.from_list(_build_records(test_examples, instruction, config))

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        model_inputs = tokenizer(
            batch["input_text"],
            truncation=True,
            max_length=config.task_model_max_source_length,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            truncation=True,
            max_length=config.task_model_max_target_length,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    tokenized_validation = validation_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=validation_dataset.column_names,
    )
    tokenized_test = test_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def compute_metrics(eval_prediction) -> dict[str, float]:
        predictions, labels = eval_prediction
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        decoded_predictions = tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
        )
        labels = labels.copy()
        labels[labels == -100] = tokenizer.pad_token_id
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        normalized_predictions = [
            normalize_prediction(prediction, config)
            for prediction in decoded_predictions
        ]
        normalized_labels = [label.strip().lower() for label in decoded_labels]
        return compute_classification_metrics(
            normalized_predictions,
            normalized_labels,
            config,
        )

    training_args = _build_training_args(
        config=config,
        output_dir=destination,
        num_train_epochs=num_train_epochs,
        use_validation=bool(validation_examples),
        Seq2SeqTrainingArguments=Seq2SeqTrainingArguments,
        torch=torch,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation if validation_examples else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if validation_examples else None,
    )

    trainer.train()
    trainer.save_model(str(destination))
    tokenizer.save_pretrained(str(destination))

    validation_metrics = {}
    if validation_examples:
        validation_metrics = trainer.evaluate(
            eval_dataset=tokenized_validation,
            metric_key_prefix="validation",
        )

    test_predictions = _predict_labels(
        model=trainer.model,
        tokenizer=tokenizer,
        examples=test_examples,
        instruction=instruction,
        config=config,
        torch=torch,
    )
    test_metrics = compute_classification_metrics(
        predictions=test_predictions,
        gold_labels=[example.label for example in test_examples],
        config=config,
    )

    summary = {
        "model_source": model_name_or_path,
        "output_dir": str(destination),
        "train_size": len(dataset_bundle.train),
        "validation_size": len(validation_examples),
        "test_size": len(test_examples),
        "num_train_epochs": (
            float(num_train_epochs)
            if num_train_epochs is not None
            else float(config.task_model_train_epochs)
        ),
        "instruction": {
            "definition": instruction.definition if instruction is not None else "",
            "examples": (
                [asdict(example) for example in instruction.examples]
                if instruction is not None
                else []
            ),
        },
        "prompt_mode": prompt_mode,
        "validation_metrics": _to_jsonable_metrics(validation_metrics),
        "test_metrics": test_metrics,
    }
    (destination / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _build_records(
    examples: list[TaskExample],
    instruction: Instruction | None,
    config: ExperimentConfig,
) -> list[dict[str, str]]:
    return [
        {
            "input_text": build_task_prompt(
                TaskExample(text=example.text, label="", aspect=example.aspect),
                config,
                instruction,
            ),
            "target_text": example.label,
        }
        for example in examples
    ]


def _predict_labels(
    *,
    model,
    tokenizer,
    examples: list[TaskExample],
    instruction: Instruction | None,
    config: ExperimentConfig,
    torch,
) -> list[str]:
    predictions: list[str] = []
    if not examples:
        return predictions

    device = model.device
    batch_size = max(1, int(config.task_model_eval_batch_size))
    model.eval()

    for start_index in range(0, len(examples), batch_size):
        batch = examples[start_index : start_index + batch_size]
        prompts = [
            build_task_prompt(
                TaskExample(text=example.text, label="", aspect=example.aspect),
                config,
                instruction,
            )
            for example in batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.task_model_max_source_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=config.task_model_generation_max_new_tokens,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        predictions.extend(
            normalize_prediction(prediction, config) for prediction in decoded
        )
    return predictions


def normalize_prediction(raw_prediction: str, config: ExperimentConfig) -> str:
    prediction = raw_prediction.lower().strip()
    if config.label_space:
        for label in config.label_space:
            if label.lower() in prediction:
                return label.lower()
    if "positive" in prediction:
        return "positive"
    if "negative" in prediction:
        return "negative"
    if "neutral" in prediction:
        return "neutral"
    if "conflict" in prediction:
        return "conflict"
    return prediction


def build_task_prompt(
    example: TaskExample,
    config: ExperimentConfig,
    instruction: Instruction | None,
) -> str:
    if instruction is not None:
        return instruction.build_prompt(example, config)

    lines = [f"Sentence: {example.text}"]
    if config.task_type == "absa" and example.aspect:
        lines.append(f"Aspect: {example.aspect}")
    label_hint = ", ".join(config.label_space or [])
    lines.append(f"Label (only output one of: {label_hint}):")
    return "\n".join(lines)


def compute_classification_metrics(
    predictions: list[str],
    gold_labels: list[str],
    config: ExperimentConfig,
) -> dict[str, float]:
    if not gold_labels:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "prediction_entropy": 0.0,
        }

    labels = config.label_space or sorted(set(gold_labels))
    correct = sum(pred == gold for pred, gold in zip(predictions, gold_labels))
    accuracy = correct / len(gold_labels)

    per_label_metrics = []
    for label in labels:
        tp = sum(
            pred == label and gold == label
            for pred, gold in zip(predictions, gold_labels)
        )
        fp = sum(
            pred == label and gold != label
            for pred, gold in zip(predictions, gold_labels)
        )
        fn = sum(
            pred != label and gold == label
            for pred, gold in zip(predictions, gold_labels)
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_label_metrics.append((precision, recall, f1))

    macro_precision = sum(item[0] for item in per_label_metrics) / len(labels)
    macro_recall = sum(item[1] for item in per_label_metrics) / len(labels)
    macro_f1 = sum(item[2] for item in per_label_metrics) / len(labels)

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "prediction_entropy": prediction_entropy(predictions),
    }


def prediction_entropy(predictions: list[str]) -> float:
    if not predictions:
        return 0.0

    counts = Counter(predictions)
    total = len(predictions)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log(probability + 1e-12)
    return entropy


def _build_training_args(
    *,
    config: ExperimentConfig,
    output_dir: Path,
    num_train_epochs: float | None,
    use_validation: bool,
    Seq2SeqTrainingArguments,
    torch,
):
    metric_name = "eval_macro_f1"
    candidate_kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "do_train": True,
        "do_eval": use_validation,
        "evaluation_strategy": "epoch" if use_validation else "no",
        "eval_strategy": "epoch" if use_validation else "no",
        "save_strategy": "epoch" if use_validation else "no",
        "logging_strategy": "epoch",
        "report_to": [],
        "per_device_train_batch_size": config.task_model_train_batch_size,
        "per_device_eval_batch_size": config.task_model_eval_batch_size,
        "gradient_accumulation_steps": config.task_model_gradient_accumulation_steps,
        "learning_rate": config.task_model_learning_rate,
        "weight_decay": config.task_model_weight_decay,
        "warmup_ratio": config.task_model_warmup_ratio,
        "num_train_epochs": (
            num_train_epochs
            if num_train_epochs is not None
            else config.task_model_train_epochs
        ),
        "seed": config.shuffle_seed,
        "data_seed": config.shuffle_seed,
        "remove_unused_columns": True,
        "disable_tqdm": False,
        "load_best_model_at_end": use_validation,
        "metric_for_best_model": metric_name if use_validation else None,
        "greater_is_better": True if use_validation else None,
        "predict_with_generate": True,
        "generation_max_length": config.task_model_max_target_length,
        "generation_num_beams": 1,
        "save_total_limit": 1 if use_validation else None,
        "use_cpu": _use_cpu(config, torch),
        "fp16": _use_fp16(config, torch),
        "bf16": False,
    }
    supported_parameters = inspect.signature(
        Seq2SeqTrainingArguments.__init__
    ).parameters
    filtered_kwargs = {
        key: value
        for key, value in candidate_kwargs.items()
        if key in supported_parameters and value is not None
    }

    if (
        "evaluation_strategy" in filtered_kwargs
        and "eval_strategy" in filtered_kwargs
    ):
        filtered_kwargs.pop("eval_strategy")

    return Seq2SeqTrainingArguments(**filtered_kwargs)


def _to_jsonable_metrics(metrics: dict[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            payload[key] = value
    return payload


def _use_cpu(config: ExperimentConfig, torch) -> bool:
    requested_device = config.task_model_device.lower()
    if requested_device == "cpu":
        return True
    if requested_device == "auto":
        return not torch.cuda.is_available()
    return False


def _use_fp16(config: ExperimentConfig, torch) -> bool:
    requested_device = config.task_model_device.lower()
    return requested_device != "cpu" and torch.cuda.is_available()


def _load_backend() -> dict[str, object]:
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForSeq2SeqLM
        from transformers import AutoTokenizer
        from transformers import DataCollatorForSeq2Seq
        from transformers import Seq2SeqTrainer
        from transformers import Seq2SeqTrainingArguments
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing fine-tuning dependencies. Install requirements.txt before training."
        ) from exc

    return {
        "torch": torch,
        "Dataset": Dataset,
        "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorForSeq2Seq": DataCollatorForSeq2Seq,
        "Seq2SeqTrainer": Seq2SeqTrainer,
        "Seq2SeqTrainingArguments": Seq2SeqTrainingArguments,
    }
