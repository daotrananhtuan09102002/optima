import gc
import inspect
import shutil
import tempfile
from pathlib import Path

from datasets import Dataset

from instaoptima.config import ExperimentConfig
from instaoptima.instruction import Instruction
from instaoptima.instruction import TaskExample

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM
    from transformers import AutoTokenizer
    from transformers import DataCollatorForSeq2Seq
    from transformers import Seq2SeqTrainer
    from transformers import Seq2SeqTrainingArguments
except ImportError:  # pragma: no cover
    torch = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None
    DataCollatorForSeq2Seq = None
    Seq2SeqTrainer = None
    Seq2SeqTrainingArguments = None


class FlanT5ObjectiveEvaluator:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._tokenizer = None

    def evaluate(
        self,
        instruction: Instruction,
        train_dataset: list[TaskExample],
        test_dataset: list[TaskExample],
    ) -> list[str]:
        self._ensure_backend()
        if not train_dataset:
            raise ValueError("Train dataset must not be empty for Flan-T5 fine-tuning.")

        tokenizer = self._get_tokenizer()
        model = self._load_model()

        train_records = [
            {
                "input_text": instruction.build_prompt(example, self.config),
                "target_text": example.label,
            }
            for example in train_dataset
        ]
        train_hf_dataset = Dataset.from_list(train_records)
        tokenized_train_dataset = train_hf_dataset.map(
            self._tokenize_batch,
            batched=True,
            remove_columns=train_hf_dataset.column_names,
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
        output_dir = self._make_training_output_dir()
        training_args = self._build_training_args(output_dir)
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        try:
            trainer.train()
            predictions = self._generate_predictions(model, tokenizer, instruction, test_dataset)
        finally:
            del trainer
            self._cleanup_model(model)
            shutil.rmtree(output_dir, ignore_errors=True)

        return predictions

    def _tokenize_batch(self, batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        tokenizer = self._get_tokenizer()
        model_inputs = tokenizer(
            batch["input_text"],
            truncation=True,
            max_length=self.config.task_model_max_source_length,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            truncation=True,
            max_length=self.config.task_model_max_target_length,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def _generate_predictions(
        self,
        model,
        tokenizer,
        instruction: Instruction,
        test_dataset: list[TaskExample],
    ) -> list[str]:
        if not test_dataset:
            return []

        predictions: list[str] = []
        device = model.device
        batch_size = max(1, int(self.config.task_model_eval_batch_size))

        model.eval()
        for start_index in range(0, len(test_dataset), batch_size):
            batch = test_dataset[start_index : start_index + batch_size]
            prompts = [
                instruction.build_prompt(example, self.config)
                for example in batch
            ]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.task_model_max_source_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=self.config.task_model_generation_max_new_tokens,
                )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            predictions.extend(decoded)

        return predictions

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.task_model_name,
                cache_dir=self.config.task_model_cache_dir,
            )
        return self._tokenizer

    def _load_model(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.task_model_name,
            cache_dir=self.config.task_model_cache_dir,
        )
        return model

    def _make_training_output_dir(self) -> str:
        artifact_root = self.config.task_model_artifact_dir
        if artifact_root:
            root = Path(artifact_root)
            root.mkdir(parents=True, exist_ok=True)
            return tempfile.mkdtemp(prefix="instaoptima-flan-", dir=root)
        return tempfile.mkdtemp(prefix="instaoptima-flan-")

    def _build_training_args(self, output_dir: str):
        candidate_kwargs = {
            "output_dir": output_dir,
            "overwrite_output_dir": True,
            "do_train": True,
            "do_eval": False,
            "evaluation_strategy": "no",
            "eval_strategy": "no",
            "save_strategy": "no",
            "logging_strategy": "no",
            "report_to": [],
            "per_device_train_batch_size": self.config.task_model_train_batch_size,
            "per_device_eval_batch_size": self.config.task_model_eval_batch_size,
            "gradient_accumulation_steps": self.config.task_model_gradient_accumulation_steps,
            "learning_rate": self.config.task_model_learning_rate,
            "weight_decay": self.config.task_model_weight_decay,
            "warmup_ratio": self.config.task_model_warmup_ratio,
            "num_train_epochs": self.config.task_model_train_epochs,
            "seed": self.config.shuffle_seed,
            "data_seed": self.config.shuffle_seed,
            "remove_unused_columns": True,
            "disable_tqdm": True,
            "use_cpu": self._use_cpu(),
            "fp16": self._use_fp16(),
            "bf16": self._use_bf16(),
        }
        supported_parameters = inspect.signature(
            Seq2SeqTrainingArguments.__init__
        ).parameters
        filtered_kwargs = {
            key: value
            for key, value in candidate_kwargs.items()
            if key in supported_parameters
        }

        if (
            "evaluation_strategy" in filtered_kwargs
            and "eval_strategy" in filtered_kwargs
        ):
            filtered_kwargs.pop("eval_strategy")

        return Seq2SeqTrainingArguments(**filtered_kwargs)

    @staticmethod
    def _cleanup_model(model) -> None:
        del model
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _ensure_backend() -> None:
        if any(
            dependency is None
            for dependency in (
                torch,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
                DataCollatorForSeq2Seq,
                Seq2SeqTrainer,
                Seq2SeqTrainingArguments,
            )
        ):
            raise ImportError(
                "Flan-T5 evaluation requires torch and transformers to be installed."
            )

    def _use_cpu(self) -> bool:
        requested_device = self.config.task_model_device.lower()
        if requested_device == "cpu":
            return True
        if requested_device == "auto":
            return torch is None or not torch.cuda.is_available()
        return False

    def _use_fp16(self) -> bool:
        requested_device = self.config.task_model_device.lower()
        return requested_device != "cpu" and torch is not None and torch.cuda.is_available()

    def _use_bf16(self) -> bool:
        return False
