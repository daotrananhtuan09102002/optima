import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

from instaoptima.config import ExperimentConfig
from instaoptima.instruction import TaskExample


@dataclass(frozen=True)
class DatasetBundle:
    train: list[TaskExample]
    validation: list[TaskExample]
    test: list[TaskExample]

    def get_split(self, split_name: str) -> list[TaskExample]:
        return getattr(self, split_name)


class ExperimentDatasetLoader:
    _AUTO_ABSA_DATASETS = {
        "laptop14": "tomaarsen/setfit-absa-semeval-laptops",
        "restaurant14": "tomaarsen/setfit-absa-semeval-restaurants",
    }

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def load(self) -> DatasetBundle:
        if self.config.dataset_source == "huggingface":
            return self._load_huggingface_bundle()
        if self.config.dataset_source == "local":
            return self._load_local_bundle()
        raise ValueError(f"Unsupported dataset_source: {self.config.dataset_source}")

    def _load_huggingface_bundle(self) -> DatasetBundle:
        dataset = load_dataset(self.config.dataset_name, self.config.dataset_subset)
        return DatasetBundle(
            train=self._select_records(
                dataset[self.config.train_split],
                self.config.train_sample_size,
            ),
            validation=self._load_optional_huggingface_split(
                dataset,
                self.config.validation_split,
                self.config.validation_sample_size,
            ),
            test=self._select_records(
                dataset[self.config.test_split],
                self.config.test_sample_size,
            ),
        )

    def _load_local_bundle(self) -> DatasetBundle:
        self._ensure_local_dataset_exists()
        return DatasetBundle(
            train=self._load_local_file(
                self.config.local_train_path,
                self.config.train_sample_size,
            ),
            validation=self._load_optional_local_file(
                self.config.local_validation_path,
                self.config.validation_sample_size,
            ),
            test=self._load_local_file(
                self.config.local_test_path,
                self.config.test_sample_size,
            ),
        )

    def _ensure_local_dataset_exists(self) -> None:
        local_paths = [
            self.config.local_train_path,
            self.config.local_validation_path,
            self.config.local_test_path,
        ]
        if all(path and Path(path).exists() for path in local_paths):
            return

        if not self.config.auto_download_local_dataset:
            return

        if self.config.dataset_source != "local" or self.config.task_type != "absa":
            return

        dataset_id = self._AUTO_ABSA_DATASETS.get(self.config.dataset_name.lower())
        if dataset_id is None:
            return

        if not all(local_paths):
            raise ValueError(
                "local_train_path, local_validation_path, and local_test_path are "
                "required for auto-downloading a local ABSA dataset."
            )

        self._materialize_absa_dataset(dataset_id)

    def _materialize_absa_dataset(self, dataset_id: str) -> None:
        dataset = load_dataset(dataset_id)
        train_split = dataset["train"]
        test_split = dataset["test"]

        if "validation" in dataset:
            validation_split = dataset["validation"]
        else:
            split_dataset = train_split.train_test_split(
                test_size=self.config.local_validation_ratio,
                seed=self.config.shuffle_seed,
            )
            train_split = split_dataset["train"]
            validation_split = split_dataset["test"]

        self._write_jsonl_split(train_split, self.config.local_train_path)
        self._write_jsonl_split(validation_split, self.config.local_validation_path)
        self._write_jsonl_split(test_split, self.config.local_test_path)

    def _write_jsonl_split(self, split, destination: str | None) -> None:
        if not destination:
            raise ValueError("Destination path is required to write a local dataset split.")

        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(self._normalize_absa_record(dict(record)), ensure_ascii=False)
            for record in split
        ]
        path.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _normalize_absa_record(record: dict) -> dict[str, str]:
        text = str(record.get("sentence", record.get("text", ""))).strip()
        aspect = str(record.get("aspect", record.get("span", ""))).strip()
        label = str(record.get("label", "")).strip().lower()
        return {
            "sentence": text,
            "aspect": aspect,
            "label": label,
        }

    def _select_records(self, split, sample_size: int | None) -> list[TaskExample]:
        if sample_size is None:
            sample_count = len(split)
        else:
            sample_count = min(int(sample_size), len(split))
        sampled = split.shuffle(seed=self.config.shuffle_seed).select(range(sample_count))
        return [self._record_to_example(dict(record)) for record in sampled]

    def _load_local_file(
        self,
        file_path: str | None,
        sample_size: int | None,
    ) -> list[TaskExample]:
        if not file_path:
            raise ValueError("Local dataset path is required when dataset_source=local.")

        path = Path(file_path)
        if not path.exists():
            resolved_path = path.resolve(strict=False)
            raise FileNotFoundError(
                "Local dataset file not found: "
                f"'{file_path}' (resolved: '{resolved_path}'). "
                "Please add the dataset file or update the config paths."
            )
        if path.suffix == ".jsonl":
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        elif path.suffix == ".json":
            rows = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".csv":
            with path.open("r", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
        else:
            raise ValueError(f"Unsupported local dataset format: {path.suffix}")

        shuffled_rows = list(rows)
        random.Random(self.config.shuffle_seed).shuffle(shuffled_rows)
        selected_rows = (
            shuffled_rows
            if sample_size is None
            else shuffled_rows[: int(sample_size)]
        )
        return [self._record_to_example(row) for row in selected_rows]

    def _load_optional_huggingface_split(
        self,
        dataset,
        split_name: str | None,
        sample_size: int | None,
    ) -> list[TaskExample]:
        if not split_name or split_name not in dataset:
            return []
        return self._select_records(dataset[split_name], sample_size)

    def _load_optional_local_file(
        self,
        file_path: str | None,
        sample_size: int | None,
    ) -> list[TaskExample]:
        if not file_path:
            return []
        return self._load_local_file(file_path, sample_size)

    def _record_to_example(self, record: dict) -> TaskExample:
        label_value = record[self.config.label_field]
        if isinstance(label_value, int):
            label = self._map_numeric_label(label_value)
        else:
            label = str(label_value).strip().lower()

        aspect = None
        if self.config.aspect_field:
            aspect_value = record.get(self.config.aspect_field)
            aspect = str(aspect_value).strip() if aspect_value is not None else None

        return TaskExample(
            text=str(record[self.config.text_field]).strip(),
            label=label,
            aspect=aspect,
        )

    def _map_numeric_label(self, label_id: int) -> str:
        if self.config.label_space and 0 <= label_id < len(self.config.label_space):
            return self.config.label_space[label_id]
        if self.config.dataset_name == "glue" and self.config.dataset_subset == "sst2":
            return "positive" if label_id == 1 else "negative"
        return str(label_id)
