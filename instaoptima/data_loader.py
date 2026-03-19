import csv
import json
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
            validation=self._select_records(
                dataset[self.config.validation_split],
                self.config.validation_sample_size,
            ),
            test=self._select_records(
                dataset[self.config.test_split],
                self.config.test_sample_size,
            ),
        )

    def _load_local_bundle(self) -> DatasetBundle:
        return DatasetBundle(
            train=self._load_local_file(
                self.config.local_train_path,
                self.config.train_sample_size,
            ),
            validation=self._load_local_file(
                self.config.local_validation_path,
                self.config.validation_sample_size,
            ),
            test=self._load_local_file(
                self.config.local_test_path,
                self.config.test_sample_size,
            ),
        )

    def _select_records(self, split, sample_size: int) -> list[TaskExample]:
        sample_count = min(sample_size, len(split))
        sampled = split.shuffle(seed=self.config.shuffle_seed).select(range(sample_count))
        return [self._record_to_example(dict(record)) for record in sampled]

    def _load_local_file(
        self,
        file_path: str | None,
        sample_size: int,
    ) -> list[TaskExample]:
        if not file_path:
            raise ValueError("Local dataset path is required when dataset_source=local.")

        path = Path(file_path)
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

        return [self._record_to_example(row) for row in rows[:sample_size]]

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
