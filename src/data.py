import os
from typing import Dict, Iterator
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding

from config import TASK_CONFIG

TASK_NAME_TO_ID = {
    "sst2": 0,
    "cola": 1,
    "qqp": 2,
    "mnli": 3,
}


def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def _preprocess_function_builder(task_name: str, tokenizer, max_length: int):
    key1, key2 = TASK_CONFIG[task_name]["input_keys"]

    def preprocess(example):
        text1 = example[key1]
        text2 = example[key2] if key2 is not None else None

        encoded = tokenizer(
            text1,
            text2,
            truncation=True,
            max_length=max_length,
        )

        encoded["labels"] = example["label"]
        encoded["task_name"] = task_name
        encoded["task_id"] = TASK_NAME_TO_ID[task_name]
        return encoded

    return preprocess


def download_and_process_all(model_name: str, max_length: int, processed_dir: str):
    tokenizer = get_tokenizer(model_name)
    os.makedirs(processed_dir, exist_ok=True)

    for task_name in ["sst2", "cola", "qqp", "mnli"]:
        save_path = os.path.join(processed_dir, task_name)

        if os.path.exists(save_path):
            print(f"[skip] {task_name} already processed at {save_path}")
            continue

        print(f"[load] {task_name}")
        raw_ds = load_dataset("glue", task_name)

        preprocess_fn = _preprocess_function_builder(task_name, tokenizer, max_length)

        tokenized = raw_ds.map(
            preprocess_fn,
            batched=False,
            remove_columns=raw_ds["train"].column_names,
        )

        tokenized.save_to_disk(save_path)
        print(f"[saved] {task_name} -> {save_path}")


def load_processed_task(task_name: str, processed_dir: str):
    path = os.path.join(processed_dir, task_name)
    return load_from_disk(path)


class TaskAwareCollator:
    def __init__(self, tokenizer):
        self.base_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features):
        task_ids = [f["task_id"] for f in features]
        task_names = [f["task_name"] for f in features]

        stripped = []
        for f in features:
            x = dict(f)
            x.pop("task_name", None)
            x.pop("task_id", None)
            stripped.append(x)

        batch = self.base_collator(stripped)
        batch["task_id"] = torch.tensor(task_ids, dtype=torch.long)
        batch["task_name"] = task_names
        return batch


def make_single_task_dataloaders(
    model_name: str,
    processed_dir: str,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int = 0,
):
    tokenizer = get_tokenizer(model_name)
    collator = TaskAwareCollator(tokenizer)

    loaders = {}
    for task_name in ["sst2", "cola", "qqp", "mnli"]:
        ds = load_processed_task(task_name, processed_dir)

        train_ds = ds["train"]
        if task_name == "mnli":
            val_ds = ds["validation_matched"]
        else:
            val_ds = ds["validation"]

        loaders[task_name] = {
            "train": DataLoader(
                train_ds,
                batch_size=train_batch_size,
                shuffle=True,
                collate_fn=collator,
                num_workers=num_workers,
            ),
            "val": DataLoader(
                val_ds,
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=num_workers,
            ),
        }

    return loaders


class RoundRobinMultiTaskIterator:
    def __init__(self, task_loaders: Dict[str, DataLoader]):
        self.task_names = list(task_loaders.keys())
        self.task_loaders = task_loaders

    def __iter__(self) -> Iterator:
        iterators = {k: iter(v) for k, v in self.task_loaders.items()}
        finished = set()

        while len(finished) < len(self.task_names):
            for task_name in self.task_names:
                if task_name in finished:
                    continue
                try:
                    yield next(iterators[task_name])
                except StopIteration:
                    finished.add(task_name)


def make_multitask_train_iterator(
    model_name: str,
    processed_dir: str,
    train_batch_size: int,
    num_workers: int = 0,
):
    loaders = make_single_task_dataloaders(
        model_name=model_name,
        processed_dir=processed_dir,
        train_batch_size=train_batch_size,
        eval_batch_size=train_batch_size,
        num_workers=num_workers,
    )
    train_loaders = {k: v["train"] for k, v in loaders.items()}
    return RoundRobinMultiTaskIterator(train_loaders)