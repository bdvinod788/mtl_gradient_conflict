import os
import random
from itertools import cycle
from typing import Dict, Iterator

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding

from config import TASK_CONFIG

TASK_NAME_TO_ID = {
    "yelp": 0,  # replaces sst2 - Yelp Polarity sentiment classification
    "cola": 1,
    "qqp":  2,
    "mnli": 3,
}

# HuggingFace loader config per task.
# Yelp Polarity has its own path and no validation split (use "test" as proxy).
DATASET_CONFIG = {
    "yelp": {"hf_path": "yelp_polarity", "hf_name": None,   "val_split": "test"},
    "cola": {"hf_path": "glue",          "hf_name": "cola", "val_split": "validation"},
    "qqp":  {"hf_path": "glue",          "hf_name": "qqp",  "val_split": "validation"},
    "mnli": {"hf_path": "glue",          "hf_name": "mnli", "val_split": "validation_matched"},
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

        encoded["labels"]    = example["label"]
        encoded["task_name"] = task_name
        encoded["task_id"]   = TASK_NAME_TO_ID[task_name]
        return encoded

    return preprocess


def download_and_process_all(model_name: str, max_length: int, processed_dir: str):
    """
    Downloads and tokenizes all 4 tasks, saves to disk.
    Skips tasks that are already processed.

    Tasks: Yelp Polarity (sentiment), CoLA, QQP, MNLI.
    Note: Yelp Polarity is loaded from "yelp_polarity"; GLUE tasks from "glue".
    """
    tokenizer = get_tokenizer(model_name)
    os.makedirs(processed_dir, exist_ok=True)

    for task_name in TASK_NAME_TO_ID:
        save_path = os.path.join(processed_dir, task_name)

        if os.path.exists(save_path):
            print(f"[skip] {task_name} already processed at {save_path}")
            continue

        print(f"[load] {task_name}")
        ds_cfg = DATASET_CONFIG[task_name]

        if ds_cfg["hf_name"] is not None:
            raw_ds = load_dataset(ds_cfg["hf_path"], ds_cfg["hf_name"])
        else:
            raw_ds = load_dataset(ds_cfg["hf_path"])

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
    """
    Handles dynamic padding via DataCollatorWithPadding and
    restores task_id / task_name into the final batch.
    """
    def __init__(self, tokenizer):
        self.base_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features):
        task_ids   = [f["task_id"]   for f in features]
        task_names = [f["task_name"] for f in features]

        stripped = []
        for f in features:
            x = dict(f)
            x.pop("task_name", None)
            x.pop("task_id",   None)
            stripped.append(x)

        batch = self.base_collator(stripped)
        batch["task_id"]   = torch.tensor(task_ids, dtype=torch.long)
        batch["task_name"] = task_names
        return batch


def make_single_task_dataloaders(
    model_name: str,
    processed_dir: str,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int = 0,
) -> Dict[str, Dict[str, DataLoader]]:
    """
    Returns {task_name: {"train": DataLoader, "val": DataLoader}}
    for all 4 tasks. Val split is task-specific (see DATASET_CONFIG).
    """
    tokenizer = get_tokenizer(model_name)
    collator  = TaskAwareCollator(tokenizer)

    loaders = {}
    for task_name in TASK_NAME_TO_ID:
        ds        = load_processed_task(task_name, processed_dir)
        val_split = DATASET_CONFIG[task_name]["val_split"]

        loaders[task_name] = {
            "train": DataLoader(
                ds["train"],
                batch_size=train_batch_size,
                shuffle=True,
                collate_fn=collator,
                num_workers=num_workers,
            ),
            "val": DataLoader(
                ds[val_split],
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=num_workers,
            ),
        }

    return loaders


class UniformMultiTaskIterator:
    def __init__(self, task_loaders: Dict[str, DataLoader]):
        self.task_loaders = task_loaders
        self.tasks = list(task_loaders.keys())
        self._iterators = {task: iter(loader) for task, loader in task_loaders.items()}

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        task = random.choice(self.tasks)
        try:
            batch = next(self._iterators[task])
        except StopIteration:
            self._iterators[task] = iter(self.task_loaders[task])
            batch = next(self._iterators[task])
        return batch

    def steps_per_epoch(self) -> int:
        return max(len(loader) for loader in self.task_loaders.values())


def make_multitask_train_iterator(
    model_name: str,
    processed_dir: str,
    train_batch_size: int,
    num_workers: int = 0,
) -> UniformMultiTaskIterator:
    """
    Builds per-task train dataloaders and wraps them in
    UniformMultiTaskIterator for balanced multitask training.
    """
    loaders = make_single_task_dataloaders(
        model_name=model_name,
        processed_dir=processed_dir,
        train_batch_size=train_batch_size,
        eval_batch_size=train_batch_size,
        num_workers=num_workers,
    )
    train_loaders = {k: v["train"] for k, v in loaders.items()}
    return UniformMultiTaskIterator(train_loaders)