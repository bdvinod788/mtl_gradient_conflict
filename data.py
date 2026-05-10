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
    "yelp": 0,  # Yelp Polarity — binary sentiment classification
    "qnli": 1,  # GLUE QNLI — question-answer natural language inference
    "qqp":  2,  # GLUE QQP — question paraphrase detection
    "mnli": 3,  # GLUE MNLI — 3-class entailment
}

# HuggingFace loader config per task.
# Yelp Polarity has no validation split — a 10k holdout is carved from train.
# All four tasks: a per-task test split is also carved from train (since GLUE
# test labels are hidden behind the leaderboard). See TEST_SIZES below.
DATASET_CONFIG = {
    "yelp": {"hf_path": "yelp_polarity", "hf_name": None,   "val_split": "validation",         "test_split": "test"},
    "qnli": {"hf_path": "glue",          "hf_name": "qnli", "val_split": "validation",         "test_split": "test"},
    "qqp":  {"hf_path": "glue",          "hf_name": "qqp",  "val_split": "validation",         "test_split": "test"},
    "mnli": {"hf_path": "glue",          "hf_name": "mnli", "val_split": "validation_matched", "test_split": "test"},
}

YELP_VAL_SIZE = 10_000

# Per-task held-out test set sizes carved from the END of the training split
# (after a deterministic seeded shuffle). Sizes chosen proportionally — bigger
# tasks lose more, smaller tasks lose less. After carving:
#   yelp: 540k train (was 550k after val carve)
#   qnli:  99.7k train (was 104.7k)
#   qqp:  353.8k train (was 363.8k)
#   mnli: 382.7k train (was 392.7k)
TEST_SIZES = {
    "yelp": 10_000,
    "qnli":  5_000,
    "qqp":  10_000,
    "mnli": 10_000,
}

# Seed used to deterministically shuffle train before carving the test split.
# Keep this fixed across runs so the test set is the same every time.
TEST_CARVE_SEED = 42


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

    Tasks: Yelp Polarity (sentiment), QNLI, QQP, MNLI.
    Note: Yelp Polarity is loaded from "yelp_polarity"; GLUE tasks from "glue".
    Note: Yelp has no validation split — last YELP_VAL_SIZE examples of train
          are carved out and saved as a "validation" split.
    Note: GLUE test sets have hidden labels (leaderboard submission only), so
          for ALL tasks we additionally carve TEST_SIZES[task] examples from
          train as a labeled "test" split. The carve uses a deterministic
          seeded shuffle (TEST_CARVE_SEED) so the test set is reproducible.
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

        # ── Step 1: carve validation split for yelp (no native val split) ────
        # Yelp has no validation split — carve last YELP_VAL_SIZE rows from train
        if task_name == "yelp":
            full_train = tokenized["train"]
            total = len(full_train)
            tokenized["train"]      = full_train.select(range(0, total - YELP_VAL_SIZE))
            tokenized["validation"] = full_train.select(range(total - YELP_VAL_SIZE, total))

        # ── Step 2: carve test split from train (all tasks) ──────────────────
        # Shuffle train deterministically with TEST_CARVE_SEED so the carved
        # test set is a uniform random sample, not just whatever order HF
        # returned. Then select last TEST_SIZES[task] rows as test.
        n_test = TEST_SIZES[task_name]
        shuffled_train = tokenized["train"].shuffle(seed=TEST_CARVE_SEED)
        total = len(shuffled_train)
        if n_test >= total:
            raise ValueError(
                f"TEST_SIZES['{task_name}']={n_test} >= train size {total}. "
                f"Reduce the test carve size."
            )
        tokenized["train"] = shuffled_train.select(range(0, total - n_test))
        tokenized["test"]  = shuffled_train.select(range(total - n_test, total))
        print(f"  [carve] {task_name}: train {total} -> {total - n_test} train + {n_test} test")

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
    Returns {task_name: {"train": DataLoader, "val": DataLoader, "test": DataLoader}}
    for all 4 tasks.

    Val split is task-specific (see DATASET_CONFIG); test split is the
    held-out test set carved from train (see TEST_SIZES).
    """
    tokenizer = get_tokenizer(model_name)
    collator  = TaskAwareCollator(tokenizer)

    loaders = {}
    for task_name in TASK_NAME_TO_ID:
        ds         = load_processed_task(task_name, processed_dir)
        val_split  = DATASET_CONFIG[task_name]["val_split"]
        test_split = DATASET_CONFIG[task_name]["test_split"]

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

        # Test split was added in v2 of the data pipeline. If older processed
        # data on disk doesn't have a test split, skip it gracefully.
        if test_split in ds:
            loaders[task_name]["test"] = DataLoader(
                ds[test_split],
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=num_workers,
            )

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