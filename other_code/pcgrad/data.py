"""
data.py  (PCGrad version)

Loads tasks from HuggingFace (Yelp Polarity, QNLI, QQP, MNLI), tokenizes, and
saves to disk with a reproducible 60 / 20 / 20 train / val / test split carved
entirely from the HuggingFace training split (GLUE test sets have no labels).

Split ratios
------------
  train : 60 %   used during training
  val   : 20 %   used for mid-epoch and epoch-end evaluation
  test  : 20 %   held out — evaluated once after training completes

Identical interface to the previous version; `make_single_task_dataloaders`
now returns an extra "test" key in each task's dict.
"""

import os
import random
from typing import Dict, Iterator

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding

from config import TASK_CONFIG

TASK_NAME_TO_ID = {
    "yelp": 0,
    "qnli": 1,
    "qqp":  2,
    "mnli": 3,
}

DATASET_CONFIG = {
    "yelp": {"hf_path": "yelp_polarity", "hf_name": None},
    "qnli": {"hf_path": "glue",          "hf_name": "qnli"},
    "qqp":  {"hf_path": "glue",          "hf_name": "qqp"},
    "mnli": {"hf_path": "glue",          "hf_name": "mnli"},
}

# 60 / 20 / 20 split parameters
SPLIT_SEED   = 42
TEST_SIZE    = 0.20    # 20 % of full train → test
VAL_FRACTION = 0.25    # 25 % of remaining 80 % → val  (= 20 % of full)


def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def _preprocess_function_builder(task_name: str, tokenizer, max_length: int):
    key1, key2 = TASK_CONFIG[task_name]["input_keys"]

    def preprocess(example):
        text1 = example[key1]
        text2 = example[key2] if key2 is not None else None
        encoded = tokenizer(text1, text2, truncation=True, max_length=max_length)
        encoded["labels"]    = example["label"]
        encoded["task_name"] = task_name
        encoded["task_id"]   = TASK_NAME_TO_ID[task_name]
        return encoded

    return preprocess


def download_and_process_all(model_name: str, max_length: int, processed_dir: str):
    """
    Downloads, tokenizes, and splits every task into 60 / 20 / 20
    train / val / test from the HuggingFace training split only.
    Skips tasks already cached on disk.
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

        tokenized_train = raw_ds["train"].map(
            preprocess_fn,
            batched=False,
            remove_columns=raw_ds["train"].column_names,
        )

        # ── 60 / 20 / 20 split ────────────────────────────────────────────────
        # Step 1: carve 20 % as test
        split1     = tokenized_train.train_test_split(test_size=TEST_SIZE, seed=SPLIT_SEED)
        train_val  = split1["train"]   # 80 % of original
        test_split = split1["test"]    # 20 % of original

        # Step 2: split 80 % → 75 % train / 25 % val  (= 60 % / 20 % of original)
        split2      = train_val.train_test_split(test_size=VAL_FRACTION, seed=SPLIT_SEED)
        train_split = split2["train"]  # 60 % of original
        val_split   = split2["test"]   # 20 % of original

        print(
            f"[split] {task_name}: "
            f"train={len(train_split):,}  "
            f"val={len(val_split):,}  "
            f"test={len(test_split):,}"
        )

        DatasetDict({
            "train":      train_split,
            "validation": val_split,
            "test":       test_split,
        }).save_to_disk(save_path)
        print(f"[saved] {task_name} -> {save_path}")


def load_processed_task(task_name: str, processed_dir: str):
    path = os.path.join(processed_dir, task_name)
    return load_from_disk(path)


class TaskAwareCollator:
    """Dynamic padding via DataCollatorWithPadding; restores task_id / task_name."""

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
    using the pre-computed 60 / 20 / 20 splits.
    """
    tokenizer = get_tokenizer(model_name)
    collator  = TaskAwareCollator(tokenizer)

    loaders = {}
    for task_name in TASK_NAME_TO_ID:
        ds = load_processed_task(task_name, processed_dir)
        loaders[task_name] = {
            "train": DataLoader(
                ds["train"],
                batch_size=train_batch_size,
                shuffle=True,
                collate_fn=collator,
                num_workers=num_workers,
            ),
            "val": DataLoader(
                ds["validation"],
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=num_workers,
            ),
            "test": DataLoader(
                ds["test"],
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
        self.tasks        = list(task_loaders.keys())
        self._iterators   = {t: iter(l) for t, l in task_loaders.items()}

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
        return max(len(l) for l in self.task_loaders.values())


def make_multitask_train_iterator(
    model_name: str,
    processed_dir: str,
    train_batch_size: int,
    num_workers: int = 0,
) -> UniformMultiTaskIterator:
    loaders = make_single_task_dataloaders(
        model_name=model_name,
        processed_dir=processed_dir,
        train_batch_size=train_batch_size,
        eval_batch_size=train_batch_size,
        num_workers=num_workers,
    )
    return UniformMultiTaskIterator({k: v["train"] for k, v in loaders.items()})
