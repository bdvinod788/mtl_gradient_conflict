"""
data.py
Loads GLUE tasks from HuggingFace datasets and provides a uniform-sampling
multi-task dataloader that cycles through tasks to handle size imbalance.
"""

from __future__ import annotations
import random
from itertools import cycle
from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import DistilBertTokenizerFast

from model import TASKS

# ── Task-specific field extractors ────────────────────────────────────────────
# Each returns (sentence_a, sentence_b_or_None, label)

def _extract_sst2(example):
    return example["sentence"], None, example["label"]

def _extract_qnli(example):
    return example["question"], example["sentence"], example["label"]

def _extract_qqp(example):
    return example["question1"], example["question2"], example["label"]

def _extract_mnli(example):
    return example["premise"], example["hypothesis"], example["label"]

EXTRACTORS = {
    "sst2": _extract_sst2,
    "qnli": _extract_qnli,
    "qqp":  _extract_qqp,
    "mnli": _extract_mnli,
}

# HuggingFace dataset names & config keys
HF_CONFIGS = {
    "sst2": ("glue", "sst2"),
    "qnli": ("glue", "qnli"),
    "qqp":  ("glue", "qqp"),
    "mnli": ("glue", "mnli"),
}

# MNLI validation split name
MNLI_VAL_SPLIT = "validation_matched"


class GLUETaskDataset(Dataset):
    """
    Wraps a single GLUE task split as a torch Dataset.
    Tokenization is LAZY — done per example in __getitem__ rather than
    upfront in __init__. This keeps RAM usage flat regardless of dataset
    size, fixing OOM kills on large tasks like QQP and MNLI.
    """

    def __init__(
        self,
        task: str,
        split: str,
        tokenizer: DistilBertTokenizerFast,
        max_length: int = 128,
        max_samples: Optional[int] = None,
    ):
        hf_path, hf_name = HF_CONFIGS[task]

        actual_split = split
        if task == "mnli" and split == "validation":
            actual_split = MNLI_VAL_SPLIT

        raw = load_dataset(hf_path, hf_name, split=actual_split)

        if max_samples is not None:
            raw = raw.select(range(min(max_samples, len(raw))))

        # Filter out unlabelled test rows (label == -1) upfront
        raw = raw.filter(lambda ex: EXTRACTORS[task](ex)[2] != -1)

        # Store raw text + labels only — no tokenization yet
        self.raw = raw
        self.task = task
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.extractor = EXTRACTORS[task]

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        ex = self.raw[idx]
        sent_a, sent_b, label = self.extractor(ex)

        enc = self.tokenizer(
            sent_a,
            sent_b,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(label, dtype=torch.long),
        }


def build_task_dataloaders(
    tokenizer: DistilBertTokenizerFast,
    split: str = "train",
    batch_size: int = 32,
    max_length: int = 128,
    max_samples_per_task: Optional[int] = None,
    num_workers: int = 2,
) -> Dict[str, DataLoader]:
    """Returns a dict of {task: DataLoader} for the given split."""
    loaders = {}
    for task in TASKS:
        ds = GLUETaskDataset(
            task=task,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            max_samples=max_samples_per_task,
        )
        shuffle = (split == "train")
        loaders[task] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


class UniformMTLSampler:
    """
    Uniform task sampler: at each step, randomly pick a task and yield
    the next batch from that task's dataloader (cycling infinitely).
    Handles dataset size imbalance as described in the proposal.
    """

    def __init__(self, task_loaders: Dict[str, DataLoader]):
        self.task_loaders = task_loaders
        self._iterators: Dict[str, Iterator] = {
            task: cycle(loader) for task, loader in task_loaders.items()
        }
        self.tasks = list(task_loaders.keys())

    def __iter__(self):
        return self

    def __next__(self):
        task = random.choice(self.tasks)
        batch = next(self._iterators[task])
        return task, batch

    def steps_per_epoch(self) -> int:
        """
        One epoch = one pass through the largest task dataset.
        """
        return max(len(loader) for loader in self.task_loaders.values())
