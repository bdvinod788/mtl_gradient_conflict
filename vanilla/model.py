"""
model.py
Vanilla MTL Model: Shared DistilBERT encoder + 4 task-specific classification heads.
Tasks: SST-2 (2-class), QNLI (2-class), QQP (2-class), MNLI (3-class)
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel

TASK_NUM_LABELS = {
    "sst2": 2,
    "qnli": 2,
    "qqp":  2,
    "mnli": 3,
}

TASKS = list(TASK_NUM_LABELS.keys())


class VanillaMTLModel(nn.Module):
    """
    Shared DistilBERT encoder with one linear classification head per task.
    Forward pass returns logits for the specified task.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", dropout: float = 0.1):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.dim  # 768

        # Task-specific heads
        self.heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_labels),
            )
            for task, num_labels in TASK_NUM_LABELS.items()
        })

    def forward(self, input_ids, attention_mask, task: str):
        """
        Args:
            input_ids:      (B, L)
            attention_mask: (B, L)
            task:           one of TASKS

        Returns:
            logits: (B, num_labels_for_task)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token representation
        cls_repr = outputs.last_hidden_state[:, 0, :]  # (B, H)
        logits = self.heads[task](cls_repr)
        return logits
