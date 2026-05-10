import torch
import torch.nn as nn
from transformers import AutoModel

class MultiTaskModel(nn.Module):
    def __init__(self, model_name: str, task_num_labels: dict, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        self.heads = nn.ModuleDict({
            task: nn.Linear(hidden_size, num_labels)
            for task, num_labels in task_num_labels.items()
        })

    def forward(self, task_name: str, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
        )

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]

        pooled = self.dropout(pooled)
        logits = self.heads[task_name](pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
        }