from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    train_batch_size: int = 16
    eval_batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 2
    warmup_ratio: float = 0.1
    seed: int = 42
    num_workers: int = 0
    output_dir: str = "checkpoints"
    processed_dir: str = "processed"

TASK_CONFIG = {
    "yelp": {
        "input_keys": ("text", None),
        "num_labels": 2,
    },
    "qnli": {
        "input_keys": ("question", "sentence"),
        "num_labels": 2,
    },
    "qqp": {
        "input_keys": ("question1", "question2"),
        "num_labels": 2,
    },
    "mnli": {
        "input_keys": ("premise", "hypothesis"),
        "num_labels": 3,
    },
}