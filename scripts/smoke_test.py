import sys
sys.path.append("src")

from config import Config
from data import make_single_task_dataloaders

if __name__ == "__main__":
    cfg = Config()
    loaders = make_single_task_dataloaders(
        model_name=cfg.model_name,
        processed_dir=cfg.processed_dir,
        train_batch_size=4,
        eval_batch_size=4,
        num_workers=0,
    )

    for task_name, bundle in loaders.items():
        batch = next(iter(bundle["train"]))
        print(f"\nTASK: {task_name}")
        for k, v in batch.items():
            if hasattr(v, "shape"):
                print(k, v.shape)
            else:
                print(k, v[:2] if isinstance(v, list) else v)