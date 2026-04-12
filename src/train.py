import os
from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from config import Config, TASK_CONFIG
from data import make_single_task_dataloaders, make_multitask_train_iterator
from model import MultiTaskModel
from utils import set_seed, ensure_dir

def compute_accuracy(preds, labels):
    preds = np.argmax(preds, axis=1)
    return accuracy_score(labels, preds)

def evaluate(model, val_loaders, device):
    model.eval()
    results = {}

    with torch.no_grad():
        for task_name, loaders in val_loaders.items():
            val_loader = loaders["val"]
            all_logits = []
            all_labels = []
            total_loss = 0.0
            total_steps = 0

            for batch in tqdm(val_loader, desc=f"eval-{task_name}", leave=False):
                task_names = batch["task_name"]
                unique_task = task_names[0]
                assert all(t == unique_task for t in task_names)

                labels = batch["labels"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)

                out = model(
                    task_name=unique_task,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )

                total_loss += out["loss"].item()
                total_steps += 1
                all_logits.append(out["logits"].cpu().numpy())
                all_labels.append(labels.cpu().numpy())

            logits = np.concatenate(all_logits, axis=0)
            labels = np.concatenate(all_labels, axis=0)
            acc = compute_accuracy(logits, labels)

            results[task_name] = {
                "val_loss": total_loss / max(total_steps, 1),
                "accuracy": acc,
            }

    return results

def train_baseline(cfg: Config):
    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    task_num_labels = {k: v["num_labels"] for k, v in TASK_CONFIG.items()}
    model = MultiTaskModel(cfg.model_name, task_num_labels).to(device)

    val_loaders = make_single_task_dataloaders(
        model_name=cfg.model_name,
        processed_dir=cfg.processed_dir,
        train_batch_size=cfg.train_batch_size,
        eval_batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    multitask_iter = make_multitask_train_iterator(
        model_name=cfg.model_name,
        processed_dir=cfg.processed_dir,
        train_batch_size=cfg.train_batch_size,
        num_workers=cfg.num_workers,
    )

    loaders_for_len = make_single_task_dataloaders(
        model_name=cfg.model_name,
        processed_dir=cfg.processed_dir,
        train_batch_size=cfg.train_batch_size,
        eval_batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
    )
    steps_per_epoch = sum(len(v["train"]) for v in loaders_for_len.values())
    total_training_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(cfg.warmup_ratio * total_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    best_avg_acc = -1.0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        step_count = 0

        for batch in tqdm(multitask_iter, total=steps_per_epoch, desc=f"train-epoch-{epoch+1}"):
            task_names = batch["task_name"]
            unique_task = task_names[0]
            assert all(t == unique_task for t in task_names)

            labels = batch["labels"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            out = model(
                task_name=unique_task,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )

            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            step_count += 1

        print(f"\nEpoch {epoch+1} train_loss={epoch_loss / max(step_count, 1):.4f}")

        results = evaluate(model, val_loaders, device)
        avg_acc = np.mean([x["accuracy"] for x in results.values()])

        for task_name, metrics in results.items():
            print(
                f"{task_name}: val_loss={metrics['val_loss']:.4f} "
                f"acc={metrics['accuracy']:.4f}"
            )
        print(f"avg_acc={avg_acc:.4f}")

        ckpt_path = os.path.join(cfg.output_dir, f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_path = os.path.join(cfg.output_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            print(f"saved best model to {best_path}")

if __name__ == "__main__":
    cfg = Config()
    train_baseline(cfg)