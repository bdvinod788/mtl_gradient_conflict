"""
train_vanilla_mtl.py

Vanilla MTL Baseline (Experiment 1):
  - Shared DistilBERT encoder + 4 task heads (SST-2, QNLI, QQP, MNLI)
  - Loss = sum of per-task cross-entropy losses
  - Gradient normalisation by number of active tasks
  - Uniform task sampling to handle dataset size imbalance
  - Per-task early stopping with head freezing
  - Gradient signals logged every epoch
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import DistilBertTokenizerFast, get_linear_schedule_with_warmup

from model import VanillaMTLModel, TASKS, TASK_NUM_LABELS
from data import build_task_dataloaders, UniformMTLSampler
from gradient_signals import compute_per_task_grads, compute_gradient_signals, combined_gradient_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, val_loaders, loss_fn, device):
    model.eval()
    per_task = {}
    for task in TASKS:
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for batch in val_loaders[task]:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask, task)
            loss = loss_fn(logits, labels)
            preds = logits.argmax(dim=-1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
        per_task[task] = {
            "loss": total_loss / total_samples,
            "acc":  total_correct / total_samples,
        }
    avg_val_loss = float(np.mean([m["loss"] for m in per_task.values()]))
    model.train()
    return avg_val_loss, per_task


def log_gradient_signals(model, train_loaders, loss_fn, device, n_batches=8):
    model.eval()
    acc_rate, acc_sev, acc_var = 0.0, 0.0, 0.0
    n_measurements = 0
    task_iters = {task: iter(loader) for task, loader in train_loaders.items()}
    for _ in range(n_batches):
        task_grads = {}
        for task in TASKS:
            try:
                batch = next(task_iters[task])
            except StopIteration:
                task_iters[task] = iter(train_loaders[task])
                batch = next(task_iters[task])
            grads = compute_per_task_grads(model, batch, task, loss_fn, device)
            task_grads[task] = grads
        rate, sev, var = compute_gradient_signals(task_grads)
        acc_rate += rate
        acc_sev  += sev
        acc_var  += var
        n_measurements += 1
    if n_measurements > 0:
        acc_rate /= n_measurements
        acc_sev  /= n_measurements
        acc_var  /= n_measurements
    model.zero_grad()
    model.train()
    score = combined_gradient_score(acc_rate, acc_sev, acc_var)
    return acc_rate, acc_sev, acc_var, score


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)
    model = VanillaMTLModel(model_name=args.model_name, dropout=args.dropout)
    model.to(device)

    print("Loading training data...")
    train_loaders = build_task_dataloaders(
        tokenizer, split="train", batch_size=args.batch_size,
        max_length=args.max_length, max_samples_per_task=args.max_train_samples,
        num_workers=args.num_workers,
    )

    print("Loading validation data...")
    val_loaders = build_task_dataloaders(
        tokenizer, split="validation", batch_size=args.batch_size * 2,
        max_length=args.max_length, max_samples_per_task=args.max_val_samples,
        num_workers=args.num_workers,
    )

    sampler = UniformMTLSampler(train_loaders)
    steps_per_epoch = sampler.steps_per_epoch()
    total_steps = steps_per_epoch * args.num_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(params, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    loss_fn = nn.CrossEntropyLoss()

    # Per-task early stopping state
    best_per_task_loss = {task: float("inf") for task in TASKS}
    patience_counters  = {task: 0 for task in TASKS}
    frozen_tasks       = set()
    history            = []
    global_step        = 0

    print(f"\nStarting Vanilla MTL training for {args.num_epochs} epochs ({steps_per_epoch} steps/epoch)")
    print(f"Tasks: {TASKS}\n")

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss  = {task: 0.0 for task in TASKS}
        running_steps = {task: 0   for task in TASKS}
        sampler_iter  = iter(sampler)

        for step in range(steps_per_epoch):
            task, batch = next(sampler_iter)

            # Skip frozen tasks
            if task in frozen_tasks:
                continue

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, task)
            loss   = loss_fn(logits, labels)

            n_active = len(TASKS) - len(frozen_tasks)
            loss = loss / max(n_active, 1)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            running_loss[task]  += loss.item() * max(n_active, 1)
            running_steps[task] += 1
            global_step         += 1

            if (step + 1) % args.log_every == 0:
                avg_losses = {t: running_loss[t] / max(running_steps[t], 1) for t in TASKS}
                frozen_str = f" [frozen: {', '.join(sorted(frozen_tasks))}]" if frozen_tasks else ""
                print(f"  Epoch {epoch} | Step {step+1}/{steps_per_epoch}{frozen_str} | "
                      + " | ".join(f"{t}: {avg_losses[t]:.4f}" for t in TASKS))

        # Evaluate
        avg_val_loss, per_task_val = evaluate(model, val_loaders, loss_fn, device)

        # Gradient signals
        print("  Computing gradient signals...")
        rate, sev, var, score = log_gradient_signals(
            model, train_loaders, loss_fn, device, n_batches=args.grad_signal_batches
        )

        epoch_time = time.time() - epoch_start

        # Per-task patience + head freezing
        newly_frozen = []
        any_improved = False
        for task in TASKS:
            if task in frozen_tasks:
                continue
            task_val_loss = per_task_val[task]["loss"]
            if task_val_loss < best_per_task_loss[task] - args.min_delta:
                best_per_task_loss[task] = task_val_loss
                patience_counters[task]  = 0
                any_improved             = True
            else:
                patience_counters[task] += 1
                if patience_counters[task] >= args.patience:
                    for param in model.heads[task].parameters():
                        param.requires_grad = False
                    frozen_tasks.add(task)
                    newly_frozen.append(task)

        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.num_epochs} | Avg Val Loss: {avg_val_loss:.4f} | Time: {epoch_time:.1f}s")
        for task, m in per_task_val.items():
            if task in frozen_tasks and task not in newly_frozen:
                status = "❄ frozen"
            elif task in newly_frozen:
                status = "❄ just frozen"
            else:
                status = f"patience {patience_counters[task]}/{args.patience}"
            print(f"  {task.upper():6s} -> loss: {m['loss']:.4f}  acc: {m['acc']*100:.2f}%  [{status}]")
        print(f"  Gradient Signals -> conflict_rate: {rate:.4f}  severity: {sev:.4f}  variance: {var:.6f}  score: {score:.4f}")

        for task in newly_frozen:
            print(f"  Froze {task.upper()} head (best val loss: {best_per_task_loss[task]:.4f})")

        if any_improved:
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  Model saved.")

        history.append({
            "epoch":                      epoch,
            "avg_val_loss":               avg_val_loss,
            "per_task_val":               per_task_val,
            "best_per_task_loss":         dict(best_per_task_loss),
            "patience_counters":          dict(patience_counters),
            "frozen_tasks":               list(frozen_tasks),
            "gradient_conflict_rate":     rate,
            "gradient_conflict_severity": sev,
            "gradient_variance":          var,
            "combined_gradient_score":    score,
            "epoch_time_s":               round(epoch_time, 1),
        })

        if frozen_tasks == set(TASKS):
            print(f"\nAll tasks frozen. Training complete at epoch {epoch}.")
            break

        print()

    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. History saved to {history_path}")
    return history


@torch.no_grad()
def predict(model, texts, task, tokenizer, device, batch_size=32, max_length=128):
    model.eval()
    all_preds, all_probs = [], []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i: i + batch_size]
        if isinstance(chunk[0], (list, tuple)):
            sents_a = [x[0] for x in chunk]
            sents_b = [x[1] for x in chunk]
        else:
            sents_a = chunk
            sents_b = None
        enc = tokenizer(sents_a, sents_b, max_length=max_length, padding="max_length",
                        truncation=True, return_tensors="pt")
        logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device), task)
        all_probs.extend(torch.softmax(logits, dim=-1).cpu().tolist())
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    return all_preds, all_probs


def parse_args():
    parser = argparse.ArgumentParser(description="Vanilla MTL Training (CSCI 567)")
    parser.add_argument("--model_name",          default="distilbert-base-uncased")
    parser.add_argument("--dropout",             type=float, default=0.1)
    parser.add_argument("--max_length",          type=int,   default=128)
    parser.add_argument("--max_train_samples",   type=int,   default=None)
    parser.add_argument("--max_val_samples",     type=int,   default=None)
    parser.add_argument("--num_workers",         type=int,   default=2)
    parser.add_argument("--batch_size",          type=int,   default=32)
    parser.add_argument("--num_epochs",          type=int,   default=20)
    parser.add_argument("--lr",                  type=float, default=2e-5)
    parser.add_argument("--weight_decay",        type=float, default=0.01)
    parser.add_argument("--max_grad_norm",       type=float, default=1.0)
    parser.add_argument("--patience",            type=int,   default=3)
    parser.add_argument("--min_delta",           type=float, default=1e-4)
    parser.add_argument("--grad_signal_batches", type=int,   default=8)
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument("--output_dir",          default="./outputs/vanilla_mtl")
    parser.add_argument("--log_every",           type=int,   default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("Vanilla MTL — Baseline (CSCI 567)")
    print("=" * 60)
    print(vars(args))
    print()
    train(args)
