"""
train_vanilla_mtl_more_validations.py

Vanilla MTL Baseline with:
  - Shared DistilBERT encoder + 4 task heads (Yelp, QNLI, QQP, MNLI)
  - Loss = sum of per-task cross-entropy losses
  - Gradient normalisation by number of active tasks
  - Uniform task sampling
  - Per-task early stopping with head freezing
  - Mid-epoch validation via --eval_every
  - Gradient signals logged every epoch
  - --steps_per_epoch: fixed steps per epoch to control task imbalance
    (default 3000 — prevents smaller tasks from being over-trained
     relative to larger ones)

Bug fixes vs. original:
  - compute_per_task_grads now clones gradients immediately after backward,
    so stored tensors are independent of the model's grad buffers. Previously,
    later tasks' backward passes would overwrite earlier tasks' grads in-place,
    causing gradient_variance to be effectively zero.
  - log_gradient_signals calls model.zero_grad() once at the end (not inside
    the per-task loop) to leave the model in a clean state without interfering
    with the stored snapshots.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from model import VanillaMTLModel, TASK_NUM_LABELS
from data import (
    download_and_process_all,
    make_single_task_dataloaders,
    make_multitask_train_iterator,
    TASK_NAME_TO_ID,
)
from gradient_signals import compute_per_task_grads, compute_gradient_signals, combined_gradient_score

TASKS = list(TASK_NAME_TO_ID.keys())  # ["yelp", "qnli", "qqp", "mnli"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loaders: dict,
    loss_fn: nn.CrossEntropyLoss,
    device: torch.device,
) -> tuple[float, dict]:
    model.eval()
    per_task = {}
    for task in TASKS:
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for batch in val_loaders[task]:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            logits = model(input_ids, attention_mask, task)
            loss   = loss_fn(logits, labels)
            preds  = logits.argmax(dim=-1)
            total_loss    += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
        per_task[task] = {
            "loss": total_loss / total_samples,
            "acc":  total_correct / total_samples,
        }
    avg_val_loss = float(np.mean([m["loss"] for m in per_task.values()]))
    model.train()
    return avg_val_loss, per_task


def log_gradient_signals(
    model: nn.Module,
    train_loaders: dict,
    loss_fn: nn.CrossEntropyLoss,
    device: torch.device,
    n_batches: int = 4,
) -> dict:
    """
    Computes averaged gradient signals over n_batches rounds.

    Each round collects one gradient snapshot per task (cloned CPU tensors,
    independent of model grad buffers), then passes all snapshots to
    compute_gradient_signals. model.zero_grad() is called once at the end.

    Returns a dict with keys:
        conflict_rate, conflict_severity, gradient_variance,
        grad_norm_ratio, grad_snr, combined_gradient_score
    """
    model.eval()

    signal_keys = ["conflict_rate", "conflict_severity", "gradient_variance",
                   "grad_norm_ratio", "grad_snr"]
    acc = {k: 0.0 for k in signal_keys}
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
            task_grads[task] = compute_per_task_grads(
                model, batch, task, loss_fn, device
            )

        signals = compute_gradient_signals(task_grads)
        for k in signal_keys:
            acc[k] += signals[k]
        n_measurements += 1

    # Clean up model grad buffers once, after all snapshots are collected
    model.zero_grad()
    model.train()

    if n_measurements > 0:
        for k in signal_keys:
            acc[k] /= n_measurements

    acc["combined_gradient_score"] = combined_gradient_score(acc)
    return acc


def check_and_freeze(
    per_task_val: dict,
    best_per_task_loss: dict,
    patience_counters: dict,
    frozen_tasks: set,
    model: nn.Module,
    args: argparse.Namespace,
    label: str,
) -> tuple[list, bool]:
    """
    Per-task patience checking and head freezing.
    Called both mid-epoch and at epoch end.
    Returns (newly_frozen, any_improved).
    """
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
                print(f"  [{label}] Froze {task.upper()} head "
                      f"(best val loss: {best_per_task_loss[task]:.4f})")
    return newly_frozen, any_improved


def train(args: argparse.Namespace) -> list:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = VanillaMTLModel(model_name=args.model_name, dropout=args.dropout)
    model.to(device)

    # ── Data loading ───────────────────────────────────────────────────────────
    print("Preprocessing data (skips tasks already on disk)...")
    download_and_process_all(args.model_name, args.max_length, args.processed_dir)

    print("Loading dataloaders...")
    all_loaders = make_single_task_dataloaders(
        model_name=args.model_name,
        processed_dir=args.processed_dir,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size * 2,
        num_workers=args.num_workers,
    )
    train_loaders = {task: all_loaders[task]["train"] for task in TASKS}
    val_loaders   = {task: all_loaders[task]["val"]   for task in TASKS}

    sampler = make_multitask_train_iterator(
        model_name=args.model_name,
        processed_dir=args.processed_dir,
        train_batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Steps per epoch ────────────────────────────────────────────────────────
    if args.steps_per_epoch > 0:
        steps_per_epoch = args.steps_per_epoch
        print(f"Using fixed steps_per_epoch={steps_per_epoch}")
    else:
        steps_per_epoch = sampler.steps_per_epoch()
        print(f"Using dataset-size-based steps_per_epoch={steps_per_epoch}")

    total_steps = steps_per_epoch * args.num_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
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
    frozen_tasks: set  = set()
    history: list      = []
    global_step        = 0

    print(f"\nStarting Vanilla MTL training for {args.num_epochs} epochs "
          f"({steps_per_epoch} steps/epoch)")
    print(f"Tasks: {TASKS}")
    if args.eval_every > 0:
        evals_per_epoch = steps_per_epoch // args.eval_every
        print(f"Mid-epoch validation every {args.eval_every} steps "
              f"({evals_per_epoch} checks per epoch)")
    print()

    sampler_iter = iter(sampler)

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_start      = time.time()
        running_loss     = {task: 0.0 for task in TASKS}
        running_steps    = {task: 0   for task in TASKS}
        mid_epoch_checks = []

        for step in range(steps_per_epoch):
            batch = next(sampler_iter)
            task  = batch["task_name"][0]

            if task in frozen_tasks:
                continue

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, task)
            loss   = loss_fn(logits, labels)

            n_active = len(TASKS) - len(frozen_tasks)
            loss     = loss / max(n_active, 1)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            running_loss[task]  += loss.item() * max(n_active, 1)
            running_steps[task] += 1
            global_step         += 1

            # ── Step logging ──────────────────────────────────────────────────
            if (step + 1) % args.log_every == 0:
                avg_losses = {
                    t: running_loss[t] / max(running_steps[t], 1) for t in TASKS
                }
                frozen_str = (
                    f" [frozen: {', '.join(sorted(frozen_tasks))}]"
                    if frozen_tasks else ""
                )
                print(f"  Epoch {epoch} | Step {step+1}/{steps_per_epoch}{frozen_str} | "
                      + " | ".join(f"{t}: {avg_losses[t]:.4f}" for t in TASKS))

            # ── Mid-epoch validation ──────────────────────────────────────────
            if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
                train_loss_snapshot = {
                    t: round(running_loss[t] / max(running_steps[t], 1), 6)
                    for t in TASKS
                }
                avg_val_loss, per_task_val = evaluate(
                    model, val_loaders, loss_fn, device
                )

                print(f"\n  [Mid-epoch step {step+1}] Val check:")
                for t, m in per_task_val.items():
                    marker = "❄" if t in frozen_tasks else f"p{patience_counters[t]}"
                    tl = train_loss_snapshot[t]
                    print(f"    {t.upper():6s} -> train: {tl:.4f}  "
                          f"val: {m['loss']:.4f}  acc: {m['acc']*100:.2f}%  [{marker}]")

                print(f"  Computing gradient signals...")
                grad_signals = log_gradient_signals(
                    model, train_loaders, loss_fn, device,
                    n_batches=args.grad_signal_batches,
                )
                print(f"  Gradient Signals -> "
                      f"conflict_rate: {grad_signals['conflict_rate']:.4f}  "
                      f"severity: {grad_signals['conflict_severity']:.4f}  "
                      f"variance: {grad_signals['gradient_variance']:.6f}  "
                      f"norm_ratio: {grad_signals['grad_norm_ratio']:.4f}  "
                      f"snr: {grad_signals['grad_snr']:.4f}  "
                      f"score: {grad_signals['combined_gradient_score']:.4f}")

                newly_frozen, any_improved = check_and_freeze(
                    per_task_val, best_per_task_loss, patience_counters,
                    frozen_tasks, model, args, label=f"step {step+1}",
                )

                if any_improved:
                    torch.save(model.state_dict(), output_dir / "best_model.pt")
                    print("  Model saved.")

                mid_epoch_checks.append({
                    "step":              step + 1,
                    "global_step":       global_step,
                    "train_loss":        train_loss_snapshot,
                    "avg_val_loss":      avg_val_loss,
                    "per_task_val":      per_task_val,
                    "frozen_tasks":      list(frozen_tasks),
                    "patience_counters": dict(patience_counters),
                    **grad_signals,
                })

                if frozen_tasks == set(TASKS):
                    break

                model.train()

        # ── End of epoch ───────────────────────────────────────────────────────
        avg_val_loss, per_task_val = evaluate(model, val_loaders, loss_fn, device)

        print("  Computing gradient signals...")
        grad_signals = log_gradient_signals(
            model, train_loaders, loss_fn, device,
            n_batches=args.grad_signal_batches,
        )

        epoch_time = time.time() - epoch_start

        newly_frozen, any_improved = check_and_freeze(
            per_task_val, best_per_task_loss, patience_counters,
            frozen_tasks, model, args, label=f"epoch {epoch}",
        )

        # ── Epoch summary ──────────────────────────────────────────────────────
        print(f"\nEpoch {epoch}/{args.num_epochs} | "
              f"Avg Val Loss: {avg_val_loss:.4f} | Time: {epoch_time:.1f}s")
        for task, m in per_task_val.items():
            if task in frozen_tasks and task not in newly_frozen:
                status = "❄ frozen"
            elif task in newly_frozen:
                status = "❄ just frozen"
            else:
                status = f"patience {patience_counters[task]}/{args.patience}"
            print(f"  {task.upper():6s} -> loss: {m['loss']:.4f}  "
                  f"acc: {m['acc']*100:.2f}%  [{status}]")
        print(f"  Gradient Signals -> "
              f"conflict_rate: {grad_signals['conflict_rate']:.4f}  "
              f"severity: {grad_signals['conflict_severity']:.4f}  "
              f"variance: {grad_signals['gradient_variance']:.6f}  "
              f"norm_ratio: {grad_signals['grad_norm_ratio']:.4f}  "
              f"snr: {grad_signals['grad_snr']:.4f}  "
              f"score: {grad_signals['combined_gradient_score']:.4f}")

        if any_improved:
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print("  Model saved.")

        avg_train_loss = {
            t: round(running_loss[t] / max(running_steps[t], 1), 6)
            for t in TASKS
        }

        history.append({
            "epoch":             epoch,
            "avg_train_loss":    avg_train_loss,
            "avg_val_loss":      avg_val_loss,
            "per_task_val":      per_task_val,
            "best_per_task_loss": dict(best_per_task_loss),
            "patience_counters": dict(patience_counters),
            "frozen_tasks":      list(frozen_tasks),
            "mid_epoch_checks":  mid_epoch_checks,
            "epoch_time_s":      round(epoch_time, 1),
            **grad_signals,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vanilla MTL Training")
    parser.add_argument("--model_name",          default="distilbert-base-uncased")
    parser.add_argument("--dropout",             type=float, default=0.1)
    parser.add_argument("--max_length",          type=int,   default=128)
    parser.add_argument("--processed_dir",       default="./processed_data")
    parser.add_argument("--num_workers",         type=int,   default=2)
    parser.add_argument("--batch_size",          type=int,   default=32)
    parser.add_argument("--num_epochs",          type=int,   default=20)
    parser.add_argument("--lr",                  type=float, default=2e-5)
    parser.add_argument("--weight_decay",        type=float, default=0.01)
    parser.add_argument("--max_grad_norm",       type=float, default=1.0)
    parser.add_argument("--patience",            type=int,   default=3)
    parser.add_argument("--min_delta",           type=float, default=1e-4)
    parser.add_argument("--grad_signal_batches", type=int,   default=4)
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument("--output_dir",          default="./outputs/vanilla_mtl")
    parser.add_argument("--log_every",           type=int,   default=100)
    parser.add_argument("--eval_every",          type=int,   default=1000)
    parser.add_argument("--steps_per_epoch",     type=int,   default=3000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("Vanilla MTL — Baseline")
    print("=" * 60)
    print(vars(args))
    print()
    train(args)