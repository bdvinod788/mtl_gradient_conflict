"""
train_pcgrad_mtl.py

PCGrad MTL (Experiment 2):
  - Shared DistilBERT encoder + 4 task heads (Yelp, QNLI, QQP, MNLI)
  - At each step: one batch per ACTIVE task; gradients projected via PCGrad
    (Gradient Surgery) before each parameter update.
  - Per-task early stopping + head freezing — identical protocol to vanilla.
  - Gradient signals (conflict rate, severity, variance, norm ratio, SNR) logged every epoch.
  - Weights & Biases logging built in (disable with --no_wandb).

Key difference from vanilla:
  Vanilla : sample ONE task → forward → loss/n_active → backward → step
  PCGrad  : collect ONE BATCH per active task → forward all → pc_backward
            (projects each task's gradient to remove conflicting components) → step

Smoke-test (small sample, local):
    python train_pcgrad_mtl.py \\
        --max_train_samples 500 --max_val_samples 200 \\
        --batch_size 16 --num_epochs 2 --num_workers 0 \\
        --steps_per_epoch 50 --eval_every 25 \\
        --grad_signal_batches 2 \\
        --output_dir ./outputs/smoke_test \\
        --wandb_project csci567-mtl --wandb_run pcgrad_smoke

Full run on CARC:
    python train_pcgrad_mtl.py \\
        --batch_size 32 --num_epochs 20 --num_workers 4 \\
        --grad_signal_batches 4 --steps_per_epoch 7000 \\
        --eval_every 1000 --patience 5 \\
        --output_dir /scratch1/$USER/mtl_outputs/pcgrad_mtl_<date> \\
        --wandb_project csci567-mtl --wandb_run pcgrad_full
"""

import argparse
import json
import random
import time
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# W&B — imported lazily so the script still works without it
# (pass --no_wandb to skip entirely)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# All imports are local to this folder — zero cross-directory dependencies
from pcgrad import PCGrad
from model import VanillaMTLModel, TASKS, TASK_NUM_LABELS
from data import (
    download_and_process_all,
    make_single_task_dataloaders,
    make_multitask_train_iterator,
    TASK_NAME_TO_ID,
)
from gradient_signals import (
    compute_per_task_grads,
    compute_gradient_signals,
    combined_gradient_score,
)


# ── W&B helper ────────────────────────────────────────────────────────────────

def _wandb_log(metrics: dict, step: int, use_wandb: bool) -> None:
    if use_wandb and WANDB_AVAILABLE:
        wandb.log(metrics, step=step)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Data helpers ──────────────────────────────────────────────────────────────

def _make_cycling_iterators(loaders: Dict) -> Dict:
    """Return a dict of infinitely-cycling iterators, one per task."""
    return {task: cycle(loader) for task, loader in loaders.items()}


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loaders, loss_fn, device):
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


# ── Gradient signal logging ───────────────────────────────────────────────────

def log_gradient_signals(model, train_loaders, loss_fn, device, n_batches=4):
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


# ── Early-stopping helper ─────────────────────────────────────────────────────

def check_and_freeze(
    per_task_val, best_per_task_loss, patience_counters,
    frozen_tasks, model, args, label,
) -> Tuple[List[str], bool]:
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
                print(
                    f"  [{label}] Froze {task.upper()} head "
                    f"(best val loss: {best_per_task_loss[task]:.4f})"
                )
    return newly_frozen, any_improved


# ── Main training loop ────────────────────────────────────────────────────────

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── W&B init ──────────────────────────────────────────────────────────────
    use_wandb = (not args.no_wandb) and WANDB_AVAILABLE
    if not args.no_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not installed — running without W&B tracking.")
        print("         Install with: pip install wandb")

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config=vars(args),
            tags=["pcgrad", "mtl", "csci567"],
        )
        print(f"W&B run: {wandb.run.url}\n")

    # ── Model + data ──────────────────────────────────────────────────────────
    model = VanillaMTLModel(model_name=args.model_name, dropout=args.dropout)
    model.to(device)

    if use_wandb:
        wandb.watch(model, log="gradients", log_freq=200)

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

    # ── Steps per epoch ────────────────────────────────────────────────────────
    if args.steps_per_epoch > 0:
        steps_per_epoch = args.steps_per_epoch
        print(f"Using fixed steps_per_epoch={steps_per_epoch}")
    else:
        steps_per_epoch = max(len(dl) for dl in train_loaders.values())
        print(f"Using dataset-size-based steps_per_epoch={steps_per_epoch}")

    total_steps = steps_per_epoch * args.num_epochs

    # ── Optimizer + PCGrad + scheduler ───────────────────────────────────────
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    base_optimizer = AdamW(param_groups, lr=args.lr)
    pcgrad         = PCGrad(base_optimizer)
    scheduler      = get_linear_schedule_with_warmup(
        base_optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    loss_fn = nn.CrossEntropyLoss()

    # ── Early-stopping state ──────────────────────────────────────────────────
    best_per_task_loss = {task: float("inf") for task in TASKS}
    patience_counters  = {task: 0 for task in TASKS}
    frozen_tasks: Set[str] = set()
    history     = []
    global_step = 0

    print(
        f"\nStarting PCGrad MTL training for {args.num_epochs} epochs "
        f"({steps_per_epoch} steps/epoch)"
    )
    print(f"Tasks: {TASKS}")
    if args.eval_every > 0:
        print(
            f"Mid-epoch validation every {args.eval_every} steps "
            f"({steps_per_epoch // args.eval_every} checks per epoch)"
        )
    print()

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_start          = time.time()
        running_loss         = {task: 0.0 for task in TASKS}
        running_steps        = {task: 0   for task in TASKS}
        mid_epoch_checks     = []
        epoch_conflict_count = 0
        epoch_steps_taken    = 0

        task_iters = {task: cycle(loader) for task, loader in train_loaders.items()}

        # ── Step loop ─────────────────────────────────────────────────────────
        for step in range(steps_per_epoch):

            active_tasks = [t for t in TASKS if t not in frozen_tasks]
            if not active_tasks:
                break

            # ── 1. Collect one batch per active task ─────────────────────────
            task_losses: List[torch.Tensor] = []
            for task in active_tasks:
                batch          = next(task_iters[task])
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                logits         = model(input_ids, attention_mask, task)
                loss           = loss_fn(logits, labels)
                task_losses.append(loss)

            # ── 2. PCGrad backward ───────────────────────────────────────────
            raw_grads, proj_grads, conflicts = pcgrad.pc_backward(task_losses)

            # ── 3. Clip + step ───────────────────────────────────────────────
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            pcgrad.step()
            scheduler.step()

            # ── 4. Bookkeeping ────────────────────────────────────────────────
            epoch_conflict_count += len(conflicts)
            epoch_steps_taken    += 1
            global_step          += 1

            for i, task in enumerate(active_tasks):
                running_loss[task]  += task_losses[i].item()
                running_steps[task] += 1

            # ── Step logging ──────────────────────────────────────────────────
            if (step + 1) % args.log_every == 0:
                avg_losses         = {t: running_loss[t] / max(running_steps[t], 1)
                                       for t in TASKS}
                conflicts_per_step = epoch_conflict_count / epoch_steps_taken
                frozen_str         = (
                    f" [frozen: {', '.join(sorted(frozen_tasks))}]"
                    if frozen_tasks else ""
                )
                print(
                    f"  Epoch {epoch} | Step {step+1}/{steps_per_epoch}{frozen_str} | "
                    + " | ".join(f"{t}: {avg_losses[t]:.4f}" for t in TASKS)
                    + f" | conflicts/step: {conflicts_per_step:.2f}"
                )

                # W&B: step-level metrics
                step_log = {
                    "train/step":               global_step,
                    "train/conflicts_per_step":  conflicts_per_step,
                    "train/lr":                  scheduler.get_last_lr()[0],
                    "train/n_frozen_tasks":      len(frozen_tasks),
                }
                for t in TASKS:
                    step_log[f"train/loss_{t}"] = avg_losses[t]
                _wandb_log(step_log, step=global_step, use_wandb=use_wandb)

            # ── Mid-epoch validation ──────────────────────────────────────────
            if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
                avg_val_loss, per_task_val = evaluate(
                    model, val_loaders, loss_fn, device
                )

                print(f"\n  [Mid-epoch step {step+1}] Val check:")
                for t, m in per_task_val.items():
                    marker = "❄" if t in frozen_tasks else f"p{patience_counters[t]}"
                    print(
                        f"    {t.upper():6s} -> loss: {m['loss']:.4f}  "
                        f"acc: {m['acc']*100:.2f}%  [{marker}]"
                    )

                newly_frozen, any_improved = check_and_freeze(
                    per_task_val, best_per_task_loss, patience_counters,
                    frozen_tasks, model, args, label=f"step {step+1}",
                )

                if any_improved:
                    torch.save(model.state_dict(), output_dir / "best_model.pt")
                    print("  Model saved.")

                # W&B: mid-epoch val metrics
                val_log = {
                    "val/avg_loss":    avg_val_loss,
                    "val/n_frozen":    len(frozen_tasks),
                }
                for t, m in per_task_val.items():
                    val_log[f"val/loss_{t}"] = m["loss"]
                    val_log[f"val/acc_{t}"]  = m["acc"] * 100
                _wandb_log(val_log, step=global_step, use_wandb=use_wandb)

                mid_epoch_checks.append({
                    "step":                    step + 1,
                    "avg_val_loss":            avg_val_loss,
                    "per_task_val":            per_task_val,
                    "frozen_tasks":            list(frozen_tasks),
                    "patience_counters":       dict(patience_counters),
                    "pcgrad_conflicts_so_far": epoch_conflict_count,
                })

                if frozen_tasks == set(TASKS):
                    break

                model.train()

        # ── End of epoch ──────────────────────────────────────────────────────
        avg_val_loss, per_task_val = evaluate(model, val_loaders, loss_fn, device)

        print("  Computing gradient signals...")
        grad_signals = log_gradient_signals(
            model, train_loaders, loss_fn, device, n_batches=args.grad_signal_batches
        )

        epoch_time         = time.time() - epoch_start
        conflicts_per_step = (
            epoch_conflict_count / epoch_steps_taken if epoch_steps_taken else 0.0
        )
        newly_frozen, any_improved = check_and_freeze(
            per_task_val, best_per_task_loss, patience_counters,
            frozen_tasks, model, args, label=f"epoch {epoch}",
        )

        # ── Epoch summary ──────────────────────────────────────────────────────
        print(
            f"\nEpoch {epoch}/{args.num_epochs} | "
            f"Avg Val Loss: {avg_val_loss:.4f} | Time: {epoch_time:.1f}s"
        )
        for task, m in per_task_val.items():
            if task in frozen_tasks and task not in newly_frozen:
                status = "❄ frozen"
            elif task in newly_frozen:
                status = "❄ just frozen"
            else:
                status = f"patience {patience_counters[task]}/{args.patience}"
            print(
                f"  {task.upper():6s} -> loss: {m['loss']:.4f}  "
                f"acc: {m['acc']*100:.2f}%  [{status}]"
            )
        print(
            f"  Gradient Signals -> "
            f"conflict_rate: {grad_signals['conflict_rate']:.4f}  "
            f"severity: {grad_signals['conflict_severity']:.4f}  "
            f"variance: {grad_signals['gradient_variance']:.6f}  "
            f"norm_ratio: {grad_signals['grad_norm_ratio']:.4f}  "
            f"snr: {grad_signals['grad_snr']:.4f}  "
            f"score: {grad_signals['combined_gradient_score']:.4f}"
        )
        print(
            f"  PCGrad Conflicts -> {epoch_conflict_count} total  "
            f"({conflicts_per_step:.2f} per step)"
        )

        if any_improved:
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print("  Model saved.")

        # W&B: epoch-level metrics
        epoch_log = {
            "epoch":                          epoch,
            "epoch/avg_val_loss":             avg_val_loss,
            "epoch/time_s":                   epoch_time,
            "epoch/n_frozen_tasks":           len(frozen_tasks),
            # gradient signals
            "gradient/conflict_rate":         grad_signals["conflict_rate"],
            "gradient/conflict_severity":     grad_signals["conflict_severity"],
            "gradient/variance":              grad_signals["gradient_variance"],
            "gradient/norm_ratio":            grad_signals["grad_norm_ratio"],
            "gradient/snr":                   grad_signals["grad_snr"],
            "gradient/combined_score":        grad_signals["combined_gradient_score"],
            # pcgrad-specific
            "pcgrad/conflicts_total":         epoch_conflict_count,
            "pcgrad/conflicts_per_step":      conflicts_per_step,
        }
        for t, m in per_task_val.items():
            epoch_log[f"epoch/val_loss_{t}"] = m["loss"]
            epoch_log[f"epoch/val_acc_{t}"]  = m["acc"] * 100
        _wandb_log(epoch_log, step=global_step, use_wandb=use_wandb)

        history.append({
            "epoch":                          epoch,
            "avg_val_loss":                   avg_val_loss,
            "per_task_val":                   per_task_val,
            "best_per_task_loss":             dict(best_per_task_loss),
            "patience_counters":              dict(patience_counters),
            "frozen_tasks":                   list(frozen_tasks),
            "mid_epoch_checks":               mid_epoch_checks,
            "conflict_rate":                  grad_signals["conflict_rate"],
            "conflict_severity":              grad_signals["conflict_severity"],
            "gradient_variance":              grad_signals["gradient_variance"],
            "grad_norm_ratio":                grad_signals["grad_norm_ratio"],
            "grad_snr":                       grad_signals["grad_snr"],
            "combined_gradient_score":        grad_signals["combined_gradient_score"],
            "pcgrad_conflicts_total":         epoch_conflict_count,
            "pcgrad_conflicts_per_step":      round(conflicts_per_step, 4),
            "epoch_time_s":                   round(epoch_time, 1),
        })

        if frozen_tasks == set(TASKS):
            print(f"\nAll tasks frozen. Training complete at epoch {epoch}.")
            break

        print()

    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. History saved to {history_path}")

    if use_wandb:
        wandb.save(str(history_path), base_path=str(output_dir))  # upload JSON to W&B
        wandb.finish()

    return history


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="PCGrad MTL Training (CSCI 567)")
    # Model
    parser.add_argument("--model_name",  default="distilbert-base-uncased")
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--max_length",  type=int,   default=128)
    # Data
    parser.add_argument("--processed_dir",       default="./processed_data",
                        help="Directory for tokenized/cached dataset splits.")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Cap training examples per task. Use 500 for smoke test.")
    parser.add_argument("--max_val_samples",   type=int, default=None,
                        help="Cap validation examples per task. Use 200 for smoke test.")
    parser.add_argument("--num_workers",       type=int, default=2,
                        help="DataLoader workers. Use 0 on Windows to avoid issues.")
    # Training
    parser.add_argument("--batch_size",        type=int,   default=32,
                        help="Batch size per task per step.")
    parser.add_argument("--num_epochs",        type=int,   default=20)
    parser.add_argument("--lr",                type=float, default=2e-5)
    parser.add_argument("--weight_decay",      type=float, default=0.01)
    parser.add_argument("--max_grad_norm",     type=float, default=1.0)
    # Early stopping
    parser.add_argument("--patience",          type=int,   default=5)
    parser.add_argument("--min_delta",         type=float, default=1e-4)
    # Epoch / validation schedule
    parser.add_argument("--steps_per_epoch",   type=int,   default=7000,
                        help="Fixed steps per epoch. Use 50 for smoke test.")
    parser.add_argument("--eval_every",        type=int,   default=1000,
                        help="Validate every N steps. Use 25 for smoke test.")
    parser.add_argument("--grad_signal_batches", type=int, default=4,
                        help="Batches used to estimate gradient signals. Use 2 for smoke test.")
    # Output
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--output_dir", default="./outputs/pcgrad_mtl")
    parser.add_argument("--log_every",  type=int, default=100,
                        help="Print step log every N steps. Use 10 for smoke test.")
    # W&B
    parser.add_argument("--wandb_project", default="csci567-mtl",
                        help="W&B project name.")
    parser.add_argument("--wandb_run",     default=None,
                        help="W&B run name. Defaults to auto-generated.")
    parser.add_argument("--no_wandb",      action="store_true",
                        help="Disable W&B logging entirely.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("PCGrad MTL — Gradient Surgery (CSCI 567)")
    print("=" * 60)
    print(vars(args))
    print()
    train(args)
