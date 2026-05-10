"""
train.py

Multi-task NLP training script for shared-encoder fine-tuning, supporting:
  - Shared DistilBERT encoder + 4 task heads (Yelp, QNLI, QQP, MNLI)
  - Vanilla MTL or PCGrad gradient surgery (toggle via --use_pcgrad)
  - Loss = sum of per-task cross-entropy losses
  - Gradient normalisation by number of active tasks
  - Uniform task sampling
  - Configurable early stopping: --early_stop_mode {per_task, global, both}
  - Mid-epoch validation via --eval_every
  - Six gradient signals logged every eval check
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
from pcgrad import PCGrad
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
def check_global_early_stop(
    avg_val_loss: float,
    global_state: dict,
    args: argparse.Namespace,
    label: str,
) -> bool:
    """
    Global early stopping based on average validation loss across all tasks.
    Mutates `global_state` in place. The dict tracks:
        best_avg_val_loss : float   — best avg val loss seen so far
        patience_counter  : int     — checks since last improvement
        should_stop       : bool    — True once patience is exhausted
    Returns:
        improved (bool) — True if avg_val_loss is a new best.
    """
    improved = False
    if avg_val_loss < global_state["best_avg_val_loss"] - args.min_delta:
        global_state["best_avg_val_loss"] = avg_val_loss
        global_state["patience_counter"]  = 0
        improved                          = True
        print(f"  [{label}] New best avg_val_loss: {avg_val_loss:.4f} "
              f"(global patience reset)")
    else:
        global_state["patience_counter"] += 1
        print(f"  [{label}] No avg-loss improvement: {avg_val_loss:.4f} "
              f"(best: {global_state['best_avg_val_loss']:.4f}, "
              f"global patience: {global_state['patience_counter']}/{args.patience})")
        if global_state["patience_counter"] >= args.patience:
            global_state["should_stop"] = True
            print(f"  [{label}] *** Global patience exhausted. ***")
    return improved
def check_per_task_early_stop(
    per_task_val: dict,
    per_task_state: dict,
    model: nn.Module,
    args: argparse.Namespace,
    label: str,
) -> bool:
    """
    Per-task patience checking with head freezing on plateau.
    Mutates `per_task_state` in place. The dict tracks:
        best_loss          : {task: float}  — best val loss per task
        patience_counter   : {task: int}    — per-task patience counter
        frozen_tasks       : set            — tasks whose heads are frozen
        should_stop        : bool           — True when all tasks frozen
    When a task's val loss has not improved for `--patience` checks,
    that task's head parameters are set requires_grad=False (frozen).
    Returns:
        any_improved (bool) — True if at least one non-frozen task improved.
    """
    any_improved = False
    newly_frozen = []
    for task in TASKS:
        if task in per_task_state["frozen_tasks"]:
            continue
        task_val_loss = per_task_val[task]["loss"]
        if task_val_loss < per_task_state["best_loss"][task] - args.min_delta:
            per_task_state["best_loss"][task]        = task_val_loss
            per_task_state["patience_counter"][task] = 0
            any_improved                             = True
        else:
            per_task_state["patience_counter"][task] += 1
            if per_task_state["patience_counter"][task] >= args.patience:
                for param in model.heads[task].parameters():
                    param.requires_grad = False
                per_task_state["frozen_tasks"].add(task)
                newly_frozen.append(task)
                print(f"  [{label}] Froze {task.upper()} head "
                      f"(best val loss: {per_task_state['best_loss'][task]:.4f})")
    if per_task_state["frozen_tasks"] == set(TASKS):
        per_task_state["should_stop"] = True
        print(f"  [{label}] *** All tasks frozen. ***")
    return any_improved
def check_early_stopping(
    avg_val_loss: float,
    per_task_val: dict,
    global_state: dict,
    per_task_state: dict,
    model: nn.Module,
    args: argparse.Namespace,
    label: str,
) -> tuple[bool, bool]:
    """
    Dispatcher that runs the early-stopping check(s) appropriate for the
    selected `args.early_stop_mode`:
        "global"   — only check global avg-val-loss patience.
        "per_task" — only check per-task patience + freeze heads.
        "both"     — run both. Either condition can trigger stopping.
                     Improvement = either rule sees an improvement
                     (so the checkpoint is saved if either fires).
    Returns:
        (improved, should_stop)
    """
    improved_global   = False
    improved_per_task = False
    if args.early_stop_mode in ("global", "both"):
        improved_global = check_global_early_stop(
            avg_val_loss, global_state, args, label,
        )
    if args.early_stop_mode in ("per_task", "both"):
        improved_per_task = check_per_task_early_stop(
            per_task_val, per_task_state, model, args, label,
        )
    improved    = improved_global or improved_per_task
    should_stop = global_state["should_stop"] or per_task_state["should_stop"]
    return improved, should_stop
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
    # ── PCGrad setup ────────────────────────────────────────────────────────────
    pcgrad = PCGrad(model, encoder_attr="encoder", reduction="mean") if args.use_pcgrad else None
    if pcgrad is not None:
        print(f"PCGrad enabled. Accumulating {args.pcgrad_accum_steps} batches per "
              f"optimizer step (≈ {steps_per_epoch // args.pcgrad_accum_steps} optimizer "
              f"steps per epoch).")
    # Early-stopping state (both kinds initialized; only the active one(s)
    # are actually mutated based on args.early_stop_mode)
    global_state = {
        "best_avg_val_loss": float("inf"),
        "patience_counter":  0,
        "should_stop":       False,
    }
    per_task_state = {
        "best_loss":        {task: float("inf") for task in TASKS},
        "patience_counter": {task: 0 for task in TASKS},
        "frozen_tasks":     set(),
        "should_stop":      False,
    }
    history: list = []
    global_step   = 0
    print(f"Early-stop mode: {args.early_stop_mode} (patience={args.patience})")
    mode_str = "PCGrad" if args.use_pcgrad else "Vanilla"
    print(f"\nStarting {mode_str} MTL training for {args.num_epochs} epochs "
          f"({steps_per_epoch} steps/epoch)")
    print(f"Tasks: {TASKS}")
    if args.eval_every > 0:
        evals_per_epoch = steps_per_epoch // args.eval_every
        print(f"Mid-epoch validation every {args.eval_every} steps "
              f"({evals_per_epoch} checks per epoch)")
    print()
    sampler_iter = iter(sampler)
    # Track PCGrad statistics across each epoch (averaged at logging time)
    pcgrad_stats_acc = {"projections": 0.0, "pairs_checked": 0.0, "n_optim_steps": 0}
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_start      = time.time()
        running_loss     = {task: 0.0 for task in TASKS}
        running_steps    = {task: 0   for task in TASKS}
        mid_epoch_checks = []
        for step in range(steps_per_epoch):
            if pcgrad is None:
                # ── ORIGINAL single-task-per-step path ────────────────────────
                batch = next(sampler_iter)
                task  = batch["task_name"][0]
                # In per_task / both modes, frozen tasks are skipped.
                # In global mode, frozen_tasks stays empty so this is a no-op.
                if task in per_task_state["frozen_tasks"]:
                    continue
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                logits = model(input_ids, attention_mask, task)
                loss   = loss_fn(logits, labels)
                # Scale by 1/n_active so per-step gradient magnitude is comparable
                # across modes. With no freezing, n_active == n_tasks.
                n_active = len(TASKS) - len(per_task_state["frozen_tasks"])
                loss     = loss / max(n_active, 1)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                running_loss[task]  += loss.item() * max(n_active, 1)
                running_steps[task] += 1
                global_step         += 1
            else:
                # ── PCGrad path: accumulate K batches, project, then step ─────
                # Pull `pcgrad_accum_steps` batches from the sampler. Frozen
                # tasks (only populated in per_task / both modes) are skipped.
                # Each task's contribution is captured in PCGrad's internal
                # per-task buffer; multiple batches from the same task in one
                # window are averaged together before projection.
                pcgrad.zero_task_grads()
                tasks_seen_this_step: list = []
                batches_consumed = 0
                while batches_consumed < args.pcgrad_accum_steps:
                    batch = next(sampler_iter)
                    task  = batch["task_name"][0]
                    batches_consumed += 1
                    if task in per_task_state["frozen_tasks"]:
                        continue
                    input_ids      = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels         = batch["labels"].to(device)
                    logits = model(input_ids, attention_mask, task)
                    loss   = loss_fn(logits, labels)
                    # Capture this task's gradient into PCGrad's buffer.
                    # backward_for_task zeroes model grads, calls loss.backward(),
                    # and snapshots the grads — so do NOT call optimizer.zero_grad
                    # or loss.backward() yourself here.
                    pcgrad.backward_for_task(task, loss)
                    tasks_seen_this_step.append(task)
                    running_loss[task]  += loss.item()
                    running_steps[task] += 1
                if not tasks_seen_this_step:
                    # Defensive — shouldn't happen with global early stopping
                    # since no tasks are ever skipped, but keeping the guard.
                    continue
                # Project conflicting encoder grads, write final grads in-place,
                # then clip + step like normal.
                pcgrad_info = pcgrad.apply_gradients()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                global_step += 1
                pcgrad_stats_acc["projections"]   += pcgrad_info["pcgrad_projections"]
                pcgrad_stats_acc["pairs_checked"] += pcgrad_info["pcgrad_pairs_checked"]
                pcgrad_stats_acc["n_optim_steps"] += 1
            # ── Step logging ──────────────────────────────────────────────────
            if (step + 1) % args.log_every == 0:
                avg_losses = {
                    t: running_loss[t] / max(running_steps[t], 1) for t in TASKS
                }
                frozen_str = (
                    f" [frozen: {', '.join(sorted(per_task_state['frozen_tasks']))}]"
                    if per_task_state["frozen_tasks"] else ""
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
                    tl = train_loss_snapshot[t]
                    if t in per_task_state["frozen_tasks"]:
                        marker = "❄"
                    elif args.early_stop_mode in ("per_task", "both"):
                        marker = f"p{per_task_state['patience_counter'][t]}"
                    else:
                        marker = ""
                    marker_str = f"  [{marker}]" if marker else ""
                    print(f"    {t.upper():6s} -> train: {tl:.4f}  "
                          f"val: {m['loss']:.4f}  acc: {m['acc']*100:.2f}%{marker_str}")
                print(f"    AVG    -> val: {avg_val_loss:.4f}")
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
                # PCGrad runtime stats: how often did we actually project?
                if pcgrad is not None and pcgrad_stats_acc["n_optim_steps"] > 0:
                    pc_rate = (pcgrad_stats_acc["projections"]
                               / max(pcgrad_stats_acc["pairs_checked"], 1))
                    print(f"  PCGrad -> projections: {int(pcgrad_stats_acc['projections'])}  "
                          f"pairs_checked: {int(pcgrad_stats_acc['pairs_checked'])}  "
                          f"runtime_conflict_rate: {pc_rate:.4f}  "
                          f"(over last {pcgrad_stats_acc['n_optim_steps']} optim steps)")
                improved, should_stop = check_early_stopping(
                    avg_val_loss, per_task_val,
                    global_state, per_task_state, model, args,
                    label=f"step {step+1}",
                )
                if improved:
                    torch.save(model.state_dict(), output_dir / "best_model.pt")
                    print("  Model saved (new best).")
                # Snapshot + reset PCGrad accumulator so each window is independent
                pcgrad_window_stats = dict(pcgrad_stats_acc) if pcgrad is not None else {}
                if pcgrad is not None:
                    pcgrad_stats_acc = {"projections": 0.0, "pairs_checked": 0.0, "n_optim_steps": 0}
                mid_epoch_checks.append({
                    "step":              step + 1,
                    "global_step":       global_step,
                    "train_loss":        train_loss_snapshot,
                    "avg_val_loss":      avg_val_loss,
                    "per_task_val":      per_task_val,
                    # Both states recorded so logs are mode-agnostic
                    "best_avg_val_loss":     global_state["best_avg_val_loss"],
                    "global_patience":       global_state["patience_counter"],
                    "best_per_task_loss":    dict(per_task_state["best_loss"]),
                    "per_task_patience":     dict(per_task_state["patience_counter"]),
                    "frozen_tasks":          sorted(per_task_state["frozen_tasks"]),
                    "pcgrad_window":     pcgrad_window_stats,
                    **grad_signals,
                })
                if should_stop:
                    break
                model.train()
        # If mid-epoch step loop broke due to early stopping, also exit epoch loop
        if global_state["should_stop"] or per_task_state["should_stop"]:
            break
        # ── End of epoch ───────────────────────────────────────────────────────
        avg_val_loss, per_task_val = evaluate(model, val_loaders, loss_fn, device)
        print("  Computing gradient signals...")
        grad_signals = log_gradient_signals(
            model, train_loaders, loss_fn, device,
            n_batches=args.grad_signal_batches,
        )
        epoch_time = time.time() - epoch_start
        # ── Epoch summary ──────────────────────────────────────────────────────
        print(f"\nEpoch {epoch}/{args.num_epochs} | "
              f"Avg Val Loss: {avg_val_loss:.4f} | Time: {epoch_time:.1f}s")
        for task, m in per_task_val.items():
            print(f"  {task.upper():6s} -> loss: {m['loss']:.4f}  "
                  f"acc: {m['acc']*100:.2f}%")
        print(f"  Gradient Signals -> "
              f"conflict_rate: {grad_signals['conflict_rate']:.4f}  "
              f"severity: {grad_signals['conflict_severity']:.4f}  "
              f"variance: {grad_signals['gradient_variance']:.6f}  "
              f"norm_ratio: {grad_signals['grad_norm_ratio']:.4f}  "
              f"snr: {grad_signals['grad_snr']:.4f}  "
              f"score: {grad_signals['combined_gradient_score']:.4f}")
        if pcgrad is not None and pcgrad_stats_acc["n_optim_steps"] > 0:
            pc_rate = (pcgrad_stats_acc["projections"]
                       / max(pcgrad_stats_acc["pairs_checked"], 1))
            print(f"  PCGrad -> projections: {int(pcgrad_stats_acc['projections'])}  "
                  f"pairs_checked: {int(pcgrad_stats_acc['pairs_checked'])}  "
                  f"runtime_conflict_rate: {pc_rate:.4f}  "
                  f"(over last {pcgrad_stats_acc['n_optim_steps']} optim steps)")
        # Snapshot PCGrad stats for the epoch record, then reset
        pcgrad_epoch_stats = dict(pcgrad_stats_acc) if pcgrad is not None else {}
        if pcgrad is not None:
            pcgrad_stats_acc = {"projections": 0.0, "pairs_checked": 0.0, "n_optim_steps": 0}
        # End-of-epoch early-stopping check (mode-agnostic dispatcher)
        improved, should_stop = check_early_stopping(
            avg_val_loss, per_task_val,
            global_state, per_task_state, model, args,
            label=f"epoch {epoch}",
        )
        if improved:
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print("  Model saved (new best).")
        avg_train_loss = {
            t: round(running_loss[t] / max(running_steps[t], 1), 6)
            for t in TASKS
        }
        history.append({
            "epoch":             epoch,
            "avg_train_loss":    avg_train_loss,
            "avg_val_loss":      avg_val_loss,
            "per_task_val":      per_task_val,
            # Both states recorded so logs are mode-agnostic
            "best_avg_val_loss":     global_state["best_avg_val_loss"],
            "global_patience":       global_state["patience_counter"],
            "best_per_task_loss":    dict(per_task_state["best_loss"]),
            "per_task_patience":     dict(per_task_state["patience_counter"]),
            "frozen_tasks":          sorted(per_task_state["frozen_tasks"]),
            "mid_epoch_checks":  mid_epoch_checks,
            "epoch_time_s":      round(epoch_time, 1),
            "pcgrad_epoch":      pcgrad_epoch_stats if pcgrad is not None else {},
            **grad_signals,
        })
        if should_stop:
            print(f"\nEarly stopping triggered at epoch {epoch} "
                  f"(mode={args.early_stop_mode}).")
            break
        print()
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. History saved to {history_path}")
    # ── Final test evaluation on held-out test set ─────────────────────────
    if args.final_eval_on_test:
        print("\n" + "=" * 60)
        print("FINAL TEST EVALUATION (held-out test set)")
        print("=" * 60)
        # Load best checkpoint
        best_ckpt = output_dir / "best_model.pt"
        if not best_ckpt.exists():
            print(f"  WARNING: best_model.pt not found at {best_ckpt}; "
                  f"using current model state.")
        else:
            print(f"  Loading best checkpoint from {best_ckpt}")
            state = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(state)
            # Re-enable gradients on any frozen heads so eval works (eval is
            # under no_grad anyway, but keeps state consistent)
            for p in model.parameters():
                p.requires_grad = True
        # Build test loaders. Some loaders dicts may not have "test" if the
        # processed_data is from an older pipeline version.
        test_loaders = {}
        for task in TASKS:
            if "test" in all_loaders[task]:
                test_loaders[task] = all_loaders[task]["test"]
            else:
                print(f"  WARNING: no test split for {task} (older processed_data?). "
                      f"Skipping.")
        if not test_loaders:
            print("  No test splits available — skipping final test evaluation.")
        else:
            avg_test_loss, per_task_test = evaluate(
                model, test_loaders, loss_fn, device,
            )
            print(f"\n  Test results (avg loss: {avg_test_loss:.4f}):")
            for task, m in per_task_test.items():
                print(f"    {task.upper():6s} -> loss: {m['loss']:.4f}  "
                      f"acc: {m['acc']*100:.2f}%")
            # Save to JSON for later analysis
            test_results = {
                "avg_test_loss": avg_test_loss,
                "per_task_test": per_task_test,
                "checkpoint":    str(best_ckpt) if best_ckpt.exists() else "current_state",
                "early_stop_mode": args.early_stop_mode,
                "use_pcgrad":      args.use_pcgrad,
            }
            test_path = output_dir / "final_test_results.json"
            with open(test_path, "w") as f:
                json.dump(test_results, f, indent=2)
            print(f"\n  Test results saved to {test_path}")
    return history
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-task NLP training (vanilla or PCGrad) with configurable "
                    "global / per-task early stopping and gradient-signal logging."
    )
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
    parser.add_argument("--patience",            type=int,   default=3,
                        help="Early-stopping patience: number of consecutive "
                             "validation checks without improvement before stopping. "
                             "Counts every eval (mid-epoch + end-of-epoch). "
                             "Interpretation depends on --early_stop_mode.")
    parser.add_argument("--early_stop_mode",
                        type=str, default="per_task",
                        choices=["per_task", "global", "both"],
                        help="Early-stopping strategy. "
                             "'per_task': each task's head is frozen after its own "
                             "val loss plateaus for `patience` checks; training stops "
                             "when all heads are frozen. "
                             "'global': training stops when avg val loss across all "
                             "tasks plateaus for `patience` checks. No head freezing. "
                             "'both': run both checks; stop on whichever fires first.")
    parser.add_argument("--final_eval_on_test", action="store_true",
                        help="After training, load best_model.pt and evaluate on the "
                             "held-out test set (carved from train, see TEST_SIZES in "
                             "data.py). Results are saved to final_test_results.json. "
                             "Requires processed_data with test splits — re-run "
                             "preprocessing with the new data.py if you have an old cache.")
    parser.add_argument("--min_delta",           type=float, default=1e-4)
    parser.add_argument("--grad_signal_batches", type=int,   default=4)
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument("--output_dir",          default="./outputs/mtl")
    parser.add_argument("--log_every",           type=int,   default=100)
    parser.add_argument("--eval_every",          type=int,   default=1000)
    parser.add_argument("--steps_per_epoch",     type=int,   default=3000)
    # ── PCGrad options ────────────────────────────────────────────────────────
    parser.add_argument("--use_pcgrad",          action="store_true",
                        help="Enable PCGrad gradient-conflict surgery on the shared encoder. "
                             "Without this flag, training uses vanilla MTL.")
    parser.add_argument("--pcgrad_accum_steps",  type=int,   default=4,
                        help="Number of batches accumulated from the multitask sampler "
                             "per PCGrad-merged optimizer step. Default 4 ≈ one batch "
                             "per task in expectation.")
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    mode = "PCGrad MTL" if args.use_pcgrad else "Vanilla MTL"
    print("=" * 60)
    print(f"{mode} (early_stop_mode={args.early_stop_mode})")
    print("=" * 60)
    print(vars(args))
    print()
    train(args)
