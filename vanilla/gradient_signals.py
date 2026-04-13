"""
gradient_signals.py

Computes inter-task gradient statistics on the shared encoder parameters:
  - Conflict Rate:    fraction of (task-pair, param) combos with negative cosine similarity
  - Conflict Severity: mean |cos_sim| among conflicting pairs
  - Gradient Variance: mean variance of gradient values across tasks per parameter

These are logged during training and will later be used as early-stopping signals.
All computations happen on CPU to avoid holding extra GPU tensors.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from itertools import combinations


def _get_shared_grads(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Collects flattened gradients from the shared encoder parameters.
    Returns {param_name: grad_flat} for params that have a gradient.
    """
    grads = {}
    for name, param in model.encoder.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().cpu().float().flatten()
    return grads


def compute_per_task_grads(
    model: nn.Module,
    batch: dict,
    task: str,
    loss_fn: nn.CrossEntropyLoss,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Does a forward+backward pass for a single task batch WITHOUT updating weights.
    Returns the shared encoder gradients for this task.
    """
    model.zero_grad()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids, attention_mask, task)
    loss = loss_fn(logits, labels)
    loss.backward()

    return _get_shared_grads(model)


def compute_gradient_signals(
    task_grads: Dict[str, Dict[str, torch.Tensor]]
) -> Tuple[float, float, float]:
    """
    Given per-task gradient dicts, computes:
      conflict_rate     ∈ [0, 1]
      conflict_severity ∈ [0, 1]  (0 if no conflicts)
      gradient_variance ∈ [0, ∞)

    Args:
        task_grads: {task_name: {param_name: grad_flat_tensor}}

    Returns:
        (conflict_rate, conflict_severity, gradient_variance)
    """
    tasks = list(task_grads.keys())
    if len(tasks) < 2:
        return 0.0, 0.0, 0.0

    task_pairs = list(combinations(tasks, 2))

    # Get common param names across all tasks
    param_names = set(task_grads[tasks[0]].keys())
    for t in tasks[1:]:
        param_names &= set(task_grads[t].keys())

    if not param_names:
        return 0.0, 0.0, 0.0

    total_pairs = 0
    conflict_count = 0
    severity_sum = 0.0

    # Per-param gradient tensors stacked for variance computation
    variance_sum = 0.0
    n_params = 0

    for pname in param_names:
        grads_for_param = [task_grads[t][pname] for t in tasks]

        # Gradient variance across tasks for this param
        stacked = torch.stack(grads_for_param, dim=0)  # (n_tasks, D)
        variance_sum += stacked.var(dim=0).mean().item()
        n_params += 1

        # Pairwise cosine similarities
        for t1, t2 in task_pairs:
            g1 = task_grads[t1][pname]
            g2 = task_grads[t2][pname]

            norm1 = g1.norm()
            norm2 = g2.norm()
            if norm1 < 1e-12 or norm2 < 1e-12:
                continue

            cos_sim = (g1 @ g2) / (norm1 * norm2)
            cos_sim = cos_sim.item()
            total_pairs += 1

            if cos_sim < 0:
                conflict_count += 1
                severity_sum += abs(cos_sim)

    conflict_rate = conflict_count / total_pairs if total_pairs > 0 else 0.0
    conflict_severity = severity_sum / conflict_count if conflict_count > 0 else 0.0
    gradient_variance = variance_sum / n_params if n_params > 0 else 0.0

    return conflict_rate, conflict_severity, gradient_variance


def combined_gradient_score(
    conflict_rate: float,
    conflict_severity: float,
    gradient_variance: float,
    w_rate: float = 0.4,
    w_severity: float = 0.4,
    w_variance: float = 0.2,
) -> float:
    """
    Combines the three gradient metrics into a single scalar score.
    Higher score → more conflict / instability → potential signal to stop.

    Gradient variance is normalised with a sigmoid-like rescaling
    before weighting so all three components live in [0,1].
    """
    # Soft-normalise variance: tanh maps [0,∞) → [0,1)
    norm_variance = float(torch.tanh(torch.tensor(gradient_variance)).item())
    score = (
        w_rate * conflict_rate
        + w_severity * conflict_severity
        + w_variance * norm_variance
    )
    return score
