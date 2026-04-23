"""
gradient_signals.py

Computes inter-task gradient statistics on the shared encoder parameters.

Metrics
-------
Existing:
  conflict_rate       fraction of (task-pair, param) combos with negative cosine sim
  conflict_severity   mean |cos_sim| among conflicting pairs
  gradient_variance   mean per-param variance of gradient values across tasks

New:
  grad_norm_ratio     max / min per-task gradient L2 norm; high → one task dominates
  grad_snr            mean(|grad|) / std(grad) per param, averaged across params & tasks;
                      low → noisy/inconsistent gradients, model in flat/chaotic region

All computations happen on CPU to avoid holding extra GPU tensors.

Identical to vanilla/gradient_signals.py — kept as a local copy so the pcgrad/
folder is fully self-contained and runnable without depending on any sibling directory.
"""

from __future__ import annotations
from typing import Dict, Tuple

import torch
import torch.nn as nn
from itertools import combinations


# ── Internal helpers ───────────────────────────────────────────────────────────

def _collect_shared_grads(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Collects flattened gradients from the shared encoder parameters.
    Returns {param_name: grad_flat} for params that have a gradient.
    Clones immediately so tensors are independent of the model's grad buffers.
    """
    grads = {}
    for name, param in model.encoder.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().cpu().float().flatten().clone()
    return grads


# ── Public API ─────────────────────────────────────────────────────────────────

def compute_per_task_grads(
    model: nn.Module,
    batch: dict,
    task: str,
    loss_fn: nn.CrossEntropyLoss,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Forward+backward pass for a single task batch WITHOUT updating weights.
    Returns cloned CPU grad tensors for shared encoder params — safe to
    accumulate across tasks while later backward passes overwrite model buffers.
    """
    model.zero_grad()
    input_ids      = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels         = batch["labels"].to(device)

    with torch.enable_grad():
        logits = model(input_ids, attention_mask, task)
        loss   = loss_fn(logits, labels)
        loss.backward()

    return _collect_shared_grads(model)


def compute_gradient_signals(
    task_grads: Dict[str, Dict[str, torch.Tensor]]
) -> Dict[str, float]:
    """
    Given per-task gradient dicts (each independently cloned), computes all
    gradient signals and returns them as a flat dict.

    Args:
        task_grads: {task_name: {param_name: grad_flat_tensor}}
                    tensors must be independent clones (not views of model grads).

    Returns dict with keys:
        conflict_rate       ∈ [0, 1]
        conflict_severity   ∈ [0, 1]   (0 if no conflicts)
        gradient_variance   ∈ [0, ∞)
        grad_norm_ratio     ∈ [1, ∞)   max/min per-task encoder grad norm
        grad_snr            ∈ [0, ∞)   mean signal-to-noise ratio across params/tasks
    """
    tasks = list(task_grads.keys())
    zero = {
        "conflict_rate":     0.0,
        "conflict_severity": 0.0,
        "gradient_variance": 0.0,
        "grad_norm_ratio":   1.0,
        "grad_snr":          0.0,
    }
    if len(tasks) < 2:
        return zero

    task_pairs = list(combinations(tasks, 2))

    # Intersect param names so we only compare params present in every task
    param_names = set(task_grads[tasks[0]].keys())
    for t in tasks[1:]:
        param_names &= set(task_grads[t].keys())

    if not param_names:
        return zero

    # ── Per-task encoder gradient L2 norms (for norm ratio) ───────────────────
    # Concatenate all shared-param grads per task into one vector, then take norm
    task_norms = {}
    for t in tasks:
        all_grads = torch.cat([task_grads[t][p] for p in param_names])
        task_norms[t] = all_grads.norm().item()

    norms = list(task_norms.values())
    max_norm = max(norms)
    min_norm = min(norms)
    grad_norm_ratio = max_norm / min_norm if min_norm > 1e-12 else float("inf")

    # ── Per-param loop: variance, conflict, SNR ────────────────────────────────
    total_pairs    = 0
    conflict_count = 0
    severity_sum   = 0.0
    variance_sum   = 0.0
    snr_sum        = 0.0
    snr_count      = 0
    n_params       = 0

    for pname in param_names:
        grads_for_param = [task_grads[t][pname] for t in tasks]
        stacked = torch.stack(grads_for_param, dim=0)   # (n_tasks, D)

        # Gradient variance across tasks
        variance_sum += stacked.var(dim=0, unbiased=True).mean().item()
        n_params += 1

        # SNR per task for this param: mean(|g|) / (std(g) + eps)
        # Averaged over all tasks — low SNR means noisy, inconsistent gradients
        for g in grads_for_param:
            mean_abs = g.abs().mean().item()
            std      = g.std().item()
            if std > 1e-12:
                snr_sum   += mean_abs / std
                snr_count += 1

        # Pairwise cosine similarities
        for t1, t2 in task_pairs:
            g1 = task_grads[t1][pname]
            g2 = task_grads[t2][pname]

            norm1 = g1.norm()
            norm2 = g2.norm()
            if norm1 < 1e-12 or norm2 < 1e-12:
                continue

            cos_sim_val = ((g1 @ g2) / (norm1 * norm2)).item()
            total_pairs += 1

            if cos_sim_val < 0:
                conflict_count += 1
                severity_sum   += abs(cos_sim_val)

    conflict_rate     = conflict_count / total_pairs    if total_pairs    > 0 else 0.0
    conflict_severity = severity_sum   / conflict_count if conflict_count > 0 else 0.0
    gradient_variance = variance_sum   / n_params       if n_params       > 0 else 0.0
    grad_snr          = snr_sum        / snr_count      if snr_count      > 0 else 0.0

    return {
        "conflict_rate":     conflict_rate,
        "conflict_severity": conflict_severity,
        "gradient_variance": gradient_variance,
        "grad_norm_ratio":   grad_norm_ratio,
        "grad_snr":          grad_snr,
    }


def combined_gradient_score(
    signals: Dict[str, float],
    w_rate:     float = 0.3,
    w_severity: float = 0.3,
    w_variance: float = 0.15,
    w_norm:     float = 0.15,
    w_snr:      float = 0.1,
) -> float:
    """
    Combines all gradient metrics into a single scalar score.
    Higher score → more conflict / instability → potential early-stop signal.

    gradient_variance and grad_norm_ratio are soft-normalised with tanh so
    all components live in [0, 1] before weighting.
    SNR is inverted (1 - tanh) so that low SNR → high instability score.
    """
    norm_variance = float(torch.tanh(torch.tensor(signals["gradient_variance"])).item())
    # norm_ratio is in [1, ∞); shift by -1 so clean training (ratio=1) → 0
    norm_ratio    = float(torch.tanh(torch.tensor(signals["grad_norm_ratio"] - 1.0)).item())
    # invert SNR: high SNR = stable = low score
    inv_snr       = 1.0 - float(torch.tanh(torch.tensor(signals["grad_snr"])).item())

    score = (
        w_rate     * signals["conflict_rate"]
        + w_severity * signals["conflict_severity"]
        + w_variance * norm_variance
        + w_norm     * norm_ratio
        + w_snr      * inv_snr
    )
    return float(score)
