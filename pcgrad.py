"""
pcgrad.py

PCGrad (Projecting Conflicting Gradients) — Yu et al., NeurIPS 2020.
https://arxiv.org/abs/2001.06782

Implements gradient surgery for multi-task learning:
  For each pair of task gradients (g_i, g_j), if their cosine similarity is
  negative (i.e. they conflict), project g_i onto the normal plane of g_j —
  removing the component of g_i that fights g_j. Repeat for every other task,
  for every task, in a random order per step (to keep the projection unbiased).

This implementation projects ONLY the shared-encoder gradients. Task-specific
heads receive gradient from exactly one task each, so they have nothing to
conflict with — their grads pass through unmodified.

Usage in a training loop:

    pcgrad = PCGrad(model, encoder_attr="encoder")

    for step in range(total_steps):
        pcgrad.zero_task_grads()

        # Run a forward+backward pass per task, accumulating into task buffers
        for task in active_tasks_this_step:
            for batch in batches_for_this_task:   # can be 1 or more
                loss = compute_loss(model, batch, task)
                pcgrad.backward_for_task(task, loss)

        # Project conflicting encoder grads, write final grads to model.parameters().grad,
        # then take an ordinary optimizer step.
        info = pcgrad.apply_gradients()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
"""

from __future__ import annotations
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class PCGrad:
    """
    PCGrad helper that maintains per-task gradient buffers for the shared
    encoder, plus a single shared buffer for task-specific heads (which never
    conflict).

    Encoder param grads are stored per-task as flat CPU tensors (cheap to keep
    around for 4 tasks; ~30M params * 4 bytes * 4 tasks ≈ 480 MB on CPU for
    DistilBERT-base). Head param grads are stored as a single set of CPU
    tensors (one per param).

    After all task backward passes are done, call apply_gradients() to:
      1. PCGrad-project the per-task encoder grads against each other.
      2. Sum (or mean) the projected grads.
      3. Write the result into model.encoder parameters' .grad attribute.
      4. Write the accumulated head grads into the head parameters' .grad.

    The model is then ready for an ordinary optimizer.step().
    """

    def __init__(
        self,
        model: nn.Module,
        encoder_attr: str = "encoder",
        reduction: str = "mean",
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model:        the multi-task model. Must expose `model.<encoder_attr>`
                          and have task-specific heads as separate submodules.
            encoder_attr: attribute name of the shared encoder on the model.
            reduction:    'mean' or 'sum'. How to combine projected per-task
                          encoder grads. 'mean' is recommended (matches the
                          1/n_active scaling in your original loop).
            device:       device to do the projection math on. Defaults to the
                          device of the model's first parameter.
        """
        assert reduction in ("mean", "sum"), f"reduction must be 'mean' or 'sum', got {reduction}"
        self.model = model
        self.encoder = getattr(model, encoder_attr)
        self.reduction = reduction
        self.device = device or next(model.parameters()).device

        # Capture the encoder parameter list once. Order matters — we use it for
        # flat ↔ per-param packing/unpacking.
        self._encoder_params: List[Tuple[str, nn.Parameter]] = [
            (n, p) for n, p in self.encoder.named_parameters() if p.requires_grad
        ]
        self._encoder_shapes = [p.shape for _, p in self._encoder_params]
        self._encoder_numels = [p.numel() for _, p in self._encoder_params]

        # Identify head params: every param in the model that is NOT in the
        # encoder. We accumulate their grads directly (no projection needed).
        encoder_param_ids = {id(p) for _, p in self._encoder_params}
        self._head_params: List[Tuple[str, nn.Parameter]] = [
            (n, p) for n, p in model.named_parameters()
            if p.requires_grad and id(p) not in encoder_param_ids
        ]

        # Storage:
        #   _task_enc_grads[task] is a flat tensor on self.device of length
        #   sum(encoder_numels), or None if the task has not contributed yet.
        self._task_enc_grads: Dict[str, torch.Tensor] = {}
        #   _head_grads is a list aligned with self._head_params, accumulated
        #   across all tasks (no projection). None until the first task adds.
        self._head_grads: Optional[List[torch.Tensor]] = None
        #   _task_counts[task] = how many backward passes this task contributed.
        self._task_counts: Dict[str, int] = {}

    # ── Buffer management ──────────────────────────────────────────────────────

    def zero_task_grads(self) -> None:
        """Reset all per-task and head accumulators. Call once per optimizer step."""
        self._task_enc_grads = {}
        self._head_grads = None
        self._task_counts = {}

    def _flatten_encoder_grads(self) -> torch.Tensor:
        """Concatenate current encoder .grad tensors into one flat vector."""
        flats = []
        for _, p in self._encoder_params:
            if p.grad is None:
                flats.append(torch.zeros(p.numel(), device=self.device))
            else:
                flats.append(p.grad.detach().to(self.device).flatten())
        return torch.cat(flats)

    def _unflatten_to_encoder_grads(self, flat: torch.Tensor) -> None:
        """Write a flat vector back into encoder params' .grad attributes."""
        offset = 0
        for (_, p), shape, numel in zip(
            self._encoder_params, self._encoder_shapes, self._encoder_numels
        ):
            chunk = flat[offset : offset + numel].view(shape).to(p.device)
            if p.grad is None:
                p.grad = chunk.clone()
            else:
                p.grad.copy_(chunk)
            offset += numel

    # ── Per-task gradient capture ──────────────────────────────────────────────

    def backward_for_task(self, task: str, loss: torch.Tensor) -> None:
        """
        Zero model grads, run loss.backward(), and snapshot the gradients into
        per-task / head buffers. Safe to call multiple times for the same task
        — the contributions are averaged when apply_gradients runs.

        Heads from other tasks (which receive no gradient from this task's loss
        because the forward pass only routed through `task`'s head) get None
        grads and are simply skipped here.
        """
        self.model.zero_grad(set_to_none=True)
        loss.backward()

        # Encoder: flatten + accumulate into the per-task buffer.
        enc_flat = self._flatten_encoder_grads()
        if task in self._task_enc_grads:
            self._task_enc_grads[task] += enc_flat
        else:
            self._task_enc_grads[task] = enc_flat
        self._task_counts[task] = self._task_counts.get(task, 0) + 1

        # Heads: accumulate raw grads (no projection — they're task-disjoint).
        if self._head_grads is None:
            self._head_grads = [
                p.grad.detach().clone() if p.grad is not None
                else torch.zeros_like(p)
                for _, p in self._head_params
            ]
        else:
            for i, (_, p) in enumerate(self._head_params):
                if p.grad is not None:
                    self._head_grads[i] += p.grad.detach()

    # ── PCGrad projection ──────────────────────────────────────────────────────

    @staticmethod
    def _pcgrad_project(grads: List[torch.Tensor]) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        """
        Apply PCGrad surgery in-place on a copy of `grads`.

        For each task gradient g_i, iterate over the OTHER task gradients in a
        random order. If <g_i, g_j> < 0, replace g_i with
            g_i  -  (<g_i, g_j> / ||g_j||^2) * g_j.

        Yu et al. note: the random order matters for unbiasedness; the
        projection of g_i against g_j uses the ORIGINAL g_j (not its already-
        projected version), which is why we keep an immutable `originals` copy.

        Returns (projected_grads, info) where info tracks how many projections
        actually happened (useful for logging).
        """
        n = len(grads)
        projected = [g.clone() for g in grads]
        originals = [g.clone() for g in grads]   # used as the "other" reference

        n_conflicts = 0
        n_pairs_checked = 0

        for i in range(n):
            others = list(range(n))
            others.remove(i)
            random.shuffle(others)
            for j in others:
                gj = originals[j]
                gj_sq = gj.dot(gj)
                if gj_sq.item() < 1e-24:
                    continue   # zero-grad task — skip
                dot = projected[i].dot(gj)
                n_pairs_checked += 1
                if dot.item() < 0:
                    projected[i] = projected[i] - (dot / gj_sq) * gj
                    n_conflicts += 1

        info = {
            "pcgrad_pairs_checked": float(n_pairs_checked),
            "pcgrad_projections":   float(n_conflicts),
            "pcgrad_conflict_rate": (n_conflicts / n_pairs_checked) if n_pairs_checked > 0 else 0.0,
        }
        return projected, info

    # ── Final gradient assembly ────────────────────────────────────────────────

    def apply_gradients(self) -> Dict[str, float]:
        """
        Combine all per-task encoder gradients via PCGrad projection, then
        write the final gradient vector into model parameters. Heads are
        written directly from the head accumulator.

        Returns an info dict with PCGrad statistics:
            n_tasks_this_step        — how many distinct tasks contributed
            pcgrad_pairs_checked     — total ordered (i,j) pairs inspected
            pcgrad_projections       — how many of those triggered a projection
            pcgrad_conflict_rate     — projections / pairs_checked
        """
        # Encoder side
        tasks = list(self._task_enc_grads.keys())
        info = {
            "n_tasks_this_step":     float(len(tasks)),
            "pcgrad_pairs_checked":  0.0,
            "pcgrad_projections":    0.0,
            "pcgrad_conflict_rate":  0.0,
        }

        if len(tasks) == 0:
            # Nothing to do — leave model.grad as-is (likely None from zero_grad)
            return info

        # Average each task's encoder grad over how many batches it saw
        per_task = [
            self._task_enc_grads[t] / max(self._task_counts[t], 1)
            for t in tasks
        ]

        if len(tasks) == 1:
            # Only one task contributed — no conflicts possible. Just write it.
            final = per_task[0]
        else:
            projected, proj_info = self._pcgrad_project(per_task)
            info.update(proj_info)
            stacked = torch.stack(projected, dim=0)   # (n_tasks, D)
            if self.reduction == "mean":
                final = stacked.mean(dim=0)
            else:
                final = stacked.sum(dim=0)

        self._unflatten_to_encoder_grads(final)

        # Head side: average across the total number of backward passes we
        # accumulated. This matches the 'mean' semantics on the encoder.
        total_backward_passes = sum(self._task_counts.values())
        if self._head_grads is not None and total_backward_passes > 0:
            divisor = total_backward_passes if self.reduction == "mean" else 1
            for (_, p), g in zip(self._head_params, self._head_grads):
                avg = (g / divisor).to(p.device)
                if p.grad is None:
                    p.grad = avg.clone()
                else:
                    p.grad.copy_(avg)

        return info