import random
from typing import List, Tuple

import torch
from torch.optim import Optimizer


class PCGrad:
    """Optimizer wrapper that applies PCGrad (Gradient Surgery) before each update.

    PCGrad resolves gradient conflicts between tasks by projecting each task's
    gradient onto the normal plane of any other task's gradient whenever the two
    are in conflict (i.e. their dot product is negative). This prevents one task
    from interfering with the learning progress of another.

    Usage::

        base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer = PCGrad(base_optimizer)

        optimizer.zero_grad()
        raw_grads, proj_grads, conflicts = optimizer.pc_backward([loss_task1, loss_task2])
        optimizer.step()

    Args:
        optimizer (Optimizer): Any standard PyTorch optimizer (Adam, AdamW, SGD, …).
    """

    def __init__(self, optimizer: Optimizer) -> None:
        self._optimizer = optimizer

    def zero_grad(self) -> None:
        """Zero out all parameter gradients via the inner optimizer."""
        self._optimizer.zero_grad()

    def step(self) -> None:
        """Perform a parameter update via the inner optimizer."""
        self._optimizer.step()

    def pc_backward(
        self,
        task_losses: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Tuple[int, int]]]:
        """Compute PCGrad-projected gradients and write them into .grad attributes.

        For each task i, this method:
          1. Computes the raw gradient g_i via autograd.
          2. Iterates over every other task j in a *random* order.
          3. If dot(g_i, g_j) < 0 (conflict), projects out the component of g_i
             that points in the direction of g_j:

                 g_i ← g_i − [dot(g_i, g_j) / (‖g_j‖² + ε)] · g_j

          4. Averages the projected gradients across all tasks.
          5. Writes the averaged gradient back into each parameter's .grad field.

        Args:
            task_losses: A list of scalar loss tensors, one per task.
                         Each must be graph-connected to the shared parameters.

        Returns:
            raw_grads  : List of flat gradient tensors before projection, one per task.
            proj_grads : List of flat gradient tensors after projection, one per task.
            conflicts  : List of (i, j) index pairs where a conflict was detected
                         and the projection was applied.
        """
        num_tasks = len(task_losses)

        # ── Step 1: compute per-task gradients ────────────────────────────────
        raw_grads: List[torch.Tensor] = []
        for idx, loss in enumerate(task_losses):
            self.zero_grad()
            retain = idx < num_tasks - 1
            loss.backward(retain_graph=retain)
            raw_grads.append(self._get_flat_grads())

        # ── Step 2: project gradients to resolve conflicts ────────────────────
        proj_grads: List[torch.Tensor] = []
        conflicts: List[Tuple[int, int]] = []

        for i in range(num_tasks):
            g_i = raw_grads[i].clone()

            other_indices = list(range(num_tasks))
            other_indices.remove(i)
            random.shuffle(other_indices)

            for j in other_indices:
                g_j = raw_grads[j]

                dot_ij = torch.dot(g_i, g_j)

                if dot_ij < 0:
                    # Conflict detected: project g_i to remove the component
                    # along g_j.  The formula is:
                    #
                    #   g_i ← g_i − (dot(g_i, g_j) / ‖g_j‖²) · g_j
                    #
                    # The small ε in the denominator guards against zero-norm g_j.
                    norm_sq_j = torch.dot(g_j, g_j) + 1e-8
                    g_i = g_i - (dot_ij / norm_sq_j) * g_j
                    conflicts.append((i, j))

            proj_grads.append(g_i)

        # ── Step 3: average projected gradients and write back ────────────────
        avg_grad = torch.stack(proj_grads).mean(dim=0)
        self._set_flat_grads(avg_grad)

        return raw_grads, proj_grads, conflicts

    def _get_flat_grads(self) -> torch.Tensor:
        """Collect and flatten all parameter gradients into a single 1-D tensor.

        Only parameters with ``requires_grad=True`` are included.  Parameters
        whose ``.grad`` is ``None`` contribute a zero vector of matching shape.

        Returns:
            A 1-D tensor containing the concatenated gradients of all trainable
            parameters across all parameter groups.
        """
        fragments: List[torch.Tensor] = []
        for group in self._optimizer.param_groups:
            for param in group["params"]:
                if not param.requires_grad:
                    continue
                if param.grad is None:
                    # No gradient computed for this parameter — use zeros.
                    fragments.append(torch.zeros_like(param).view(-1))
                else:
                    fragments.append(param.grad.view(-1))

        return torch.cat(fragments)

    def _set_flat_grads(self, flat_grad: torch.Tensor) -> None:
        """Write a flat gradient vector back into each parameter's ``.grad`` field.

        The flat vector is split according to each parameter's number of elements
        and reshaped to match the original parameter shape.

        Args:
            flat_grad: A 1-D tensor whose total length equals the total number of
                       trainable parameters across all parameter groups.
        """
        offset = 0
        for group in self._optimizer.param_groups:
            for param in group["params"]:
                if not param.requires_grad:
                    continue
                numel = param.numel()
                # Slice out the portion belonging to this parameter and reshape.
                param.grad = flat_grad[offset : offset + numel].view(param.shape).clone()
                offset += numel
