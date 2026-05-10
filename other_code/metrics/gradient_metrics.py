import itertools
from typing import Dict, List

import torch
import torch.nn as nn


def compute_gradient_metrics(
    task_grads: List[torch.Tensor],
) -> Dict[str, object]:
    """Compute pairwise and aggregate gradient-conflict metrics across tasks.

    For every ordered pair (i, j) produced by ``itertools.combinations``, the
    function computes the cosine similarity between the two gradient vectors and
    records whether a conflict (negative cosine similarity) exists.

    A combined **Gradient Conflict Score (GCS)** is derived as a weighted sum:

        GCS = 0.4 * conflict_rate
            + 0.3 * (conflict_severity / (mean_norm + 1e-8))
            + 0.2 * magnitude_disparity
            + 0.1 * cosine_variance

    Args:
        task_grads: A list of 1-D ``torch.Tensor`` objects, one per task.
                    Each tensor is a flattened gradient vector over shared
                    parameters.

    Returns:
        A ``dict`` with the following keys:

        - ``conflict_rate`` (*float*): Fraction of task pairs in conflict.
        - ``mean_cosine_sim`` (*float*): Mean pairwise cosine similarity.
        - ``conflict_severity`` (*float*): Mean magnitude of negative dot
          products across conflicting pairs.
        - ``magnitude_disparity`` (*float*): Coefficient of variation (std/mean)
          of per-task gradient L2 norms.
        - ``cosine_variance`` (*float*): Variance of all pairwise cosine
          similarities.
        - ``gradient_conflict_score`` (*float*): Combined GCS (see above).
        - ``gradient_norms`` (*List[float]*): L2 norm for each task gradient.
        - ``pairwise_cosine_sims`` (*List[float]*): Cosine similarity for every
          evaluated pair, in the order produced by ``itertools.combinations``.
    """
    # ── Degenerate case ───────────────────────────────────────────────────────
    zeros: Dict[str, object] = {
        "conflict_rate": 0.0,
        "mean_cosine_sim": 0.0,
        "conflict_severity": 0.0,
        "magnitude_disparity": 0.0,
        "cosine_variance": 0.0,
        "gradient_conflict_score": 0.0,
        "gradient_norms": [],
        "pairwise_cosine_sims": [],
    }
    if not task_grads:
        return zeros

    # ── Per-task norms ────────────────────────────────────────────────────────
    gradient_norms: List[float] = [g.norm().item() for g in task_grads]

    # ── Pairwise statistics ───────────────────────────────────────────────────
    cosine_sims: List[float] = []
    conflict_flags: List[float] = []
    conflict_severities: List[float] = []

    for i, j in itertools.combinations(range(len(task_grads)), 2):
        norm_i = gradient_norms[i]
        norm_j = gradient_norms[j]

        # Skip near-zero gradient vectors to avoid numerical noise.
        if norm_i < 1e-8 or norm_j < 1e-8:
            continue

        g_i, g_j = task_grads[i], task_grads[j]
        dot = torch.dot(g_i, g_j).item()

        cos_sim = dot / (norm_i * norm_j)
        cosine_sims.append(cos_sim)

        if cos_sim < 0:
            conflict_flags.append(1.0)
            # Severity is the unsigned magnitude of the negative dot product.
            conflict_severities.append(abs(dot))
        else:
            conflict_flags.append(0.0)
            conflict_severities.append(0.0)

    # If every pair was skipped (all near-zero), return zeros.
    if not cosine_sims:
        return {**zeros, "gradient_norms": gradient_norms}

    # ── Aggregate scalars ─────────────────────────────────────────────────────
    conflict_rate = sum(conflict_flags) / len(conflict_flags)
    mean_cosine_sim = sum(cosine_sims) / len(cosine_sims)
    conflict_severity = sum(conflict_severities) / len(conflict_severities)

    # Variance of pairwise cosine similarities.
    n = len(cosine_sims)
    mean_cos = mean_cosine_sim
    cosine_variance = sum((c - mean_cos) ** 2 for c in cosine_sims) / n

    # ── Magnitude disparity (coefficient of variation of norms) ──────────────
    mean_norm = sum(gradient_norms) / len(gradient_norms)
    if mean_norm < 1e-8:
        magnitude_disparity = 0.0
    else:
        variance_norms = sum((n_ - mean_norm) ** 2 for n_ in gradient_norms) / len(gradient_norms)
        std_norms = variance_norms ** 0.5
        magnitude_disparity = std_norms / mean_norm

    # ── Combined Gradient Conflict Score ─────────────────────────────────────
    gcs = (
        0.4 * conflict_rate
        + 0.3 * (conflict_severity / (mean_norm + 1e-8))
        + 0.2 * magnitude_disparity
        + 0.1 * cosine_variance
    )

    return {
        "conflict_rate": conflict_rate,
        "mean_cosine_sim": mean_cosine_sim,
        "conflict_severity": conflict_severity,
        "magnitude_disparity": magnitude_disparity,
        "cosine_variance": cosine_variance,
        "gradient_conflict_score": gcs,
        "gradient_norms": gradient_norms,
        "pairwise_cosine_sims": cosine_sims,
    }


def compute_task_gradients(
    model: nn.Module,
    loaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    num_batches: int = 5,
) -> List[torch.Tensor]:
    """Estimate per-task encoder gradients by averaging over several batches.

    For each task the function:
      1. Draws up to ``num_batches`` batches from the task's dataloader.
      2. Runs a forward pass through ``model`` (using the task name as key).
      3. Calls ``loss.backward()`` to accumulate gradients on the encoder.
      4. Averages the accumulated gradients across batches.
      5. Flattens all encoder parameter gradients into a single 1-D tensor.

    The model weights are **never updated** (no ``optimizer.step()`` call).

    Args:
        model: A ``MultiTaskModel`` instance that exposes a ``.encoder``
               sub-module and accepts ``task_name`` as its first forward
               argument.  The forward pass must return a dict with a ``"loss"``
               key when ``labels`` are provided.
        loaders: A ``dict`` mapping task name strings to ``DataLoader``
                 instances.  Each batch must be a dict with at least
                 ``input_ids``, ``attention_mask``, and ``labels`` keys.
                 An optional ``token_type_ids`` key is forwarded if present.
        device: The ``torch.device`` on which tensors should be placed.
        num_batches: Number of batches to average gradients over per task.
                     Defaults to ``5``.

    Returns:
        A list of 1-D ``torch.Tensor`` objects, one per task, containing the
        averaged flattened encoder gradients.  The order matches
        ``list(loaders.keys())``.
    """
    model.eval()  # Disable dropout / batch-norm side effects but keep autograd.
    loss_fn = nn.CrossEntropyLoss()

    task_flat_grads: List[torch.Tensor] = []

    for task_name, loader in loaders.items():
        # Accumulator tensors for encoder parameters (lazily initialised).
        accumulated: List[torch.Tensor] = []
        batches_seen = 0

        for batch in loader:
            if batches_seen >= num_batches:
                break

            # ── Move batch to device ──────────────────────────────────────────
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            # ── Forward + backward ────────────────────────────────────────────
            model.zero_grad()  # Clear stale gradients before each batch.

            outputs = model(
                task_name=task_name,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )

            loss: torch.Tensor = outputs["loss"]
            loss.backward()

            # ── Accumulate encoder gradients ──────────────────────────────────
            batch_grads: List[torch.Tensor] = []
            for param in model.encoder.parameters():
                if param.grad is not None:
                    batch_grads.append(param.grad.detach().clone().view(-1))
                else:
                    batch_grads.append(torch.zeros(param.numel(), device=device))

            batch_flat = torch.cat(batch_grads)

            if batches_seen == 0:
                accumulated = batch_flat
            else:
                accumulated = accumulated + batch_flat

            batches_seen += 1

        # ── Average over batches ──────────────────────────────────────────────
        if batches_seen > 0:
            task_flat_grads.append(accumulated / batches_seen)
        else:
            # Dataloader was empty — emit a zero vector matching encoder size.
            num_encoder_params = sum(
                p.numel() for p in model.encoder.parameters()
            )
            task_flat_grads.append(torch.zeros(num_encoder_params, device=device))

    return task_flat_grads
