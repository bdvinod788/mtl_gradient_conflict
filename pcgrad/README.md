# PCGrad MTL — Setup and Running Guide

Gradient Conflict as a Signal for Early Stopping in Multi-Task Learning
CSCI 567 — Spring 2026

---

## What This Experiment Does

Trains a shared DistilBERT encoder on four GLUE tasks simultaneously using
**PCGrad (Gradient Surgery)** — the key difference from the vanilla baseline.

At each training step, instead of updating on a single task at a time, PCGrad:
1. Collects **one batch per active task**
2. Computes each task's gradient separately
3. For every pair of tasks where `dot(g_i, g_j) < 0` (conflict), **projects**
   `g_i` to remove the component that directly opposes `g_j`
4. Averages the projected gradients and applies the update

This prevents tasks from actively harming each other's learning.

Tasks:
- **SST-2** — sentiment classification (positive/negative)
- **QNLI** — question-answer natural language inference
- **QQP** — question paraphrase detection
- **MNLI** — natural language inference (3-class)

---

## File Structure

```
pcgrad/
├── pcgrad.py             # PCGrad optimizer wrapper (Gradient Surgery)
├── model.py              # Shared DistilBERT encoder + 4 task heads
├── data.py               # GLUE data loading + task iterators
├── gradient_signals.py   # Conflict rate / severity / variance metrics
├── train_pcgrad_mtl.py   # Main training script (Experiment 2)
└── plot_results.py       # Plot and compare vanilla vs PCGrad results
```

> This folder is **fully self-contained** — no files from `vanilla/` or
> `models/` are imported at runtime.  All shared utilities are local copies.

---

## Setup on CARC (USC Discovery Cluster)

Same environment as vanilla/:

```bash
source /scratch1/$USER/mtl_env/bin/activate
export HF_HOME=/scratch1/$USER/hf_home
export HF_DATASETS_CACHE=/scratch1/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch1/$USER/hf_home
export HF_HUB_CACHE=/scratch1/$USER/hf_home/hub
cd /home1/<netid>/CSCI567/pcgrad
```

---

## Running Training

### Smoke test

```bash
python train_pcgrad_mtl.py \
    --max_train_samples 500 \
    --max_val_samples 200 \
    --batch_size 16 \
    --num_epochs 2 \
    --num_workers 2 \
    --steps_per_epoch 100 \
    --eval_every 50 \
    --grad_signal_batches 2 \
    --output_dir /scratch1/$USER/mtl_outputs/pcgrad_smoke
```

### Full training run (Experiment 2)

Use **identical hyperparameters** to the vanilla run for a fair comparison:

```bash
python train_pcgrad_mtl.py \
    --batch_size 32 \
    --num_epochs 20 \
    --num_workers 4 \
    --grad_signal_batches 4 \
    --steps_per_epoch 7000 \
    --eval_every 1000 \
    --patience 5 \
    --output_dir /scratch1/$USER/mtl_outputs/pcgrad_mtl_<date>
```

> **Note on GPU time**: PCGrad runs N_active forward+backward passes per step
> (instead of 1), so each step takes roughly 4× longer in epoch 1.  As tasks
> freeze the overhead decreases.  Keep `--steps_per_epoch` the same as vanilla
> so wall-clock comparisons are honest.

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--batch_size` | 32 | Batch size *per task* per step |
| `--num_epochs` | 20 | Maximum epochs |
| `--steps_per_epoch` | 7000 | Fixed steps per epoch — keep identical to vanilla |
| `--eval_every` | 1000 | Validate every N steps within an epoch |
| `--patience` | 5 | Checks without improvement before freezing a task head |
| `--grad_signal_batches` | 4 | Batches used to estimate gradient signals per epoch |
| `--num_workers` | 2 | DataLoader workers — use 4 on CARC |
| `--output_dir` | ./outputs/pcgrad_mtl | Where checkpoints and history are saved |

---

## What Happens During Training

**Each step:**
1. For each active (non-frozen) task, draw one batch
2. Run a forward pass through shared encoder → task head → cross-entropy loss
3. Call `pcgrad.pc_backward(task_losses)` which:
   - Computes each task's gradient individually via `.backward()`
   - For each conflicting pair `(i, j)` where `dot(g_i, g_j) < 0`:
     project `g_i ← g_i − [dot(g_i,g_j) / ‖g_j‖²] · g_j`
   - Averages projected gradients and writes to `.grad`
4. Clip gradients, call `optimizer.step()`

**Every 1000 steps (mid-epoch validation):**
- Same as vanilla: val loss + accuracy for all 4 tasks
- Per-task patience counters updated
- Head frozen if counter ≥ `--patience`

**End of each epoch:**
- Full validation + gradient signal computation
- `training_history.json` updated with all metrics plus:
  - `pcgrad_conflicts_total` — total conflict pairs detected this epoch
  - `pcgrad_conflicts_per_step` — average conflicts per step

---

## Outputs

```
<output_dir>/
├── best_model.pt          # Best checkpoint by validation loss
└── training_history.json  # Per-epoch metrics (same schema as vanilla + pcgrad fields)
```

---

## Generating Plots

After both vanilla and PCGrad runs complete, run from the `pcgrad/` directory:

```bash
pip install matplotlib   # if not already installed

# Compare vanilla vs PCGrad
python plot_results.py \
    --history  /scratch1/$USER/mtl_outputs/vanilla_mtl_<date>/training_history.json \
    --history2 /scratch1/$USER/mtl_outputs/pcgrad_mtl_<date>/training_history.json \
    --output   /scratch1/$USER/plots/comparison \
    --title    "Vanilla MTL" \
    --title2   "PCGrad MTL"
```

Download plots to your local machine:

```bash
# Run on your local machine
scp -r <netid>@discovery.usc.edu:/scratch1/<netid>/plots ~/Downloads/mtl_plots
```

Plots generated:
- `accuracy_per_task.png` — 4-panel per-task accuracy with freeze markers
- `task_accuracy_overlay.png` — all 4 tasks on one chart
- `avg_val_loss.png` — average validation loss comparison
- `conflict_rate.png` — gradient conflict rate over epochs (**key research plot**)
- `gradient_score.png` — combined GCS over epochs
- `pcgrad_conflicts.png` — raw PCGrad conflict counts per epoch
- `summary_grid.png` — 2×2 summary grid for the report

---

## Expected Results

PCGrad should show:
- **Lower gradient conflict rate** than vanilla (the main hypothesis)
- Comparable or better accuracy on harder tasks (QNLI, MNLI)
- Higher per-step training time due to N_active backward passes

| Task | Vanilla (actual) | PCGrad (expected) |
|---|---|---|
| SST-2 | 86.01% | 87–90% |
| QNLI | 86.31% | 87–90% |
| QQP | 89.35% | 88–91% |
| MNLI | 80.36% | 79–83% |
