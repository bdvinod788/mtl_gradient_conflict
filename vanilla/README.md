# Vanilla MTL — Setup and Running Guide

Gradient Conflict as a Signal for Early Stopping in Multi-Task Learning
CSCI 567 — Spring 2026

---

## Project Overview

Trains a shared DistilBERT encoder on four GLUE tasks simultaneously:
- **SST-2** — sentiment classification (positive/negative)
- **QNLI** — question-answer natural language inference
- **QQP** — question paraphrase detection
- **MNLI** — natural language inference (entailment/neutral/contradiction)

Gradient conflict metrics (conflict rate, severity, variance) are logged every epoch alongside validation loss to study whether gradient signals can replace validation-based early stopping.

---

## File Structure

```
CSCI567/
├── model.py                              # VanillaMTLModel architecture
├── data.py                               # GLUE data loading + UniformMTLSampler
├── gradient_signals.py                   # Conflict rate / severity / variance
├── train_vanilla_mtl_more_validations.py # Main training script
├── train_pcgrad_mtl.py                   # PCGrad training script
├── pcgrad.py                             # PCGrad optimizer wrapper
├── infer.py                              # Inference script
└── plot_results.py                       # Plot training curves
```

---

## Setup on CARC (USC Discovery Cluster)

### Step 1: SSH into CARC

```bash
ssh your_netid@discovery.usc.edu
```

### Step 2: Create virtual environment on scratch1

Do this once. Do NOT create the venv in your home directory — it will exceed the quota.

```bash
python -m venv /scratch1/$USER/mtl_env
source /scratch1/$USER/mtl_env/bin/activate
```

### Step 3: Install dependencies

```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.38.0 datasets numpy matplotlib
```

### Step 4: Redirect HuggingFace cache to scratch1

Add these to your `~/.bashrc` so they are set automatically every session:

```bash
echo 'export HF_HOME=/scratch1/$USER/hf_home' >> ~/.bashrc
echo 'export HF_DATASETS_CACHE=/scratch1/$USER/hf_cache' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/scratch1/$USER/hf_home' >> ~/.bashrc
echo 'export HF_HUB_CACHE=/scratch1/$USER/hf_home/hub' >> ~/.bashrc
source ~/.bashrc
```

### Step 5: Upload your files

Upload all `.py` files to your project folder on CARC using the JupyterLab file browser or scp:

```bash
# From your local machine
scp *.py your_netid@discovery.usc.edu:/home1/your_netid/CSCI567/
```

---

## Running Training

### Every new session — activate environment first

```bash
source /scratch1/$USER/mtl_env/bin/activate
export HF_HOME=/scratch1/$USER/hf_home
export HF_DATASETS_CACHE=/scratch1/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch1/$USER/hf_home
export HF_HUB_CACHE=/scratch1/$USER/hf_home/hub
cd /home1/arpitasa/CSCI567
```

### Verify GPU is available

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Should print: True and the GPU name
```

### Smoke test (run this first to catch errors quickly)

```bash
python train_vanilla_mtl_more_validations.py \
    --max_train_samples 500 \
    --max_val_samples 200 \
    --batch_size 16 \
    --num_epochs 2 \
    --num_workers 2 \
    --steps_per_epoch 100 \
    --eval_every 50 \
    --grad_signal_batches 2 \
    --output_dir /scratch1/$USER/mtl_outputs/smoke_test
```

If it completes without errors and saves `best_model.pt` and `training_history.json` you are ready for the full run.

### Full training run

```bash
python train_vanilla_mtl_more_validations.py \
    --batch_size 32 \
    --num_epochs 20 \
    --num_workers 2 \
    --grad_signal_batches 4 \
    --steps_per_epoch 7000 \
    --eval_every 1000 \
    --patience 5 \
    --output_dir /scratch1/$USER/mtl_outputs/vanilla_mtl
```

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--batch_size` | 32 | Examples per training step |
| `--num_epochs` | 20 | Maximum epochs (stops early if all tasks freeze) |
| `--steps_per_epoch` | 3000 | Fixed steps per epoch. Set to 7000 to reduce task imbalance |
| `--eval_every` | 1000 | Validate every N steps within an epoch |
| `--patience` | 3 | Consecutive checks without improvement before freezing a task head |
| `--grad_signal_batches` | 4 | Batches used to estimate gradient signals per epoch |
| `--num_workers` | 2 | DataLoader worker processes. Use 2 on CARC |
| `--output_dir` | ./outputs/vanilla_mtl | Where checkpoints and history are saved |
| `--max_train_samples` | None | Cap training examples per task (use 500 for smoke test) |

---

## What Happens During Training

**Each step:**
1. Sampler randomly picks one of the 4 tasks
2. A batch is fed through shared DistilBERT encoder and that task's head
3. Cross-entropy loss computed and divided by number of active tasks (gradient normalisation)
4. Gradients flow backward and weights are updated

**Every 1000 steps (mid-epoch validation):**
1. Validation loss and accuracy computed for all 4 tasks
2. Each task's patience counter updated
3. If a task's counter reaches `--patience`, its head is frozen (weights locked)
4. Best model saved if any task improved

**End of each epoch:**
1. Full validation run
2. Gradient signals computed (conflict rate, severity, variance, combined score)
3. All metrics saved to `training_history.json`
4. Training stops if all 4 task heads are frozen

---

## Outputs

After training your output directory will contain:

```
vanilla_mtl/
├── best_model.pt           # Model checkpoint at best validation loss
└── training_history.json   # Per-epoch metrics including gradient signals
```

The `training_history.json` contains for each epoch:
- `avg_val_loss` — average validation loss across all tasks
- `per_task_val` — per-task validation loss and accuracy
- `gradient_conflict_rate` — fraction of task pairs with conflicting gradients
- `gradient_conflict_severity` — magnitude of conflicts when they occur
- `gradient_variance` — spread of gradients across tasks
- `combined_gradient_score` — weighted combination of the above three
- `frozen_tasks` — which task heads were frozen at this point
- `mid_epoch_checks` — validation results at every 1000-step checkpoint

---

## Generating Plots

After training finishes:

```bash
pip install matplotlib

# Vanilla MTL only
python plot_results.py \
    --history /scratch1/$USER/mtl_outputs/vanilla_mtl/training_history.json \
    --output /scratch1/$USER/plots/vanilla \
    --title "Vanilla MTL"

# Compare with PCGrad
python plot_results.py \
    --history /scratch1/$USER/mtl_outputs/vanilla_mtl/training_history.json \
    --history2 /scratch1/$USER/mtl_outputs/pcgrad_mtl/training_history.json \
    --output /scratch1/$USER/plots/comparison \
    --title "Vanilla MTL" --title2 "PCGrad MTL"
```

Download plots to your local machine:

```bash
# Run on your local machine
scp -r arpitasa@discovery.usc.edu:/scratch1/arpitasa/plots ~/Downloads/mtl_plots
```

---

## Expected Results

Training should complete in roughly 10-15 epochs. Expected final validation accuracies:

| Task | Expected Accuracy |
|---|---|
| SST-2 | 89-91% |
| QNLI | 87-90% |
| QQP | 88-91% |
| MNLI | 78-82% |

Tasks typically freeze in this order: SST-2 first (smallest dataset, converges fastest), then QNLI, then QQP and MNLI last.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'datasets'`**
Venv is not activated. Run `source /scratch1/$USER/mtl_env/bin/activate` first.

**`Disk quota exceeded`**
Home directory is full. Make sure HuggingFace cache variables are set to scratch1. Run `du -sh /home1/$USER/* | sort -rh` to find what is taking space.

**`Using device: cpu`**
PyTorch cannot see the GPU. Run `pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121` to reinstall with CUDA support.

**Job gets killed (OOM)**
Reduce `--grad_signal_batches` to 2 or reduce `--batch_size` to 16.

**`tmux: command not found`**
Use nohup instead:
```bash
nohup python train_vanilla_mtl_more_validations.py [args] > train.log 2>&1 &
```
Then monitor with `tail -f train.log`.