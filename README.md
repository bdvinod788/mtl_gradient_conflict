# Gradient Conflict as a Signal for Early Stopping in Multi-Task Learning

Repository for our project on early stopping in MTL. We implement a DistilBERT encoder on four NLP classification tasks (Yelp, QNLI,
QQP, MNLI) with optional PCGrad gradient surgery, and explore using
gradient statistics as an early-stopping signal in place of validation
loss.

## What's in this repo

```
.
├── train.py                    # main training script (vanilla or PCGrad)
├── pcgrad.py                   # encoder-only PCGrad implementation
├── model.py                    # DistilBERT shared encoder + 4 task heads
├── data.py                     # data loaders, test-split carving
├── config.py                   # task configs and hyperparameters
├── gradient_signals.py         # five-signal logger (severity, SNR, etc.)
│
├── plot_all_graphs.py          # 19 per-run plots from training_history.json
├── plot_severity_vs_val.py     # severity-vs-val-loss alignment figure
├── print_test_acc.py           # quick numerical summary of test result
│
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv mtl_env
source mtl_env/bin/activate
pip install -r requirements.txt
```

The first training run downloads the four datasets (Yelp polarity,
QNLI, QQP, MNLI) via HuggingFace `datasets` and tokenizes them into
`processed_data/`. Subsequent runs reuse this cache.

## Training

The same `train.py` handles both vanilla MTL and PCGrad — toggle with
`--use_pcgrad`. Two stopping strategies are supported via
`--early_stop_mode`.

### Quick start (vanilla, global stopping)

```bash
python train.py \
    --early_stop_mode global \
    --patience 5 \
    --batch_size 32 \
    --num_epochs 4 \
    --steps_per_epoch 10000 \
    --eval_every 1000 \
    --processed_dir ./processed_data \
    --output_dir ./outputs/vanilla_global \
    --final_eval_on_test
```

### PCGrad with global stopping

```bash
python train.py \
    --use_pcgrad \
    --pcgrad_accum_steps 4 \
    --early_stop_mode global \
    --patience 5 \
    --batch_size 32 \
    --num_epochs 4 \
    --steps_per_epoch 10000 \
    --eval_every 1000 \
    --processed_dir ./processed_data \
    --output_dir ./outputs/pcgrad_global \
    --final_eval_on_test
```

### Per-task stopping (either optimizer)

Replace `--early_stop_mode global` with `--early_stop_mode per_task`
and increase `--num_epochs` to 10 (per-task stopping freezes heads
gradually, so it needs more total budget). Add `--use_pcgrad` if you
want PCGrad+Per-Task.

### Long no-stopping runs (for U-shape figures)

To let validation loss complete its full descent → minimum → ascent
trajectory, set patience high enough that it never fires:

```bash
python train.py \
    --early_stop_mode global \
    --patience 9999 \
    --num_epochs 5 \
    --steps_per_epoch 10000 \
    --eval_every 1000 \
    --processed_dir ./processed_data \
    --output_dir ./outputs/vanilla_5ep
```

## Output structure

Each run writes to its `--output_dir`:

```
outputs/<run_name>/
├── training_history.json       # per-step time series with gradient signals
├── final_test_results.json     # test eval (if --final_eval_on_test)
└── best_model.pt               # checkpoint at best validation loss
```

`training_history.json` is a list of epoch dicts. Each epoch contains
end-of-epoch summary stats and a `mid_epoch_checks` list with
per-1000-step entries. Each check has `step`, `per_task_val` (loss/acc
per task), `frozen_tasks`, and the six gradient signals
(`conflict_rate`, `conflict_severity`, `gradient_variance`,
`grad_norm_ratio`, `grad_snr`, `combined_gradient_score`) at the top
level.

## Plotting

### Per-run diagnostic plots (19 figures)

```bash
python plot_all_graphs.py \
    ./outputs/<run_name>/training_history.json \
    --output_dir ./outputs/<run_name>/plots
```

Produces overall train/val curves, per-task panels, gradient-signal
trajectories, and per-task overlays with gradient signals.


### Quick numerical summary

```bash
python summarize_runs.py \
    ./outputs/vanilla_global/training_history.json \
    ./outputs/pcgrad_global/training_history.json \
    ...
```
### Quick test-set summary (across runs)
 
```bash
python print_test_acc.py \
    ./outputs/vanilla_global/final_test_results.json \
    ./outputs/pcgrad_global/final_test_results.json \
    ...
```
 

Prints final per-task accuracy, val loss, frozen-task list, and total
epochs for each run.

## Key training flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--use_pcgrad` | off | Enable PCGrad gradient surgery on the shared encoder |
| `--pcgrad_accum_steps` | 4 | Sampler batches accumulated per PCGrad optimizer step |
| `--early_stop_mode` | `per_task` | Stopping strategy: `global`, `per_task`, or `both` |
| `--patience` | 3 | Eval checks without improvement before triggering stop |
| `--batch_size` | 32 | Per-task mini-batch size |
| `--num_epochs` | 20 | Epoch cap |
| `--steps_per_epoch` | 3000 | Sampler steps per epoch |
| `--eval_every` | 1000 | Run mid-epoch validation + log gradient signals every N steps |
| `--grad_signal_batches` | 4 | Mini-batches per task used to estimate gradient signals |
| `--lr` | 2e-5 | Learning rate (AdamW) |
| `--max_length` | 128 | Tokenizer max length |
| `--final_eval_on_test` | off | Evaluate best checkpoint on held-out test split |
| `--processed_dir` | `./processed_data` | Tokenized data cache |
| `--output_dir` | `./outputs/mtl` | Where to write training_history.json and checkpoints |

## Hyperparameters used in the paper

| Parameter | Value |
|-----------|-------|
| Encoder | distilbert-base-uncased |
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Weight decay | 0.01 |
| Batch size | 32 |
| Max sequence length | 128 |
| Warmup | linear, 10% of total steps |
| LR schedule | linear decay to zero after warmup |
| Eval frequency | every 1000 sampler steps |
| Patience (val-based) | 5 evaluation checks |
| PCGrad accumulation | K=4 sampler batches per optimizer step |
| Test split seed | 42 |

## Compute requirements

A single GPU with at least 16 GB memory is sufficient. Training one
configuration takes 1–4 hours on an NVIDIA A40 depending on epoch
count. PCGrad runs ~3–4× longer in wall-clock per optimizer step than
vanilla because of the projection step, but reach comparable accuracy
in a similar number of optimizer steps overall.

CPU-only training is technically possible but impractically slow.

## Reproducing the paper results

The four ablation configurations and the two long no-stopping runs
that produced our final tables and figures:

```bash
# 2x2 ablation
python train.py --early_stop_mode global   --patience 5 --num_epochs 4  ... --output_dir ./outputs/vanilla_global
python train.py --early_stop_mode per_task --patience 5 --num_epochs 10 ... --output_dir ./outputs/vanilla_pertask
python train.py --use_pcgrad --early_stop_mode global   --patience 5 --num_epochs 4  ... --output_dir ./outputs/pcgrad_global
python train.py --use_pcgrad --early_stop_mode per_task --patience 5 --num_epochs 10 ... --output_dir ./outputs/pcgrad_pertask

# Long runs for U-shape figures
python train.py --early_stop_mode global --patience 9999 --num_epochs 5 ... --output_dir ./outputs/vanilla_5ep
python train.py --use_pcgrad --early_stop_mode global --patience 9999 --num_epochs 5 ... --output_dir ./outputs/pcgrad_5ep
```

(Replace `...` with the common flags from the Quick Start section.)

Then generate plots:

```bash
# Per-run diagnostic plots
for run in vanilla_global vanilla_pertask pcgrad_global pcgrad_pertask vanilla_5ep pcgrad_5ep; do
    python plot_all_graphs.py ./outputs/$run/training_history.json --output_dir ./outputs/$run/plots
done
```

## Datasets

Loaded automatically via HuggingFace `datasets`:

| Task | Source | Description |
|------|--------|-------------|
| Yelp | `yelp_polarity` | Binary review sentiment |
| QNLI | `glue/qnli` | Binary question-sentence entailment |
| QQP | `glue/qqp` | Binary paraphrase detection |
| MNLI | `glue/mnli` | 3-way natural language inference |

Test splits are carved deterministically (seed 42) from each task's
training data; original validation splits are unmodified.

## Acknowledgments

- PCGrad algorithm from Yu et al., "Gradient surgery for multi-task learning" (NeurIPS 2020).
- DistilBERT from Sanh et al. via HuggingFace `transformers`.
- GLUE benchmark tasks via HuggingFace `datasets`.
