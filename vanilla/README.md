# Vanilla MTL Training Pipeline

## Overview

This repository contains code for training a model using a “vanilla” training pipeline with custom gradient signal handling and dataset configuration.

---

## Environment Setup

Activate your Python environment and configure cache directories (important for HPC / scratch usage):

```bash
source /scratch1/$USER/mtl_env/bin/activate
export HF_HOME=/scratch1/$USER/hf_home
export HF_DATASETS_CACHE=/scratch1/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch1/$USER/hf_home
export HF_HUB_CACHE=/scratch1/$USER/hf_home/hub
```

These environment variables ensure that Hugging Face models and datasets are cached in the scratch space instead of the home directory.

---

## Project Directory

Navigate to the project directory before running training:

```bash
cd /home1/arpitasa/CSCI567
```

---

## Training Script

The main training script is:

```
train_vanilla_new_data.py
```

### Key Components

- `model.py` – Model architecture  
- `data.py` – Dataset loading and preprocessing  
- `config.py` – Configuration settings  
- `gradient_signals.py` – Custom gradient signal logic  

---

## Running Training

Use the following command to start training:

```bash
python train_vanilla_new_data.py \
    --batch_size 32 \
    --num_epochs 20 \
    --num_workers 2 \
    --grad_signal_batches 4 \
    --steps_per_epoch 10000 \
    --eval_every 1000 \
    --patience 5 \
    --output_dir /scratch1/$USER/mtl_outputs/vanilla_new_data_0415
```

---

## Argument Descriptions

| Argument | Description |
|---------|------------|
| `--batch_size` | Number of samples per batch |
| `--num_epochs` | Total training epochs |
| `--num_workers` | Data loading workers |
| `--grad_signal_batches` | Number of batches used for gradient signal computation |
| `--steps_per_epoch` | Training steps per epoch |
| `--eval_every` | Evaluation frequency (in steps) |
| `--patience` | Early stopping patience |
| `--output_dir` | Directory to save outputs/checkpoints |

---

## Outputs

Training outputs (models, logs, checkpoints) will be saved to:

```
/scratch1/$USER/mtl_outputs/vanilla_new_data_0415
```

---

## Notes

- Ensure sufficient space in `/scratch1` before training.  
- Adjust `num_workers` based on available CPU resources.  
- You may tune `steps_per_epoch` and `eval_every` depending on dataset size.  

---

## Example Workflow

```bash
# 1. Activate environment
source /scratch1/$USER/mtl_env/bin/activate

# 2. Set cache paths
export HF_HOME=/scratch1/$USER/hf_home
export HF_DATASETS_CACHE=/scratch1/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch1/$USER/hf_home
export HF_HUB_CACHE=/scratch1/$USER/hf_home/hub

# 3. Navigate to repo
cd /home1/arpitasa/CSCI567

# 4. Run training
python train_vanilla_new_data.py --batch_size 32 ...
```

---

## NLP Tasks in the MTL Setup

This MTL code trains and evaluates 4 NLP tasks with a shared encoder and separate task heads:

- Yelp sentiment classification  
- QNLI (Question Natural Language Inference)  
- QQP (Quora Question Pairs)  
- MNLI (Multi-Genre Natural Language Inference)  

### What the code does for these tasks

- Uses one shared encoder across all tasks  
- Keeps a separate classification head for each task  
- Computes validation metrics per task  
- Applies per-task early stopping  
- Freezes a task head once its validation loss stops improving  
- Logs gradient signals across the four tasks  

### Task Workflow

- Preprocess the datasets for Yelp, QNLI, QQP, and MNLI  
- Build task-specific train and validation dataloaders  
- Sample tasks uniformly during multitask training  
- Train the shared model and task heads jointly  
- Evaluate each task separately during validation  
- Freeze task heads independently based on patience  

