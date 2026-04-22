# Training Pipeline README

## Overview
This repository contains code for multitask learning (MTL) across four NLP tasks using a shared model and task-specific heads.

---

## Environment Setup

```bash
source /scratch1/$USER/mtl_env/bin/activate

export HF_HOME=/scratch1/$USER/hf_home
export HF_DATASETS_CACHE=/scratch1/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch1/$USER/hf_home
export HF_HUB_CACHE=/scratch1/$USER/hf_home/hub
```

---

## Project Directory

```bash
cd /home1/arpitasa/CSCI567
```

---

## NLP Tasks (MTL)

The model is trained jointly on 4 NLP tasks:

- Yelp – Sentiment classification  
- QNLI – Question Natural Language Inference  
- QQP – Quora Question Pairs (paraphrase detection)  
- MNLI – Multi-Genre Natural Language Inference  

### MTL Setup
- Shared encoder across all tasks  
- Task-specific classification heads  
- Per-task validation and early stopping  
- Task heads frozen individually when performance plateaus  
- Gradient signals tracked across tasks  

---

## Training

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

## Outputs

Saved to:
```
/scratch1/$USER/mtl_outputs/vanilla_new_data_0415
```

---

## Files

- `train_vanilla_new_data.py` – training loop  
- `model.py` – model architecture  
- `data.py` – dataset handling  
- `config.py` – configs  
- `gradient_signals.py` – gradient signal tracking  

---

## Notes

- Use scratch space for caching to avoid quota issues  
- Adjust workers based on CPU availability  
- Tune hyperparameters as needed  
