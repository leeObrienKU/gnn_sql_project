#!/bin/bash
cd /content/gnn_sql_project

python main.py \
  --model GCN \
  --epochs 150 \
  --batch_size 2048 \
  --hidden_dim 256 \
  --num_layers 3 \
  --dropout 0.5 \
  --patience 30 \
  --lr 0.002 \
  --lr_decay_type step \
  --lr_decay_step_size 10 \
  --lr_decay_gamma 0.90 \
  --task attrition \
  --threshold_mode target_recall \
  --target_recall 0.58 \
  --cutoff 2000-01-01 \
  --wandb \
  --wandb_project sql_to_gnn \
  --wandb_api_key 16f84cf08205b725a7c2e2a21b572843e5bd1c69

# Configuration explanation:
# - Balanced dropout (0.5) for regularization
# - Target 58% recall (down from 65% for better precision)
# - 3 layers for model capacity
# - Step decay LR schedule for stability
# - Longer patience for convergence
# 
# Previous results for comparison:
# - 65% recall → 39.9% precision (too many false alarms)
# - 51% recall → 44.9% precision (missed too many)