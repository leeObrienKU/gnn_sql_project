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
  --cutoff 1999-01-01 \
  --class_weights true \
  --wandb \
  --wandb_project sql_to_gnn \
  --wandb_api_key 16f84cf08205b725a7c2e2a21b572843e5bd1c69

# Configuration explanation:
# - Earlier cutoff (1999-01-01) for more stable attrition period
# - Class weights enabled to handle 80/20 imbalance
# - Balanced dropout (0.5) for regularization
# - Target 58% recall with class-weighted predictions
# - 3 layers for model capacity
# 
# Data characteristics:
# - Class imbalance: 80% current, 20% former employees
# - Peak attrition ~3.7% around 2000
# - More stable attrition rates pre-2000
# - Using class weights to balance training