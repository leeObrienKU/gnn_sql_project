#!/bin/bash
cd /content/gnn_sql_project

python main.py \
  --model GraphSAGE \
  --epochs 150 \
  --batch_size 4096 \
  --hidden_dim 256 \
  --num_layers 4 \
  --dropout 0.5 \
  --patience 30 \
  --lr 0.001 \
  --lr_decay_type step \
  --lr_decay_step_size 15 \
  --lr_decay_gamma 0.85 \
  --task attrition \
  --threshold_mode target_recall \
  --target_recall 0.58 \
  --cutoff 1999-01-01 \
  --class_weights \
  --wandb \
  --wandb_project sql_to_gnn \
  --wandb_api_key 16f84cf08205b725a7c2e2a21b572843e5bd1c69

# Configuration explanation:
# GraphSAGE-specific choices:
# - Larger batch size (4096) for better neighborhood sampling
# - More layers (4) as GraphSAGE handles depth well
# - Lower learning rate (0.001) with stronger decay
# - Longer step size (15) for more stable training
# 
# Keeping successful strategies from GCN:
# - 1999-01-01 cutoff for stable period
# - Class weights for 80/20 imbalance
# - Target 58% recall for balance
# - Dropout 0.5 for regularization
# 
# Architecture differences from GCN:
# - GraphSAGE uses mean aggregation
# - Better at capturing local structure
# - More robust to larger neighborhoods
# - Generally needs more epochs to converge