#!/bin/bash
cd /content/gnn_sql_project

python main.py \
  --model GAT \
  --epochs 150 \
  --batch_size 2048 \
  --hidden_dim 128 \
  --num_layers 3 \
  --dropout 0.6 \
  --patience 30 \
  --lr 0.001 \
  --lr_decay_type step \
  --lr_decay_step_size 12 \
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
# GAT-specific choices:
# - Smaller hidden_dim (128) as each head multiplies parameters
# - Higher dropout (0.6) for attention mechanism
# - Lower learning rate (0.001) for stability
# - Moderate batch size (2048) for attention computation
# - 3 layers with attention at each level
# 
# Keeping successful strategies:
# - 1999-01-01 cutoff for stable period
# - Class weights for 80/20 imbalance
# - Target 58% recall for balance
# - Step decay with shorter steps (12)
# 
# Architecture differences:
# - GAT uses attention mechanisms
# - Multiple attention heads per layer
# - More parameters per layer than GCN
# - More sensitive to learning rate
# - Benefits from higher dropout