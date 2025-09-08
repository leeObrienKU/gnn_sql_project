#!/bin/bash
cd /content/gnn_sql_project

python main.py \
  --model GCN \
  --epochs 150 \
  --batch_size 2048 \
  --hidden_dim 256 \
  --num_layers 3 \
  --lr 0.002 \
  --lr_decay_type step \
  --lr_decay_step_size 10 \
  --lr_decay_gamma 0.90 \
  --task attrition \
  --auto_threshold \
  --cutoff 2000-01-01 \
  --wandb \
  --wandb_project sql_to_gnn \
  --wandb_api_key 16f84cf08205b725a7c2e2a21b572843e5bd1c69

# Optional parameters:
# --current_edges_only  # uncomment for sparser, current-only edges
# --wandb_entity your_team  # optional if you have a W&B team