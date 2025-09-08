#!/bin/bash
cd /content/gnn_sql_project

python main.py \
  --model GCN \
  --epochs 150 \
  --batch_size 2048 \
  --hidden_dim 256 \
  --num_layers 3 \
  --dropout 0.4 \
  --patience 30 \
  --lr 0.002 \
  --lr_decay_type step \
  --lr_decay_step_size 10 \
  --lr_decay_gamma 0.90 \
  --task attrition \
  --threshold_mode target_recall \
  --target_recall 0.65 \
  --cutoff 2000-01-01 \
  --wandb \
  --wandb_project sql_to_gnn \
  --wandb_api_key 16f84cf08205b725a7c2e2a21b572843e5bd1c69

# Configuration explanation:
# - Lower dropout (0.4) to catch more patterns
# - Longer patience (30) for better convergence
# - Target 65% recall with auto threshold
# - 3 layers for more capacity
# - Step decay LR schedule for stability