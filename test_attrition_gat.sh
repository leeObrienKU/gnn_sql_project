#!/bin/bash
cd /content/gnn_sql_project

python main.py \
  --model GAT \
  --epochs 150 \
  --batch_size 4096 \
  --hidden_dim 64 \
  --num_layers 3 \
  --num_heads 4 \
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
# 1. Architecture:
#    - 4 attention heads per layer
#    - Reduced base hidden dim (64) as it gets multiplied by num_heads
#    - Effective hidden dim: 64 * 4 = 256 after concatenation
#    - Deeper network (3 layers)
#    - Higher dropout (0.6) for regularization
#
# 2. Training:
#    - Larger batch size (4096) for stability
#    - Higher initial LR (0.001)
#    - Step decay every 12 epochs
#    - Gamma 0.85 for gradual decay
#    - Early stopping patience 30
#
# 3. Task Settings:
#    - Class weights for imbalance
#    - Target recall 0.58 (balanced)
#    - 1999-01-01 cutoff (stable period)
#
# Expected improvements:
# - Better feature learning (4 attention perspectives)
# - More stable training (larger batch + step decay)
# - Better generalization (higher dropout)
# - More balanced metrics (target recall 0.58)
# - Handling class imbalance (weights + cutoff)