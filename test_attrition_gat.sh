#!/bin/bash
cd /content/gnn_sql_project

python main.py \
  --model GAT \
  --epochs 150 \
  --batch_size 4096 \
  --hidden_dim 128 \
  --num_layers 3 \
  --num_heads 4 \
  --concat_heads \
  --residual \
  --feat_dropout 0.5 \
  --attn_dropout 0.3 \
  --lr 0.001 \
  --lr_warmup_epochs 10 \
  --lr_decay_type cosine \
  --lr_decay_gamma 0.85 \
  --gradient_clip 1.0 \
  --weight_decay 0.01 \
  --task attrition \
  --threshold_mode target_recall \
  --target_recall 0.58 \
  --cutoff 1999-01-01 \
  --class_weights \
  --wandb \
  --wandb_project sql_to_gnn \
  --wandb_api_key 16f84cf08205b725a7c2e2a21b572843e5bd1c69

# Configuration explanation:
# 1. Architecture Enhancements:
#    - 4 attention heads per layer
#    - Concatenated head outputs
#    - Residual connections
#    - 3 layers with 128 hidden dims
#
# 2. Regularization Strategy:
#    - Feature dropout: 0.5
#    - Attention dropout: 0.3
#    - Weight decay: 0.01
#    - Gradient clipping: 1.0
#
# 3. Learning Rate Schedule:
#    - Higher initial LR: 0.001
#    - 10 epoch warmup
#    - Cosine decay
#    - Gamma: 0.85
#
# 4. Training Dynamics:
#    - Larger batch size: 4096
#    - Class weights enabled
#    - Target recall: 0.58 (balanced)
#
# Expected improvements:
# - Better feature extraction (multiple heads)
# - More stable training (warmup + cosine)
# - Better generalization (dropouts + weight decay)
# - Faster convergence (residual + larger batch)
# - More balanced precision-recall