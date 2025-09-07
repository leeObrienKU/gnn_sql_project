#!/bin/bash

# Base parameters for all experiments
BASE_PARAMS="
    --hidden_dim 256 \
    --num_layers 2 \
    --dropout 0.5 \
    --epochs 100 \
    --batch_size 2048 \
    --lr 0.001 \
    --wandb True \
    --wandb_project sql_to_gnn \
    --train_cutoff 1999-01-01 \
    --val_cutoff 2000-01-01 \
    --test_cutoff 2001-01-01"

echo "🔄 Running GNN Experiments..."

# 1. GCN on Homogeneous Graph
echo "🔹 Running GCN Experiment..."
python main.py \
    --model GCN \
    --graph_type homogeneous \
    $BASE_PARAMS \
    --experiment_name "gcn_homogeneous"

# 2. GAT on Homogeneous Graph
echo "🔹 Running GAT Experiment..."
python main.py \
    --model GAT \
    --graph_type homogeneous \
    --heads 4 \
    $BASE_PARAMS \
    --experiment_name "gat_homogeneous"

# 3. GraphSAGE on Homogeneous Graph
echo "🔹 Running GraphSAGE Experiment..."
python main.py \
    --model GraphSAGE \
    --graph_type homogeneous \
    $BASE_PARAMS \
    --experiment_name "graphsage_homogeneous"

# 4. Heterogeneous Graph Experiment
echo "🔹 Running Heterogeneous Graph Experiment..."
python main.py \
    --model HeteroGNN \
    --graph_type heterogeneous \
    --heads 4 \
    $BASE_PARAMS \
    --experiment_name "hetero_gnn"

echo "✨ All experiments completed!"

# Generate comprehensive EDA report
echo "📊 Generating EDA Reports..."
python db_inspector.py 2001-01-01  # Use test cutoff for final analysis