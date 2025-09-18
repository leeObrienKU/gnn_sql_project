import argparse
import json
import os
import time
from datetime import datetime

import torch
from torch_geometric.loader import NeighborLoader

from utils.plots import plot_training_curves, plot_confusion_matrix
from utils.experiment_logger import ExperimentLogger
import numpy as np
import pandas as pd
import networkx as nx

from utils.data_loader import load_employees_db
from utils.graph_builder import create_graph
from models.gnn_model import GNN
from models.trainer import train_and_evaluate


### references below ### 
# code and references used for this project
# https://github.com/snap-stanford/relbench used for reference code throughout the project
# https://github.com/snap-stanford/relbench/blob/main/relbench/modeling/graph.py
# https://github.com/snap-stanford/relbench/blob/main/relbench/modeling/loader.py
# https://github.com/sailab-code/gnn/blob/master/GNN.py
# https://github.com/chaitjo/efficient-gnns/blob/main/ppi_pyg/scripts/run.sh
#Generative AI prompt reference :
# O’Brien, Lee. (2025). “Tell me how to structure a scalable, reproducible GNN attrition project 
# from SQL data (supporting GCN, GAT, GraphSAGE) with end-to-end experiment logging.” 
# [Research prompt; via Cursor; claude-monnet.3.7].
# 


# Weights & Biases - this helped me with tracking and visuals ( did not use in final report)
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False

def main():
    parser = argparse.ArgumentParser(description='GNN Employee Database Training')
    
    # model selection from the script files 
    parser.add_argument('--model', type=str, default='GCN',
                        choices=['GCN', 'GAT', 'GraphSAGE'],
                        help='Type of GNN model')
    
    #  parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for neighbor sampling')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads for GAT')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--lr_decay_type', type=str, default='exponential',
                        choices=['none','exponential','step'],
                        help='Learning rate decay scheduler type')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.95,
                        help='LR decay factor (gamma)')
    parser.add_argument('--lr_decay_step_size', type=int, default=20,
                        help='Step size for StepLR (epochs)')
    parser.add_argument('--task', type=str, default='attrition',
                        choices=['dept', 'attrition'],
                        help='Prediction task')
    parser.add_argument('--cutoff', type=str, default="2002-12-31",
                        help='Cutoff date YYYY-MM-DD for attrition labeling')
    parser.add_argument('--class_weights', action='store_true',
                        help='Enable class weights for imbalanced training')
    parser.add_argument('--current_edges_only', action='store_true',
                        help='Keep only current edges')
    parser.add_argument('--pos_threshold', type=float, default=0.5,
                        help='Positive class threshold')
    parser.add_argument('--auto_threshold', action='store_true',
                        help='Auto-select threshold')
    parser.add_argument('--threshold_mode', type=str, default=None,
                        choices=['fixed', 'max_f1', 'target_precision', 'target_recall'],
                        help='Threshold selection mode')
    parser.add_argument('--target_precision', type=float, default=None,
                        help='Target precision')
    parser.add_argument('--target_recall', type=float, default=None,
                        help='Target recall')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable W&B logging')
    parser.add_argument('--wandb_project', type=str, default='sql_to_gnn',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity')
    parser.add_argument('--wandb_api_key', type=str, default=None,
                        help='W&B API key')
    
    args = parser.parse_args()
    
    #  logging
    logger = ExperimentLogger()
    
    # w&b setup
    if args.wandb and _WANDB_AVAILABLE:
        api_key = args.wandb_api_key or os.environ.get('WANDB_API_KEY')
        if api_key:
            os.environ['WANDB_API_KEY'] = api_key
        logger.wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={},
            name=f"{args.model}"
        )
    
    # Log parameters
    logger.log_params(vars(args))
    
    # need some data to make the magic happen
    print("\nLoading and preparing data...")
    employees, departments, dept_emp, dept_manager, titles, salaries = load_employees_db()
    
    # build graph
    print("\nBuilding graph...")
    data = create_graph(
        employees=employees,
        departments=departments,
        dept_emp=dept_emp,
        dept_manager=dept_manager,
        titles=titles,
        salaries=salaries,
        task=args.task,
        cutoff_date=args.cutoff,
        use_all_history_edges=not args.current_edges_only
    )
    
    # create appropriate model
    input_dim = data.x.shape[1]
    num_classes = data.num_classes
    
    model = GNN(
        model_type=args.model,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_heads=args.num_heads
    )
    

    if logger.wandb_run is not None:
        try:
            wandb.watch(model, log="all", log_freq=100)  # Log gradients and parameters
        except Exception:
            pass
    
    # Training setup
    # Adjust num_neighbors based on num_layers
    num_neighbors = [30] * args.num_layers
    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=args.batch_size,
        input_nodes=data.train_mask,
        shuffle=True
    )
    
    # train and evaluate
    threshold_mode = args.threshold_mode or ('max_f1' if args.auto_threshold else 'fixed')
    test_acc = train_and_evaluate(
        model=model,
        data=data,
        train_loader=train_loader,
        epochs=args.epochs,
        lr=args.lr,
        logger=logger,
        pos_threshold=args.pos_threshold,
        auto_threshold=args.auto_threshold,
        threshold_mode=threshold_mode,
        target_precision=args.target_precision,
        target_recall=args.target_recall,
        lr_decay_type=args.lr_decay_type,
        lr_decay_gamma=args.lr_decay_gamma,
        lr_decay_step_size=args.lr_decay_step_size,
        patience=args.patience,
        use_class_weights=args.class_weights
    )
    
    # plots
    out_dir = logger.log_dir
    plot_training_curves(logger.metrics["training"], out_dir)
    cm = np.array(logger.metrics.get("confusion_matrix", [[0, 0], [0, 0]]))
    class_names = ["Stay", "Leave"]
    cm_path, cm_norm_path = plot_confusion_matrix(cm, class_names, os.path.join(out_dir, "confusion_matrix.png"))
    
    # log
    if logger.wandb_run is not None:
        try:
            logger.wandb_run.log({
                "plots/training_loss": wandb.Image(os.path.join(out_dir, "training_loss.png")),
                "plots/val_accuracy": wandb.Image(os.path.join(out_dir, "val_accuracy.png")),
                "plots/confusion_matrix": wandb.Image(cm_path),
                "plots/confusion_matrix_normalized": wandb.Image(cm_norm_path),
                "confusion_matrix/raw": cm
            })
        except Exception:
            pass
    
   
    logger.finalize(
        test_acc=test_acc,
        model_summary={
            "type": args.model,
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "parameters": sum(p.numel() for p in model.parameters())
        }
    )
    
    if logger.wandb_run is not None:
        try:
            logger.wandb_run.finish()
        except Exception:
            pass
    
    print("\nFinal Training Summary")
    print(f"Model: {args.model}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())} total")

if __name__ == "__main__":
    main()