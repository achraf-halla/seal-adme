#!/usr/bin/env python3
"""
Training script for SEAL-ADME models.

Supports:
- Multi-task pretraining on classification tasks
- Multi-task finetuning on regression tasks
- Both GCN and GIN encoder architectures

Usage:
    # Pretraining
    python scripts/train.py pretrain --config configs/model_config.yaml
    
    # Finetuning
    python scripts/train.py finetune --config configs/model_config.yaml --encoder checkpoints/pretrained_encoder.pt
    
    # Full pipeline
    python scripts/train.py all --config configs/model_config.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml
import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_graphs
from src.models import build_pretrain_model, build_finetune_model
from src.training import (
    PretrainDataset,
    PretrainTrainer,
    FinetuneTrainer,
    load_task_graphs,
)
from src.evaluation import (
    extract_explanations_all_tasks,
    visualize_all_tasks,
)


def setup_logging(log_file=None, level="INFO"):
    """Configure logging."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode='w'))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        handlers=handlers
    )


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_pretrain_datasets(graph_dir, metadata_dir):
    """Load pretraining datasets from metadata parquet files."""
    logger = logging.getLogger("load_pretrain")
    
    graph_dir = Path(graph_dir)
    metadata_dir = Path(metadata_dir)
    
    # Load train metadata
    train_meta_path = metadata_dir / "pretrain_train.parquet"
    valid_meta_path = metadata_dir / "pretrain_valid.parquet"
    
    if not train_meta_path.exists():
        raise FileNotFoundError(f"Train metadata not found: {train_meta_path}")
    
    train_meta = pd.read_parquet(train_meta_path)
    valid_meta = pd.read_parquet(valid_meta_path) if valid_meta_path.exists() else train_meta.iloc[:0]
    
    # Add graph_id column if not present
    if 'graph_id' not in train_meta.columns:
        train_meta['graph_id'] = train_meta.index
    if 'graph_id' not in valid_meta.columns:
        valid_meta['graph_id'] = valid_meta.index
    
    # Add label column
    if 'label' not in train_meta.columns:
        train_meta['label'] = train_meta['Y']
    if 'label' not in valid_meta.columns:
        valid_meta['label'] = valid_meta['Y']
    
    train_dataset = PretrainDataset(graph_dir, train_meta)
    valid_dataset = PretrainDataset(graph_dir, valid_meta)
    
    logger.info(f"Loaded pretrain datasets: train={len(train_meta)}, valid={len(valid_meta)}")
    
    return train_dataset, valid_dataset


def load_finetune_datasets(graph_dir, task_names):
    """Load finetuning datasets as graph lists per task."""
    logger = logging.getLogger("load_finetune")
    
    graph_dir = Path(graph_dir)
    task_datasets = {}
    
    for task_name in task_names:
        task_dir = graph_dir / task_name
        if not task_dir.exists():
            # Try alternate naming
            for subdir in graph_dir.iterdir():
                if subdir.is_dir() and task_name in subdir.name:
                    task_dir = subdir
                    break
        
        if not task_dir.exists():
            logger.warning(f"Task directory not found: {task_dir}")
            continue
        
        task_data = {'train': [], 'valid': [], 'test': []}
        
        for split in ['train', 'valid', 'test']:
            # Try different file patterns
            patterns = [
                f"graph_*_{split}_*.pt",
                f"*_{task_name}_{split}_*.pt",
                f"graph_{task_name}_{split}_*.pt",
            ]
            
            graphs = []
            for pattern in patterns:
                for path in sorted(task_dir.glob(pattern)):
                    try:
                        g = torch.load(path, weights_only=False)
                        graphs.append(g)
                    except Exception as e:
                        logger.warning(f"Failed to load {path}: {e}")
                
                if graphs:
                    break
            
            task_data[split] = graphs
            logger.info(f"  {task_name}/{split}: {len(graphs)} graphs")
        
        task_datasets[task_name] = task_data
    
    return task_datasets


def run_pretrain(config, args):
    """Run pretraining."""
    logger = logging.getLogger("pretrain")
    logger.info("Starting pretraining...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load datasets
    pretrain_config = config.get("pretrain", {})
    data_config = config.get("data", {})
    
    graph_dir = Path(data_config.get("graph_dir", "data/graphs/pretrain_train"))
    metadata_dir = Path(data_config.get("split_dir", "data/splits"))
    
    train_dataset, valid_dataset = load_pretrain_datasets(graph_dir, metadata_dir)
    
    # Build model
    model_config = config.get("model", {})
    encoder_type = model_config.get("encoder_type", "gcn")
    
    model = build_pretrain_model(
        task_names=train_dataset.task_names,
        encoder_type=encoder_type,
        input_features=model_config.get("input_features", 25),
        hidden_features=model_config.get("hidden_features", 256),
        num_layers=model_config.get("num_layers", 4),
        dropout=model_config.get("dropout", 0.1),
        regularize_encoder=model_config.get("regularize_encoder", 1e-4),
        regularize_contribution=model_config.get("regularize_contribution", 0.5),
        train_eps=model_config.get("train_eps", False),
    )
    
    logger.info(f"Built {encoder_type.upper()} model with {len(train_dataset.task_names)} tasks")
    
    # Create trainer
    out_dir = Path(args.output_dir) / "pretrain" / "checkpoints"
    trainer = PretrainTrainer(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        device=device,
        out_dir=str(out_dir)
    )
    
    # Train
    results = trainer.train(
        epochs=pretrain_config.get("epochs", 50),
        batch_size=pretrain_config.get("batch_size", 64),
        lr=pretrain_config.get("lr", 1e-3),
        weight_decay=pretrain_config.get("weight_decay", 1e-5),
        samples_per_task_per_epoch=pretrain_config.get("samples_per_task", None),
        grad_clip=pretrain_config.get("grad_clip", 1.0),
        lr_patience=pretrain_config.get("lr_patience", 5),
        early_stop_patience=pretrain_config.get("early_stop_patience", 15),
    )
    
    logger.info(f"Pretraining complete! Best AUROC: {results['best_val_auroc']:.4f}")
    
    return results, str(out_dir / "pretrained_encoder.pt")


def run_finetune(config, args, encoder_path=None):
    """Run finetuning."""
    logger = logging.getLogger("finetune")
    logger.info("Starting finetuning...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load config
    finetune_config = config.get("finetune", {})
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    
    # Get task names
    task_names = finetune_config.get("task_names", [
        "solubility_aqsoldb", "caco2_wang", "half_life_obach"
    ])
    
    # Load datasets
    graph_dir = Path(data_config.get("graph_dir", "data/graphs"))
    task_datasets = load_finetune_datasets(graph_dir, task_names)
    
    if not task_datasets:
        raise ValueError("No task datasets loaded!")
    
    # Build model
    encoder_type = model_config.get("encoder_type", "gcn")
    encoder_checkpoint = encoder_path or args.encoder
    
    model = build_finetune_model(
        task_names=list(task_datasets.keys()),
        encoder_checkpoint=encoder_checkpoint,
        encoder_type=encoder_type,
        input_features=model_config.get("input_features", 25),
        hidden_features=model_config.get("hidden_features", 256),
        num_layers=model_config.get("num_layers", 4),
        dropout=model_config.get("dropout", 0.1),
        freeze_encoder=finetune_config.get("freeze_encoder", False),
        regularize_encoder=model_config.get("regularize_encoder", 1e-4),
        regularize_contribution=model_config.get("regularize_contribution", 0.5),
        device=str(device),
        train_eps=model_config.get("train_eps", False),
    )
    
    if encoder_checkpoint:
        logger.info(f"Loaded encoder from {encoder_checkpoint}")
    else:
        logger.info("Training from scratch (no pretrained encoder)")
    
    # Create trainer
    out_dir = Path(args.output_dir) / "finetune" / "checkpoints"
    trainer = FinetuneTrainer(
        model=model,
        task_datasets=task_datasets,
        device=device,
        out_dir=str(out_dir)
    )
    
    # Train
    results = trainer.train(
        epochs=finetune_config.get("epochs", 120),
        batch_size=finetune_config.get("batch_size", 64),
        lr=finetune_config.get("lr", 3e-4),
        weight_decay=finetune_config.get("weight_decay", 1e-6),
        mse_weight=finetune_config.get("mse_weight", 1.0),
        grad_clip=finetune_config.get("grad_clip", 1.0),
        lr_patience=finetune_config.get("lr_patience", 8),
        early_stop_patience=finetune_config.get("early_stop_patience", 25),
        task_sampling=finetune_config.get("task_sampling", "proportional"),
        validate_batch_size=finetune_config.get("validate_batch_size", 128),
    )
    
    logger.info(f"Finetuning complete! Best Spearman: {results['best_avg_val_spearman']:.4f}")
    
    return results, model, task_datasets


def run_inference(config, args, model, task_datasets):
    """Run inference and extract explanations."""
    logger = logging.getLogger("inference")
    logger.info("Running inference and extracting explanations...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inference_config = config.get("inference", {})
    out_dir = Path(args.output_dir) / "finetune" / "inference"
    
    # Extract explanations
    explanations = extract_explanations_all_tasks(
        model=model,
        task_datasets=task_datasets,
        task_dataframes=None,
        output_dir=str(out_dir),
        batch_size=inference_config.get("batch_size", 8),
        device=device,
        splits=inference_config.get("splits", ["test"]),
    )
    
    # Visualize
    if inference_config.get("visualize", True):
        visualize_all_tasks(
            task_explanations=explanations,
            output_dir=str(out_dir),
            sample_size=inference_config.get("vis_samples", 10),
            normalize=True,
            cmap="RdBu_r",
        )
    
    logger.info(f"Inference complete! Results saved to {out_dir}")
    
    return explanations


def main():
    parser = argparse.ArgumentParser(description="SEAL-ADME Training")
    parser.add_argument("mode", choices=["pretrain", "finetune", "all"],
                       help="Training mode")
    parser.add_argument("--config", type=Path, default=Path("configs/model_config.yaml"))
    parser.add_argument("--encoder", type=str, default=None,
                       help="Path to pretrained encoder for finetuning")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    
    args = parser.parse_args()
    
    setup_logging(args.log_file, args.log_level)
    logger = logging.getLogger("main")
    
    # Load config
    if args.config.exists():
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        logger.warning(f"Config not found: {args.config}, using defaults")
        config = {}
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run based on mode
    encoder_path = args.encoder
    
    if args.mode in ["pretrain", "all"]:
        logger.info("=" * 60)
        logger.info("PRETRAINING")
        logger.info("=" * 60)
        pretrain_results, encoder_path = run_pretrain(config, args)
    
    if args.mode in ["finetune", "all"]:
        logger.info("=" * 60)
        logger.info("FINETUNING")
        logger.info("=" * 60)
        finetune_results, model, task_datasets = run_finetune(config, args, encoder_path)
        
        # Run inference
        logger.info("=" * 60)
        logger.info("INFERENCE")
        logger.info("=" * 60)
        run_inference(config, args, model, task_datasets)
    
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
