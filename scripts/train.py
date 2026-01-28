#!/usr/bin/env python3
"""
Training script for SEAL-ADME models.

Supports both pretraining on classification tasks and finetuning
on regression tasks.

Usage:
    # Pretraining
    python scripts/train.py --mode pretrain --config configs/model_config.yaml
    
    # Finetuning with pretrained encoder
    python scripts/train.py --mode finetune --encoder checkpoints/pretrained_encoder.pt
    
    # Finetuning from scratch
    python scripts/train.py --mode finetune --from-scratch
    
    # Both phases sequentially
    python scripts/train.py --mode both --config configs/model_config.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    GCNEncoder,
    GINEncoder,
    create_encoder,
    create_model,
    load_pretrained_encoder,
    MultiTaskModel,
    PretrainModel,
    FinetuneModel,
)
from src.training import (
    MultiTaskDataset,
    TaskGraphDataset,
    BalancedMultiTaskSampler,
    load_task_graphs,
    create_data_loader,
    PretrainTrainer,
    FinetuneTrainer,
    compute_regression_metrics,
)
from src.explanations import (
    extract_explanations,
    visualize_explanations,
    MoleculeExplanation,
)


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """Configure logging."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='w'))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers
    )


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def get_default_config() -> dict:
    """Return default configuration."""
    return {
        "model": {
            "input_dim": 25,
            "hidden_dim": 256,
            "num_layers": 4,
            "dropout": 0.1,
            "encoder_type": "gcn",
            "reg_encoder": 1e-4,
            "reg_contribution": 0.5,
        },
        "pretrain": {
            "epochs": 50,
            "batch_size": 64,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "samples_per_task": None,
            "grad_clip": 1.0,
            "lr_patience": 5,
            "early_stop_patience": 15,
        },
        "finetune": {
            "epochs": 120,
            "batch_size": 64,
            "lr": 3e-4,
            "weight_decay": 1e-6,
            "mse_weight": 1.0,
            "grad_clip": 1.0,
            "lr_patience": 8,
            "early_stop_patience": 25,
            "task_sampling": "round_robin",
        },
        "tasks": {
            "finetune": [
                ("solubility_aqsoldb", "solubility_aqsoldb"),
                ("caco2", "caco2"),
                ("half_life_obach", "half_life_obach"),
                ("AKA", "AKA"),
                ("AKB", "AKB"),
            ]
        },
        "inference": {
            "batch_size": 128,
            "extract_explanations": True,
            "visualize": True,
            "visualization_samples": 10,
        }
    }


def run_pretraining(args, config: dict) -> dict:
    """
    Run pretraining phase on classification tasks.
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
        
    Returns:
        Training results dictionary
    """
    import pandas as pd
    
    logger = logging.getLogger("pretrain")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir) / "pretrain"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.train_meta or not args.valid_meta:
        raise ValueError("--train-meta and --valid-meta required for pretraining")
    
    train_meta = pd.read_parquet(args.train_meta)
    valid_meta = pd.read_parquet(args.valid_meta)
    
    train_dataset = MultiTaskDataset(graph_dir, train_meta)
    valid_dataset = MultiTaskDataset(graph_dir, valid_meta)
    
    logger.info(f"Train tasks: {train_dataset.task_names}")
    logger.info(f"Valid tasks: {valid_dataset.task_names}")
    
    model_config = config.get("model", {})
    
    encoder = create_encoder(
        encoder_type=model_config.get("encoder_type", "gcn"),
        input_dim=model_config.get("input_dim", 25),
        hidden_dim=model_config.get("hidden_dim", 256),
        num_layers=model_config.get("num_layers", 4),
        dropout=model_config.get("dropout", 0.1),
        train_eps=model_config.get("train_eps", False),
    )
    
    model = PretrainModel(
        encoder=encoder,
        task_names=train_dataset.task_names,
        freeze_encoder=False,
        reg_encoder=model_config.get("reg_encoder", 1e-4),
        reg_contribution=model_config.get("reg_contribution", 0.5),
    )
    
    train_config = config.get("pretrain", {})
    
    trainer = PretrainTrainer(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        device=device,
        output_dir=output_dir / "checkpoints",
    )
    
    results = trainer.train(
        epochs=train_config.get("epochs", 50),
        batch_size=train_config.get("batch_size", 64),
        lr=train_config.get("lr", 1e-3),
        weight_decay=train_config.get("weight_decay", 1e-5),
        samples_per_task=train_config.get("samples_per_task"),
        grad_clip=train_config.get("grad_clip", 1.0),
        lr_patience=train_config.get("lr_patience", 5),
        early_stop_patience=train_config.get("early_stop_patience", 15),
    )
    
    logger.info(f"Pretraining complete. Best val AUROC: {results['best_val_auroc']:.4f}")
    
    return results


def run_finetuning(args, config: dict) -> dict:
    """
    Run finetuning phase on regression tasks.
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
        
    Returns:
        Training results dictionary
    """
    logger = logging.getLogger("finetune")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    output_dir = Path(args.output_dir) / "finetune"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tasks_config = config.get("tasks", {}).get("finetune", [])
    if isinstance(tasks_config, list) and tasks_config:
        if isinstance(tasks_config[0], str):
            task_configs = [(t, t) for t in tasks_config]
        else:
            task_configs = tasks_config
    else:
        task_configs = [
            ("solubility_aqsoldb", "solubility_aqsoldb"),
            ("caco2", "caco2"),
            ("half_life_obach", "half_life_obach"),
            ("AKA", "AKA"),
            ("AKB", "AKB"),
        ]
    
    task_datasets = load_task_graphs(
        base_dir=Path(args.graph_dir),
        task_configs=task_configs,
    )
    
    if not task_datasets:
        raise ValueError("No task datasets found")
    
    logger.info(f"Loaded {len(task_datasets)} tasks: {list(task_datasets.keys())}")
    
    model_config = config.get("model", {})
    
    encoder = create_encoder(
        encoder_type=model_config.get("encoder_type", "gcn"),
        input_dim=model_config.get("input_dim", 25),
        hidden_dim=model_config.get("hidden_dim", 256),
        num_layers=model_config.get("num_layers", 4),
        dropout=model_config.get("dropout", 0.1),
        train_eps=model_config.get("train_eps", False),
    )
    
    if args.encoder_path and Path(args.encoder_path).exists():
        logger.info(f"Loading pretrained encoder from {args.encoder_path}")
        state_dict = torch.load(args.encoder_path, map_location=device)
        if isinstance(state_dict, dict) and 'encoder_state' in state_dict:
            state_dict = state_dict['encoder_state']
        encoder.load_state_dict(state_dict)
    elif not args.from_scratch:
        logger.warning("No pretrained encoder specified, training from scratch")
    
    model = FinetuneModel(
        encoder=encoder,
        task_names=list(task_datasets.keys()),
        freeze_encoder=args.freeze_encoder,
        reg_encoder=model_config.get("reg_encoder", 1e-4),
        reg_contribution=model_config.get("reg_contribution", 0.5),
    )
    
    train_config = config.get("finetune", {})
    
    trainer = FinetuneTrainer(
        model=model,
        task_datasets=task_datasets,
        device=device,
        output_dir=output_dir / "checkpoints",
    )
    
    results = trainer.train(
        epochs=train_config.get("epochs", 120),
        batch_size=train_config.get("batch_size", 64),
        lr=train_config.get("lr", 3e-4),
        weight_decay=train_config.get("weight_decay", 1e-6),
        mse_weight=train_config.get("mse_weight", 1.0),
        grad_clip=train_config.get("grad_clip", 1.0),
        lr_patience=train_config.get("lr_patience", 8),
        early_stop_patience=train_config.get("early_stop_patience", 25),
        task_sampling=train_config.get("task_sampling", "round_robin"),
    )
    
    logger.info(
        f"Finetuning complete. "
        f"Best val Spearman: {results['best_avg_val_spearman']:.4f}"
    )
    
    inference_config = config.get("inference", {})
    if args.extract_explanations or inference_config.get("extract_explanations", False):
        logger.info("Extracting explanations...")
        
        for task_name in task_datasets.keys():
            test_graphs = task_datasets[task_name].test
            if len(test_graphs) == 0:
                logger.info(f"Skipping {task_name} (no test data)")
                continue
            
            expl_dir = output_dir / task_name
            expl_dir.mkdir(parents=True, exist_ok=True)
            
            explanations = extract_explanations(
                model=model,
                task_name=task_name,
                graphs=test_graphs,
                device=device,
                save_path=expl_dir / "explanations_test.pt",
            )
            
            if args.visualize or inference_config.get("visualize", False):
                n_samples = inference_config.get("visualization_samples", 10)
                visualize_explanations(
                    task_name=task_name,
                    explanations=explanations,
                    output_dir=expl_dir / "visualizations",
                    sample_size=n_samples,
                    display_inline=False,
                )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="SEAL-ADME Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pretrain on classification tasks
  %(prog)s --mode pretrain --graph-dir data/graphs --train-meta data/train.parquet --valid-meta data/valid.parquet
  
  # Finetune with pretrained encoder
  %(prog)s --mode finetune --graph-dir data/graphs --encoder checkpoints/pretrained_encoder.pt
  
  # Finetune from scratch
  %(prog)s --mode finetune --graph-dir data/graphs --from-scratch
  
  # Run both phases
  %(prog)s --mode both --graph-dir data/graphs --train-meta data/train.parquet --valid-meta data/valid.parquet
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["pretrain", "finetune", "both"],
        default="finetune",
        help="Training mode (default: finetune)"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model_config.yaml"),
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--graph-dir",
        type=Path,
        required=True,
        help="Directory containing graph .pt files"
    )
    parser.add_argument(
        "--train-meta",
        type=Path,
        help="Training metadata parquet (for pretraining)"
    )
    parser.add_argument(
        "--valid-meta",
        type=Path,
        help="Validation metadata parquet (for pretraining)"
    )
    
    parser.add_argument(
        "--encoder-path",
        type=Path,
        help="Path to pretrained encoder weights"
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Train encoder from scratch (no pretraining)"
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze encoder during finetuning"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results"),
        help="Output directory (default: ./results)"
    )
    
    parser.add_argument(
        "--extract-explanations",
        action="store_true",
        help="Extract explanations after training"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations"
    )
    
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Log file path (optional)"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger("main")
    
    config = get_default_config()
    if args.config.exists():
        user_config = load_config(args.config)
        for key, value in user_config.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        logger.info(f"Loaded configuration from {args.config}")
    else:
        logger.info("Using default configuration")
    
    logger.info("=" * 70)
    logger.info("SEAL-ADME Training Pipeline")
    logger.info("=" * 70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Graph directory: {args.graph_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    results = {}
    
    try:
        if args.mode in ["pretrain", "both"]:
            logger.info("")
            logger.info("=" * 70)
            logger.info("PRETRAINING PHASE")
            logger.info("=" * 70)
            
            results["pretrain"] = run_pretraining(args, config)
            
            if args.mode == "both":
                pretrain_dir = args.output_dir / "pretrain" / "checkpoints"
                args.encoder_path = pretrain_dir / "pretrained_encoder.pt"
                logger.info(f"Using pretrained encoder: {args.encoder_path}")
        
        if args.mode in ["finetune", "both"]:
            logger.info("")
            logger.info("=" * 70)
            logger.info("FINETUNING PHASE")
            logger.info("=" * 70)
            
            results["finetune"] = run_finetuning(args, config)
        
        results_path = args.output_dir / "training_summary.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("Training complete!")
        logger.info(f"Results saved to: {results_path}")
        logger.info("=" * 70)
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
