#!/usr/bin/env python3
"""
Training script for SEAL-ADME models.

Supports both pretraining on classification tasks and fine-tuning
on regression tasks.

Usage:
    # Pretraining
    python scripts/train.py --mode pretrain --config configs/model_config.yaml
    
    # Fine-tuning
    python scripts/train.py --mode finetune --encoder checkpoints/pretrained_encoder.pt
    
    # Fine-tuning without pretraining
    python scripts/train.py --mode finetune --from-scratch
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    FragmentAwareEncoder,
    PretrainModel,
    RegressionModel,
    build_model
)
from src.training import (
    MultiTaskDataset,
    PretrainTrainer,
    RegressionTrainer,
    load_task_datasets,
)
from src.evaluation import (
    extract_explanations,
    save_explanations,
    visualize_explanations,
)


def setup_logging(log_level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def run_pretraining(args, config: dict):
    """Run pretraining phase."""
    import pandas as pd
    
    logger = logging.getLogger("pretrain")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load pretrain metadata
    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir) / "pretrain"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_meta = pd.read_parquet(args.train_meta)
    valid_meta = pd.read_parquet(args.valid_meta)
    
    train_dataset = MultiTaskDataset(graph_dir, train_meta)
    valid_dataset = MultiTaskDataset(graph_dir, valid_meta)
    
    # Model config
    model_config = config.get('model', {})
    
    # Create encoder
    encoder = FragmentAwareEncoder(
        input_features=model_config.get('input_features', 25),
        hidden_features=model_config.get('hidden_features', 256),
        num_layers=model_config.get('num_layers', 4),
        dropout=model_config.get('dropout', 0.1),
        conv_type=model_config.get('conv_type', 'seal')
    )
    
    # Create model
    model = PretrainModel(
        encoder=encoder,
        task_names=train_dataset.task_names,
        regularize_encoder=model_config.get('regularize_encoder', 1e-4),
        regularize_contribution=model_config.get('regularize_contribution', 0.5)
    )
    
    # Training config
    train_config = config.get('pretrain', {})
    
    trainer = PretrainTrainer(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        device=device,
        output_dir=output_dir / "checkpoints"
    )
    
    results = trainer.train(
        epochs=train_config.get('epochs', 50),
        batch_size=train_config.get('batch_size', 64),
        lr=train_config.get('lr', 1e-3),
        weight_decay=train_config.get('weight_decay', 1e-5),
        samples_per_task=train_config.get('samples_per_task'),
        grad_clip=train_config.get('grad_clip', 1.0),
        lr_patience=train_config.get('lr_patience', 5),
        early_stop_patience=train_config.get('early_stop_patience', 15)
    )
    
    logger.info(f"Pretraining complete! Best val AUROC: {results['best_val_auroc']:.4f}")
    return results


def run_finetuning(args, config: dict):
    """Run fine-tuning phase."""
    logger = logging.getLogger("finetune")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    output_dir = Path(args.output_dir) / "finetune"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load task datasets
    task_datasets = load_task_datasets(
        base_dir=Path(args.graph_dir),
        task_configs=None  # Uses default tasks
    )
    
    if not task_datasets:
        raise ValueError("No task datasets found")
    
    # Model config
    model_config = config.get('model', {})
    
    # Create or load encoder
    encoder = FragmentAwareEncoder(
        input_features=model_config.get('input_features', 25),
        hidden_features=model_config.get('hidden_features', 256),
        num_layers=model_config.get('num_layers', 4),
        dropout=model_config.get('dropout', 0.1),
        conv_type=model_config.get('conv_type', 'seal')
    )
    
    # Load pretrained weights if available
    if args.encoder_path and Path(args.encoder_path).exists():
        logger.info(f"Loading pretrained encoder from {args.encoder_path}")
        encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
    elif not args.from_scratch:
        logger.warning("No pretrained encoder specified, training from scratch")
    
    # Create model
    model = RegressionModel(
        encoder=encoder,
        task_names=list(task_datasets.keys()),
        freeze_encoder=args.freeze_encoder,
        regularize_encoder=model_config.get('regularize_encoder', 1e-4),
        regularize_contribution=model_config.get('regularize_contribution', 0.5)
    )
    
    # Training config
    train_config = config.get('finetune', {})
    
    trainer = RegressionTrainer(
        model=model,
        task_datasets=task_datasets,
        device=device,
        output_dir=output_dir / "checkpoints"
    )
    
    results = trainer.train(
        epochs=train_config.get('epochs', 120),
        batch_size=train_config.get('batch_size', 64),
        lr=train_config.get('lr', 3e-4),
        weight_decay=train_config.get('weight_decay', 1e-6),
        mse_weight=train_config.get('mse_weight', 1.0),
        grad_clip=train_config.get('grad_clip', 1.0),
        lr_patience=train_config.get('lr_patience', 8),
        early_stop_patience=train_config.get('early_stop_patience', 25),
        task_sampling=train_config.get('task_sampling', 'round_robin')
    )
    
    logger.info(
        f"Fine-tuning complete! "
        f"Best val Spearman: {results['best_avg_val_spearman']:.4f}"
    )
    
    # Extract explanations if requested
    if args.extract_explanations:
        logger.info("Extracting explanations...")
        
        for task_name in task_datasets.keys():
            test_graphs = task_datasets[task_name].test
            if len(test_graphs) == 0:
                continue
            
            explanations = extract_explanations(
                model=model,
                task_name=task_name,
                graphs=test_graphs,
                device=device
            )
            
            expl_dir = output_dir / task_name
            expl_dir.mkdir(parents=True, exist_ok=True)
            
            save_explanations(
                explanations,
                expl_dir / "explanations_test.pt",
                task_name
            )
            
            if args.visualize:
                visualize_explanations(
                    explanations,
                    output_dir=expl_dir / "visualizations",
                    task_name=task_name,
                    sample_size=10
                )
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train SEAL-ADME models")
    
    # Mode
    parser.add_argument(
        "--mode",
        choices=["pretrain", "finetune", "both"],
        default="finetune",
        help="Training mode"
    )
    
    # Config
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model_config.yaml"),
        help="Config file path"
    )
    
    # Data paths
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
    
    # Model
    parser.add_argument(
        "--encoder-path",
        type=Path,
        help="Path to pretrained encoder"
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Train from scratch (no pretraining)"
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze encoder during fine-tuning"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results"),
        help="Output directory"
    )
    
    # Evaluation
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
    
    # Misc
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    logger = logging.getLogger("main")
    
    # Load config
    config = load_config(args.config)
    
    logger.info("=" * 80)
    logger.info("SEAL-ADME Training Pipeline")
    logger.info("=" * 80)
    
    if args.mode in ["pretrain", "both"]:
        logger.info("\n" + "=" * 40)
        logger.info("PRETRAINING PHASE")
        logger.info("=" * 40 + "\n")
        
        run_pretraining(args, config)
        
        # Update encoder path for fine-tuning
        if args.mode == "both":
            args.encoder_path = args.output_dir / "pretrain/checkpoints/pretrained_encoder.pt"
    
    if args.mode in ["finetune", "both"]:
        logger.info("\n" + "=" * 40)
        logger.info("FINE-TUNING PHASE")
        logger.info("=" * 40 + "\n")
        
        run_finetuning(args, config)
    
    logger.info("\n" + "=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
