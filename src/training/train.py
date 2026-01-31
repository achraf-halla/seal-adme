#!/usr/bin/env python3
"""
Training script for SEAL-ADME.

Supports:
- pretrain: Multi-task classification pretraining
- finetune: Multi-task regression finetuning
- all: Full pipeline (pretrain -> finetune)

Usage:
    python scripts/train.py pretrain --data-dir data --output-dir outputs
    python scripts/train.py finetune --data-dir data --encoder outputs/pretrain/pretrained_encoder.pt
    python scripts/train.py all --data-dir data --output-dir outputs
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import PRETRAIN_TASKS, FINETUNE_TASKS
from src.models import build_pretrain_model, build_finetune_model
from src.training import (
    load_pretrain_dataset,
    load_finetune_datasets,
    PretrainTrainer,
    MultiTaskFinetuneTrainer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s"
)
logger = logging.getLogger(__name__)


def run_pretrain(args) -> Path:
    """Run pretraining."""
    logger.info("=" * 60)
    logger.info("PRETRAINING")
    logger.info("=" * 60)
    
    # Load data
    logger.info(f"Loading pretrain data from {args.data_dir}/graphs/pretrain")
    dataset = load_pretrain_dataset(args.data_dir / "graphs")
    
    logger.info(f"Tasks: {dataset.task_names}")
    logger.info(f"Total graphs: {len(dataset)}")
    
    # Build model
    model = build_pretrain_model(
        task_names=dataset.task_names,
        encoder_type=args.encoder_type,
        input_features=args.input_features,
        hidden_features=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        regularize_encoder=args.reg_encoder,
        regularize_contribution=args.reg_contribution
    )
    
    logger.info(f"Model: {args.encoder_type.upper()} encoder, {args.hidden_dim} hidden dim")
    
    # Output directory
    output_dir = args.output_dir / "pretrain"
    
    # Trainer
    trainer = PretrainTrainer(
        model=model,
        train_dataset=dataset,
        valid_dataset=dataset,  # Using same for pretrain (no split)
        device=args.device,
        out_dir=output_dir
    )
    
    # Train
    results = trainer.train(
        epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        lr_patience=args.lr_patience,
        early_stop_patience=args.early_stop_patience
    )
    
    encoder_path = output_dir / "pretrained_encoder.pt"
    logger.info(f"Pretrained encoder saved to: {encoder_path}")
    
    return encoder_path


def run_finetune(args, encoder_path: Path = None) -> dict:
    """Run finetuning."""
    logger.info("=" * 60)
    logger.info("FINETUNING")
    logger.info("=" * 60)
    
    # Load data
    logger.info(f"Loading finetune data from {args.data_dir}/graphs/finetune")
    datasets = load_finetune_datasets(args.data_dir / "graphs")
    
    # Load normalization stats
    norm_stats = {}
    for task_name in datasets.keys():
        stats_path = args.data_dir / "graphs" / "finetune" / task_name / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
                norm_stats[task_name] = stats.get('normalization', {'mean': 0.0, 'std': 1.0})
        else:
            norm_stats[task_name] = {'mean': 0.0, 'std': 1.0}
    
    logger.info(f"Tasks: {list(datasets.keys())}")
    for task, stats in norm_stats.items():
        logger.info(f"  {task}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
    
    # Use provided encoder or from args
    if encoder_path is None and args.encoder is not None:
        encoder_path = Path(args.encoder)
    
    if encoder_path and encoder_path.exists():
        logger.info(f"Using pretrained encoder: {encoder_path}")
    else:
        logger.info("Training from scratch (no pretrained encoder)")
        encoder_path = None
    
    # Build model
    model = build_finetune_model(
        task_names=list(datasets.keys()),
        encoder_checkpoint=str(encoder_path) if encoder_path else None,
        encoder_type=args.encoder_type,
        input_features=args.input_features,
        hidden_features=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        freeze_encoder=args.freeze_encoder,
        regularize_encoder=args.reg_encoder,
        regularize_contribution=args.reg_contribution,
        device=args.device
    )
    
    # Output directory
    output_dir = args.output_dir / "finetune"
    
    # Trainer
    trainer = MultiTaskFinetuneTrainer(
        model=model,
        task_datasets=datasets,
        norm_stats=norm_stats,
        device=args.device,
        out_dir=output_dir
    )
    
    # Train
    results = trainer.train(
        epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        lr=args.finetune_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        lr_patience=args.lr_patience,
        early_stop_patience=args.finetune_patience,
        task_sampling=args.task_sampling
    )
    
    # Print summary
    logger.info("=" * 60)
    logger.info("FINETUNING RESULTS SUMMARY")
    logger.info("=" * 60)
    for task in datasets.keys():
        test_sp = results['final_metrics'][task]['test']['spearman']
        test_rmse = results['final_metrics'][task]['test']['rmse_denorm']
        logger.info(f"{task:25s}: Spearman={test_sp:.4f}, RMSE={test_rmse:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="SEAL-ADME Training")
    parser.add_argument(
        "mode",
        choices=["pretrain", "finetune", "all"],
        help="Training mode"
    )
    
    # Data arguments
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    
    # Model arguments
    parser.add_argument("--encoder-type", type=str, default="gcn", choices=["gcn", "gin"])
    parser.add_argument("--input-features", type=int, default=25)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Regularization
    parser.add_argument("--reg-encoder", type=float, default=1e-4, help="L1 on inter-fragment weights")
    parser.add_argument("--reg-contribution", type=float, default=0.5, help="L1 on fragment contributions")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3, help="Pretrain learning rate")
    parser.add_argument("--finetune-lr", type=float, default=3e-4, help="Finetune learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    
    # Epochs
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--finetune-epochs", type=int, default=120)
    
    # Early stopping / LR scheduling
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=15, help="Pretrain early stop")
    parser.add_argument("--finetune-patience", type=int, default=25, help="Finetune early stop")
    
    # Finetune specific
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--task-sampling", type=str, default="round_robin", 
                        choices=["round_robin", "proportional"])
    
    # Pretrained encoder (for finetune mode)
    parser.add_argument("--encoder", type=Path, default=None, help="Path to pretrained encoder")
    
    # Device
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Device: {args.device}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run
    if args.mode == "pretrain":
        run_pretrain(args)
    
    elif args.mode == "finetune":
        run_finetune(args)
    
    elif args.mode == "all":
        encoder_path = run_pretrain(args)
        run_finetune(args, encoder_path=encoder_path)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
