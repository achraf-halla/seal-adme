#!/usr/bin/env python3
"""
Training script for SEAL-ADME.

Supports:
- pretrain: Multi-task classification pretraining
- finetune: Single-task regression finetuning
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
    FinetuneTrainer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s"
)
logger = logging.getLogger(__name__)


def run_pretrain(args):
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
        dropout=args.dropout
    )
    
    logger.info(f"Model: {args.encoder_type.upper()} encoder, {args.hidden_dim} hidden")
    
    # Output directory
    output_dir = args.output_dir / "pretrain"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Trainer
    trainer = PretrainTrainer(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        checkpoint_dir=output_dir / "checkpoints"
    )
    
    # Train
    history = trainer.train(
        epochs=args.pretrain_epochs,
        log_interval=1,
        save_interval=10
    )
    
    # Save history
    with open(output_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    encoder_path = output_dir / "checkpoints" / "pretrained_encoder.pt"
    logger.info(f"Pretrained encoder saved to: {encoder_path}")
    
    return encoder_path


def run_finetune(args, encoder_path: Path = None):
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
    logger.info(f"Normalization stats: {norm_stats}")
    
    # Use provided encoder or from pretrain
    if encoder_path is None and args.encoder is not None:
        encoder_path = Path(args.encoder)
    
    if encoder_path and encoder_path.exists():
        logger.info(f"Using pretrained encoder: {encoder_path}")
    else:
        logger.info("Training from scratch (no pretrained encoder)")
        encoder_path = None
    
    results = {}
    
    for task_name, dataset in datasets.items():
        logger.info("-" * 40)
        logger.info(f"Training: {task_name}")
        logger.info("-" * 40)
        
        # Build model
        model = build_finetune_model(
            task_names=[task_name],
            encoder_checkpoint=str(encoder_path) if encoder_path else None,
            encoder_type=args.encoder_type,
            input_features=args.input_features,
            hidden_features=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            freeze_encoder=args.freeze_encoder,
            device=args.device
        )
        
        # Output directory
        task_output = args.output_dir / "finetune" / task_name
        task_output.mkdir(parents=True, exist_ok=True)
        
        # Trainer
        trainer = FinetuneTrainer(
            model=model,
            task_name=task_name,
            train_graphs=dataset.train,
            valid_graphs=dataset.valid,
            test_graphs=dataset.test,
            batch_size=args.batch_size,
            lr=args.finetune_lr,
            device=args.device,
            checkpoint_dir=task_output / "checkpoints",
            norm_stats=norm_stats.get(task_name, {'mean': 0.0, 'std': 1.0})
        )
        
        # Train
        history = trainer.train(
            epochs=args.finetune_epochs,
            patience=args.patience
        )
        
        results[task_name] = {
            'test_rmse_denorm': history.get('test_rmse_denorm'),
            'best_epoch': history.get('best_epoch'),
            'best_valid_rmse': history.get('best_valid_rmse')
        }
    
    # Save summary
    summary_path = args.output_dir / "finetune" / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("FINETUNING RESULTS")
    logger.info("=" * 60)
    for task, res in results.items():
        logger.info(f"{task}: Test RMSE = {res['test_rmse_denorm']:.4f}")
    
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
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3, help="Pretrain learning rate")
    parser.add_argument("--finetune-lr", type=float, default=3e-4, help="Finetune learning rate")
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--finetune-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--freeze-encoder", action="store_true")
    
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
