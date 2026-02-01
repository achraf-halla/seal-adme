#!/usr/bin/env python3
"""
Inference script for SEAL-ADME.

Runs inference on trained models and extracts explanations.

Usage:
    python scripts/inference.py --checkpoint outputs/finetune/best_checkpoint.pt \
        --data-dir data --output-dir inference_results
    
    # With specific task
    python scripts/inference.py --checkpoint outputs/finetune/Caco2_Wang/checkpoints/best_model.pt \
        --task Caco2_Wang --data-dir data --output-dir inference_results
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import FINETUNE_TASKS
from src.models import build_finetune_model
from src.training import load_finetune_datasets
from src.evaluation import (
    run_inference_and_save,
    visualize_task_explanations,
    extract_explanations_for_task,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s"
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, task_names: list, device: str, args):
    """Load model from checkpoint."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Build model
    model = build_finetune_model(
        task_names=task_names,
        encoder_type=args.encoder_type,
        input_features=args.input_features,
        hidden_features=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=device
    )
    
    # Load state dict
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    logger.info(f"Loaded model for tasks: {task_names}")
    
    return model


def run_inference(args):
    """Run inference on test data."""
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine tasks
    if args.task:
        task_names = [args.task]
    else:
        task_names = FINETUNE_TASKS
    
    # Load model
    model = load_model(args.checkpoint, task_names, device, args)
    
    # Load datasets
    logger.info(f"Loading data from {args.data_dir}/graphs/finetune")
    datasets = load_finetune_datasets(args.data_dir / "graphs")
    
    # Load normalization stats
    norm_stats = {}
    for task_name in task_names:
        stats_path = args.data_dir / "graphs" / "finetune" / task_name / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
                norm_stats[task_name] = stats.get('normalization', {'mean': 0.0, 'std': 1.0})
        else:
            norm_stats[task_name] = {'mean': 0.0, 'std': 1.0}
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    logger.info("=" * 60)
    logger.info("RUNNING INFERENCE")
    logger.info("=" * 60)
    
    for task_name in task_names:
        if task_name not in datasets:
            logger.warning(f"Task {task_name} not found in datasets")
            continue
        
        logger.info(f"\nTask: {task_name}")
        logger.info("-" * 40)
        
        task_dir = output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        
        ds = datasets[task_name]
        
        # Run on each split
        for split_name, graphs in [('train', ds.train), ('valid', ds.valid), ('test', ds.test)]:
            if not graphs:
                continue
            
            if not args.all_splits and split_name != 'test':
                continue
            
            logger.info(f"  Processing {split_name} ({len(graphs)} molecules)")
            
            results = run_inference_and_save(
                model=model,
                task_name=task_name,
                graphs=graphs,
                dataset_name=split_name,
                output_dir=str(task_dir),
                norm_stats=norm_stats.get(task_name),
                extract_explanations=args.extract_explanations,
                batch_size=args.batch_size,
                device=device
            )
            
            if results is not None:
                all_results[(task_name, split_name)] = results
                
                # Visualize if requested
                if args.visualize and 'explanations' in results:
                    vis_dir = task_dir / f"visualizations_{split_name}"
                    visualize_task_explanations(
                        task_name=task_name,
                        explanations=results['explanations'],
                        output_dir=str(vis_dir),
                        sample_size=args.vis_samples,
                        display_inline=False,
                        normalize=True
                    )
    
    # Save summary
    summary = {}
    for (task_name, split_name), results in all_results.items():
        if 'metrics' in results:
            if task_name not in summary:
                summary[task_name] = {}
            summary[task_name][split_name] = results['metrics']
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("INFERENCE SUMMARY")
    logger.info("=" * 60)
    
    for task_name, splits in summary.items():
        logger.info(f"\n{task_name}:")
        for split_name, metrics in splits.items():
            logger.info(
                f"  {split_name:6s}: Spearman={metrics['spearman']:.4f}, "
                f"RMSE={metrics['rmse_denorm']:.4f}"
            )
    
    logger.info(f"\nResults saved to: {output_dir}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="SEAL-ADME Inference")
    
    # Required arguments
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    
    # Data arguments
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("inference_results"))
    
    # Model arguments (must match training)
    parser.add_argument("--encoder-type", type=str, default="gcn", choices=["gcn", "gin"])
    parser.add_argument("--input-features", type=int, default=25)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Inference arguments
    parser.add_argument("--task", type=str, default=None, help="Specific task (default: all)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--all-splits", action="store_true", help="Run on train/valid/test (default: test only)")
    
    # Explanation arguments
    parser.add_argument("--extract-explanations", action="store_true", default=True)
    parser.add_argument("--no-explanations", dest="extract_explanations", action="store_false")
    
    # Visualization arguments
    parser.add_argument("--visualize", action="store_true", help="Generate molecule visualizations")
    parser.add_argument("--vis-samples", type=int, default=10, help="Number of molecules to visualize")
    
    # Device
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    run_inference(args)


if __name__ == "__main__":
    main()
