#!/usr/bin/env python3
"""
Inference script for SEAL-ADME models.

Run predictions on new molecules and extract explanations.

Usage:
    # Run inference with all tasks
    python scripts/inference.py \
        --model-path results/finetune/checkpoints/final_model.pt \
        --graph-dir data/graphs/test \
        --output-dir results/inference
    
    # Run specific task with explanations
    python scripts/inference.py \
        --model-path results/finetune/checkpoints/final_model.pt \
        --graph-dir data/graphs/generated \
        --output-dir results/inference \
        --tasks AKA AKB \
        --extract-explanations \
        --visualize
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import build_model
from src.evaluation import (
    InferenceRunner,
    load_graphs_from_directory,
    visualize_explanations,
    load_explanations,
)


def setup_logging(log_level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def load_model(
    model_path: Path,
    config_path: Path = None,
    device: str = 'cpu'
) -> torch.nn.Module:
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        config_path: Optional config file for model architecture
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    logger = logging.getLogger("load_model")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get task names from checkpoint
    state_dict = checkpoint.get('model_state', checkpoint)
    task_names = []
    
    for key in state_dict.keys():
        if key.startswith('task_heads.'):
            task_name = key.split('.')[1]
            if task_name not in task_names:
                task_names.append(task_name)
    
    logger.info(f"Found tasks: {task_names}")
    
    # Load config if provided
    model_config = {}
    if config_path and config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
            model_config = config.get('model', {})
    
    # Build model
    model = build_model(
        task_names=task_names,
        input_features=model_config.get('input_features', 25),
        hidden_features=model_config.get('hidden_features', 256),
        num_layers=model_config.get('num_layers', 4),
        dropout=model_config.get('dropout', 0.1),
        conv_type=model_config.get('conv_type', 'seal'),
        model_type='regression',
        device=device
    )
    
    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Run SEAL-ADME inference")
    
    # Model
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Model config file (for architecture params)"
    )
    
    # Data
    parser.add_argument(
        "--graph-dir",
        type=Path,
        required=True,
        help="Directory containing graph .pt files"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="test",
        help="Name for the dataset (used in output filenames)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pt",
        help="Glob pattern for graph files"
    )
    
    # Tasks
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Specific tasks to run (default: all)"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./inference_results"),
        help="Output directory for results"
    )
    
    # Options
    parser.add_argument(
        "--extract-explanations",
        action="store_true",
        help="Extract fragment-level explanations"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization images"
    )
    parser.add_argument(
        "--vis-samples",
        type=int,
        default=10,
        help="Number of samples to visualize"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for inference"
    )
    
    # Misc
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device (cuda/cpu, default: auto)"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    logger = logging.getLogger("main")
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    model = load_model(args.model_path, args.config, device)
    
    # Filter tasks if specified
    if args.tasks:
        # Only keep specified tasks
        tasks_to_run = [t for t in args.tasks if t in model.task_names]
        if not tasks_to_run:
            logger.error(f"None of specified tasks found in model. Available: {model.task_names}")
            return
        logger.info(f"Running tasks: {tasks_to_run}")
    else:
        tasks_to_run = model.task_names
    
    # Load graphs
    logger.info(f"Loading graphs from {args.graph_dir}...")
    graphs = load_graphs_from_directory(args.graph_dir, args.pattern)
    
    if not graphs:
        logger.error("No graphs loaded!")
        return
    
    # Create runner
    runner = InferenceRunner(
        model=model,
        output_dir=args.output_dir,
        device=device
    )
    
    # Run inference
    logger.info("=" * 60)
    logger.info("RUNNING INFERENCE")
    logger.info("=" * 60)
    
    all_results = {}
    
    for task_name in tasks_to_run:
        results = runner.run_task(
            task_name=task_name,
            graphs=graphs,
            dataset_name=args.dataset_name,
            extract_explanations=args.extract_explanations,
            batch_size=args.batch_size
        )
        all_results[task_name] = results
        
        # Visualize if requested
        if args.visualize and 'explanations' in results:
            logger.info(f"Generating visualizations for {task_name}...")
            vis_dir = args.output_dir / task_name / f"visualizations_{args.dataset_name}"
            visualize_explanations(
                results['explanations'],
                output_dir=vis_dir,
                task_name=task_name,
                sample_size=args.vis_samples
            )
    
    # Summary
    logger.info("=" * 60)
    logger.info("INFERENCE COMPLETE")
    logger.info("=" * 60)
    
    for task_name, results in all_results.items():
        n_samples = len(results.get('predictions', []))
        logger.info(f"  {task_name}: {n_samples} predictions")
        
        if 'metrics' in results:
            m = results['metrics']
            logger.info(f"    Spearman: {m['spearman']:.4f}, RMSE: {m['rmse']:.4f}")
    
    logger.info(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
