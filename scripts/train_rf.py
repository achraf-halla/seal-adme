#!/usr/bin/env python3
"""
Train Random Forest baseline models.

Usage:
    # Train on a single task
    python scripts/train_rf.py \
        --task caco2 \
        --data-dir data/features \
        --output-dir results/rf
    
    # Train on all tasks
    python scripts/train_rf.py \
        --all-tasks \
        --data-dir data/features \
        --output-dir results/rf
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import RandomForestBaseline, train_rf_baseline


def setup_logging(log_level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


# Default tasks
DEFAULT_TASKS = [
    'solubility_aqsoldb',
    'caco2',
    'half_life_obach',
    'AKA',
    'AKB'
]


def find_task_files(
    data_dir: Path,
    task_name: str
) -> tuple:
    """
    Find train/valid/test parquet files for a task.
    
    Returns:
        Tuple of (train_path, valid_path, test_path)
    """
    data_dir = Path(data_dir)
    
    # Try common patterns
    patterns = [
        # Pattern 1: task_name/task_name_split.parquet
        (
            data_dir / task_name / f"{task_name}_train.parquet",
            data_dir / task_name / f"{task_name}_valid.parquet",
            data_dir / task_name / f"{task_name}_test.parquet"
        ),
        # Pattern 2: task_name_split.parquet
        (
            data_dir / f"{task_name}_train.parquet",
            data_dir / f"{task_name}_valid.parquet",
            data_dir / f"{task_name}_test.parquet"
        ),
        # Pattern 3: features_task_name_split.parquet
        (
            data_dir / f"features_{task_name}_train.parquet",
            data_dir / f"features_{task_name}_valid.parquet",
            data_dir / f"features_{task_name}_test.parquet"
        ),
    ]
    
    for train_path, valid_path, test_path in patterns:
        if train_path.exists() and valid_path.exists() and test_path.exists():
            return train_path, valid_path, test_path
    
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Train Random Forest baseline")
    
    # Task selection
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Single task to train"
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Train on all default tasks"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="List of tasks to train"
    )
    
    # Data
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing task data"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./rf_results"),
        help="Output directory"
    )
    
    # Training options
    parser.add_argument(
        "--n-iter",
        type=int,
        default=50,
        help="Number of random search iterations"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds"
    )
    parser.add_argument(
        "--normalize-y",
        action="store_true",
        default=True,
        help="Normalize target values"
    )
    parser.add_argument(
        "--no-normalize-y",
        action="store_false",
        dest="normalize_y",
        help="Don't normalize target values"
    )
    parser.add_argument(
        "--compute-shap",
        action="store_true",
        default=True,
        help="Compute SHAP values"
    )
    parser.add_argument(
        "--no-shap",
        action="store_false",
        dest="compute_shap",
        help="Skip SHAP computation"
    )
    
    # Misc
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    logger = logging.getLogger("main")
    
    # Determine tasks to train
    if args.all_tasks:
        tasks = DEFAULT_TASKS
    elif args.tasks:
        tasks = args.tasks
    elif args.task:
        tasks = [args.task]
    else:
        logger.error("Specify --task, --tasks, or --all-tasks")
        return
    
    logger.info("=" * 60)
    logger.info("RANDOM FOREST BASELINE TRAINING")
    logger.info("=" * 60)
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    all_results = {}
    
    for task_name in tasks:
        logger.info(f"\n{'='*40}")
        logger.info(f"Training: {task_name}")
        logger.info("=" * 40)
        
        # Find data files
        train_path, valid_path, test_path = find_task_files(
            args.data_dir, task_name
        )
        
        if train_path is None:
            logger.warning(f"Data files not found for {task_name}, skipping")
            continue
        
        logger.info(f"  Train: {train_path}")
        logger.info(f"  Valid: {valid_path}")
        logger.info(f"  Test: {test_path}")
        
        # Create output directory
        task_output = args.output_dir / task_name
        task_output.mkdir(parents=True, exist_ok=True)
        
        # Train model
        try:
            results = train_rf_baseline(
                task_name=task_name,
                train_path=train_path,
                valid_path=valid_path,
                test_path=test_path,
                output_dir=task_output,
                compute_shap=args.compute_shap
            )
            all_results[task_name] = results
            
        except Exception as e:
            logger.error(f"Training failed for {task_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    
    for task_name, results in all_results.items():
        test_metrics = results.get('test_metrics', {})
        sp = test_metrics.get('spearman', 'N/A')
        rmse = test_metrics.get('rmse', 'N/A')
        
        if isinstance(sp, float):
            sp = f"{sp:.4f}"
        if isinstance(rmse, float):
            rmse = f"{rmse:.4f}"
        
        logger.info(f"  {task_name}: Spearman={sp}, RMSE={rmse}")
    
    logger.info(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
