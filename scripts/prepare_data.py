#!/usr/bin/env python3
"""
Data preparation pipeline for SEAL-ADME.

Simplified pipeline:
1. Load TDC classification tasks for pretraining (no split)
2. Load TDC regression tasks for finetuning with scaffold split
3. Build PyG graphs with BRICS fragmentation
4. Normalize regression labels to mean=0, std=1

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --data-dir data
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    TDCLoader,
    GraphFeaturizer,
    save_graphs,
    PRETRAIN_TASKS,
    FINETUNE_TASKS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s"
)
logger = logging.getLogger(__name__)


def prepare_pretrain_data(data_dir: Path, featurizer: GraphFeaturizer):
    """
    Prepare pretraining data (classification tasks).
    
    No splitting - all data used for training.
    Graphs are saved per task with task_name attribute.
    """
    logger.info("=" * 60)
    logger.info("PREPARING PRETRAIN DATA (Classification)")
    logger.info("=" * 60)
    
    loader = TDCLoader()
    task_data = loader.load_pretrain_tasks()
    
    output_dir = data_dir / "graphs" / "pretrain"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_graphs = []
    task_stats = {}
    
    for task_name, df in task_data.items():
        logger.info(f"\nProcessing {task_name}...")
        
        # Create graphs (no normalization for classification)
        graphs = featurizer(
            df,
            task_name=task_name,
            normalize_y=False
        )
        
        all_graphs.extend(graphs)
        task_stats[task_name] = len(graphs)
        logger.info(f"  Created {len(graphs)} graphs")
    
    # Save all pretrain graphs together
    logger.info(f"\nSaving {len(all_graphs)} total pretrain graphs...")
    save_graphs(all_graphs, output_dir, prefix="graph_pretrain")
    
    # Save stats
    stats = {
        "total_graphs": len(all_graphs),
        "tasks": task_stats
    }
    with open(output_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Pretrain data saved to {output_dir}")
    return stats


def prepare_finetune_data(
    data_dir: Path,
    featurizer: GraphFeaturizer,
    seed: int = 42,
    frac: list = None
):
    """
    Prepare finetuning data (regression tasks).
    
    Uses TDC scaffold split and normalizes Y to mean=0, std=1.
    """
    logger.info("=" * 60)
    logger.info("PREPARING FINETUNE DATA (Regression)")
    logger.info("=" * 60)
    
    if frac is None:
        frac = [0.7, 0.1, 0.2]
    
    loader = TDCLoader()
    task_splits = loader.load_finetune_tasks(seed=seed, frac=frac)
    norm_stats = loader.compute_normalization_stats(task_splits)
    
    output_dir = data_dir / "graphs" / "finetune"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_stats = {"tasks": {}, "normalization": norm_stats}
    
    for task_name, splits in task_splits.items():
        logger.info(f"\nProcessing {task_name}...")
        
        task_dir = output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Get normalization stats (computed from training data)
        y_mean = norm_stats[task_name]['mean']
        y_std = norm_stats[task_name]['std']
        
        task_graph_counts = {}
        
        for split_name, df in splits.items():
            logger.info(f"  {split_name}: {len(df)} samples")
            
            # Create graphs with normalization
            graphs = featurizer(
                df,
                task_name=task_name,
                normalize_y=True,
                y_mean=y_mean,
                y_std=y_std
            )
            
            # Save graphs for this split
            split_dir = task_dir / split_name
            save_graphs(graphs, split_dir, prefix=f"graph_{task_name}_{split_name}")
            
            task_graph_counts[split_name] = len(graphs)
        
        all_stats["tasks"][task_name] = task_graph_counts
        
        # Save task-specific stats
        task_stats = {
            "normalization": {"mean": y_mean, "std": y_std},
            "splits": task_graph_counts
        }
        with open(task_dir / "stats.json", 'w') as f:
            json.dump(task_stats, f, indent=2)
    
    # Save overall stats
    with open(output_dir / "stats.json", 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    logger.info(f"\nFinetune data saved to {output_dir}")
    return all_stats


def main():
    parser = argparse.ArgumentParser(description="SEAL-ADME Data Preparation")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Base data directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for scaffold split"
    )
    parser.add_argument(
        "--pretrain-only",
        action="store_true",
        help="Only prepare pretrain data"
    )
    parser.add_argument(
        "--finetune-only",
        action="store_true",
        help="Only prepare finetune data"
    )
    
    args = parser.parse_args()
    
    # Create featurizer
    featurizer = GraphFeaturizer(
        y_column='Y',
        smiles_col='Drug',
        store_fragments=True
    )
    
    # Prepare data
    args.data_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = {"seed": args.seed}
    
    if not args.finetune_only:
        pretrain_stats = prepare_pretrain_data(args.data_dir, featurizer)
        manifest["pretrain"] = pretrain_stats
    
    if not args.pretrain_only:
        finetune_stats = prepare_finetune_data(
            args.data_dir,
            featurizer,
            seed=args.seed
        )
        manifest["finetune"] = finetune_stats
    
    # Save manifest
    manifest_path = args.data_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info("=" * 60)
    logger.info(f"DATA PREPARATION COMPLETE")
    logger.info(f"Manifest saved to: {manifest_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
