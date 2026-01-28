#!/usr/bin/env python3
"""
Data preparation pipeline for SEAL-ADME.

This script orchestrates the complete data processing workflow:
1. Load raw data from TDC and/or ChEMBL
2. Validate and canonicalize SMILES
3. Deduplicate by label consistency
4. Split into train/valid/test sets
5. Create PyTorch Geometric graphs

Usage:
    python scripts/prepare_data.py --config configs/data_config.yaml
    python scripts/prepare_data.py --steps load_tdc,validate,graphs
    python scripts/prepare_data.py --dataset solubility --output-dir data/
    
Examples:
    # Run full pipeline with default config
    python scripts/prepare_data.py
    
    # Only load and validate data
    python scripts/prepare_data.py --steps load_tdc,load_aurora,validate
    
    # Create graphs from already-preprocessed data
    python scripts/prepare_data.py --steps graphs --data-dir data/
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    # Constants
    PRETRAIN_TASKS,
    FINETUNE_TASKS,
    CLASSIFICATION_TASKS,
    REGRESSION_TASKS,
    # Loaders
    load_tdc_tasks,
    load_pretrain_data,
    load_finetune_data,
    fetch_aurora_activities,
    find_aurora_target_ids,
    load_generated_molecules,
    read_csv_safe,
    # Preprocessing
    canonicalize_smiles,
    validate_smiles_column,
    deduplicate_by_label_consistency,
    preprocess_dataset,
    create_scaffold_split,
    create_random_split,
    summarize_dataset,
    # Graph creation
    GraphFeaturizer,
    featurize_dataset,
)


# =============================================================================
# Logging setup
# =============================================================================

def setup_logging(log_file: Optional[Path] = None, level: str = "INFO") -> None:
    """Configure logging for the pipeline."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='w'))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


# =============================================================================
# Configuration
# =============================================================================

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def get_default_config() -> Dict[str, Any]:
    """Return default configuration."""
    return {
        "tdc": {
            "pretrain_tasks": PRETRAIN_TASKS,
            "finetune_tasks": FINETUNE_TASKS,
        },
        "chembl": {
            "aurora_filters": {
                "min_target_count": 200,
                "organism": "Homo sapiens",
                "assay_type": "B",
                "require_pchembl": True,
            }
        },
        "preprocessing": {
            "isomeric_smiles": True,
            "deduplicate": True,
            "drop_invalid": True,
        },
        "splitting": {
            "method": "scaffold",  # "scaffold" or "random"
            "train_ratio": 0.8,
            "valid_ratio": 0.1,
            "seed": 42,
        },
        "graph": {
            "atom_features": {
                "include_optional": True,
                "include_numeric": True,
                "include_gasteiger": True,
            },
            "store_fragments": True,
        },
    }


# =============================================================================
# Pipeline Steps
# =============================================================================

def step_load_tdc(config: Dict, output_dir: Path) -> Dict[str, Any]:
    """
    Step 1a: Load TDC ADME datasets.
    
    Loads both pretraining (classification) and finetuning (regression) tasks.
    """
    logger = get_logger("load_tdc")
    logger.info("Loading TDC datasets...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    
    # Load pretrain data
    try:
        tdc_config = config.get("tdc", {})
        pretrain_tasks = tdc_config.get("pretrain_tasks", PRETRAIN_TASKS)
        
        logger.info(f"Loading {len(pretrain_tasks)} pretrain tasks...")
        pretrain_df = load_tdc_tasks(pretrain_tasks, source_tag="pretrain")
        
        pretrain_path = output_dir / "raw_pretrain.csv"
        pretrain_df.to_csv(pretrain_path, index=False)
        
        results["pretrain"] = {
            "rows": len(pretrain_df),
            "tasks": pretrain_df["task_name"].nunique(),
            "unique_molecules": pretrain_df["Drug_ID"].nunique(),
            "path": str(pretrain_path),
        }
        logger.info(f"Saved pretrain data: {len(pretrain_df)} rows")
        
    except Exception as e:
        logger.error(f"Failed to load pretrain data: {e}")
        results["pretrain"] = {"error": str(e)}
    
    # Load finetune data
    try:
        finetune_tasks = tdc_config.get("finetune_tasks", FINETUNE_TASKS)
        
        logger.info(f"Loading {len(finetune_tasks)} finetune tasks...")
        finetune_df = load_tdc_tasks(finetune_tasks, source_tag="finetune")
        
        finetune_path = output_dir / "raw_finetune.csv"
        finetune_df.to_csv(finetune_path, index=False)
        
        results["finetune"] = {
            "rows": len(finetune_df),
            "tasks": finetune_df["task_name"].nunique(),
            "unique_molecules": finetune_df["Drug_ID"].nunique(),
            "path": str(finetune_path),
        }
        logger.info(f"Saved finetune data: {len(finetune_df)} rows")
        
    except Exception as e:
        logger.error(f"Failed to load finetune data: {e}")
        results["finetune"] = {"error": str(e)}
    
    return results


def step_load_aurora(config: Dict, output_dir: Path) -> Dict[str, Any]:
    """
    Step 1b: Load Aurora kinase data from ChEMBL.
    
    Fetches bioactivity data for Aurora kinase targets and applies filters.
    """
    logger = get_logger("load_aurora")
    logger.info("Loading Aurora kinase data from ChEMBL...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Find Aurora targets
        target_ids = find_aurora_target_ids()
        logger.info(f"Found {len(target_ids)} Aurora kinase targets")
        
        # Fetch activities
        df_raw = fetch_aurora_activities(target_ids)
        logger.info(f"Fetched {len(df_raw)} raw activity records")
        
        # Save raw data
        raw_path = output_dir / "raw_aurora.csv"
        df_raw.to_csv(raw_path, index=False)
        
        if df_raw.empty:
            return {
                "raw_rows": 0,
                "filtered_rows": 0,
                "targets": target_ids,
                "raw_path": str(raw_path),
            }
        
        # Apply filters
        chembl_config = config.get("chembl", {}).get("aurora_filters", {})
        df_filtered = filter_aurora_data(
            df_raw,
            min_target_count=chembl_config.get("min_target_count", 200),
            organism=chembl_config.get("organism", "Homo sapiens"),
            assay_type=chembl_config.get("assay_type", "B"),
            require_pchembl=chembl_config.get("require_pchembl", True),
        )
        
        # Standardize columns
        df_filtered = standardize_aurora_columns(df_filtered)
        
        # Save filtered data
        filtered_path = output_dir / "aurora_filtered.csv"
        df_filtered.to_csv(filtered_path, index=False)
        
        logger.info(f"Filtered to {len(df_filtered)} records")
        
        return {
            "raw_rows": len(df_raw),
            "filtered_rows": len(df_filtered),
            "targets": target_ids,
            "raw_path": str(raw_path),
            "filtered_path": str(filtered_path),
        }
        
    except Exception as e:
        logger.error(f"Failed to load Aurora data: {e}")
        return {"error": str(e)}


def filter_aurora_data(
    df: pd.DataFrame,
    min_target_count: int = 200,
    organism: str = "Homo sapiens",
    assay_type: str = "B",
    require_pchembl: bool = True,
) -> pd.DataFrame:
    """
    Apply quality filters to Aurora kinase data.
    
    Args:
        df: Raw activity DataFrame
        min_target_count: Minimum records per target
        organism: Required assay organism
        assay_type: Required assay type (B = binding)
        require_pchembl: Require pChEMBL value
    """
    if df.empty:
        return df
    
    df_filtered = df.copy()
    
    # Filter by organism if column exists
    if "assay_organism" in df_filtered.columns and organism:
        df_filtered = df_filtered[
            df_filtered["assay_organism"].str.contains(organism, case=False, na=False)
        ]
    
    # Filter by assay type
    if "assay_type" in df_filtered.columns and assay_type:
        df_filtered = df_filtered[df_filtered["assay_type"] == assay_type]
    
    # Require pChEMBL value
    if require_pchembl and "pchembl_value" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["pchembl_value"].notna()]
    
    # Filter targets with enough data
    if "target_chembl_id" in df_filtered.columns and min_target_count > 0:
        target_counts = df_filtered["target_chembl_id"].value_counts()
        valid_targets = target_counts[target_counts >= min_target_count].index
        df_filtered = df_filtered[df_filtered["target_chembl_id"].isin(valid_targets)]
    
    return df_filtered


def standardize_aurora_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize Aurora DataFrame column names to match TDC format."""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Rename columns
    rename_map = {
        "molecule_chembl_id": "Drug_ID",
        "canonical_smiles": "original_smiles",
        "pchembl_value": "Y",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    # Add task_name based on target
    if "target_chembl_id" in df.columns:
        target_map = {
            "CHEMBL4722": "aurora_kinase_a",
            "CHEMBL2185": "aurora_kinase_b",
        }
        df["task_name"] = df["target_chembl_id"].map(
            lambda x: target_map.get(x, f"aurora_{x}")
        )
    else:
        df["task_name"] = "aurora"
    
    # Add metadata columns
    df["source"] = "chembl"
    df["task"] = "regression"
    
    return df


def step_validate(
    config: Dict,
    input_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Step 2: Validate SMILES and deduplicate data.
    
    Processes all CSV files in the input directory.
    """
    logger = get_logger("validate")
    logger.info("Validating and preprocessing data...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    preproc_config = config.get("preprocessing", {})
    results = {}
    
    # Find all CSV files to process
    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {input_dir}")
        return {"error": "No input files found"}
    
    for csv_file in sorted(csv_files):
        name = csv_file.stem
        logger.info(f"Processing {name}...")
        
        try:
            # Load data
            df = read_csv_safe(csv_file)
            initial_rows = len(df)
            
            # Run preprocessing pipeline
            df_processed, stats = preprocess_dataset(
                df,
                smiles_col="original_smiles",
                isomeric=preproc_config.get("isomeric_smiles", True),
                deduplicate=preproc_config.get("deduplicate", True),
                drop_invalid=preproc_config.get("drop_invalid", True),
            )
            
            # Save processed data
            output_path = output_dir / f"{name}_preprocessed.csv"
            df_processed.to_csv(output_path, index=False)
            
            results[name] = {
                "input_rows": initial_rows,
                "output_rows": len(df_processed),
                "dropped": initial_rows - len(df_processed),
                "path": str(output_path),
                "stats": stats,
            }
            
            logger.info(
                f"  {name}: {initial_rows} -> {len(df_processed)} rows "
                f"({initial_rows - len(df_processed)} dropped)"
            )
            
        except Exception as e:
            logger.error(f"Failed to process {name}: {e}")
            results[name] = {"error": str(e)}
    
    return results


def step_split(
    config: Dict,
    input_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Step 3: Split data into train/valid/test sets.
    
    Uses scaffold or random splitting based on configuration.
    """
    logger = get_logger("split")
    logger.info("Splitting data into train/valid/test...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    split_config = config.get("splitting", {})
    method = split_config.get("method", "scaffold")
    train_ratio = split_config.get("train_ratio", 0.8)
    valid_ratio = split_config.get("valid_ratio", 0.1)
    seed = split_config.get("seed", 42)
    
    results = {}
    
    # Find preprocessed files
    csv_files = list(input_dir.glob("*_preprocessed.csv"))
    if not csv_files:
        logger.warning(f"No preprocessed files found in {input_dir}")
        return {"error": "No input files found"}
    
    for csv_file in sorted(csv_files):
        name = csv_file.stem.replace("_preprocessed", "")
        logger.info(f"Splitting {name}...")
        
        try:
            df = pd.read_csv(csv_file)
            
            # Determine split method
            if method == "scaffold":
                train_df, valid_df, test_df = create_scaffold_split(
                    df,
                    smiles_col="canonical_smiles",
                    train_ratio=train_ratio,
                    valid_ratio=valid_ratio,
                    seed=seed,
                )
            else:
                # Check if classification for stratification
                stratify_col = None
                if "task" in df.columns:
                    task_type = df["task"].iloc[0]
                    if task_type == "classification":
                        stratify_col = "Y"
                
                train_df, valid_df, test_df = create_random_split(
                    df,
                    train_ratio=train_ratio,
                    valid_ratio=valid_ratio,
                    seed=seed,
                    stratify_col=stratify_col,
                )
            
            # Create output subdirectory
            task_dir = output_dir / name
            task_dir.mkdir(parents=True, exist_ok=True)
            
            # Save splits
            for split_name, split_df in [
                ("train", train_df),
                ("valid", valid_df),
                ("test", test_df),
            ]:
                split_path = task_dir / f"{name}_{split_name}.parquet"
                split_df.to_parquet(split_path, index=False)
            
            results[name] = {
                "train": len(train_df),
                "valid": len(valid_df),
                "test": len(test_df),
                "method": method,
                "output_dir": str(task_dir),
            }
            
            logger.info(
                f"  {name}: train={len(train_df)}, "
                f"valid={len(valid_df)}, test={len(test_df)}"
            )
            
        except Exception as e:
            logger.error(f"Failed to split {name}: {e}")
            results[name] = {"error": str(e)}
    
    return results


def step_combine_pretrain(
    config: Dict,
    splits_dir: Path,
    graphs_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Step 5: Combine classification tasks into pretrain format.
    
    Creates combined metadata parquets and symlinks/copies graphs
    for multi-task pretraining.
    """
    import shutil
    
    logger = get_logger("combine_pretrain")
    logger.info("Combining classification tasks for pretraining...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    pretrain_graphs_dir = output_dir / "graphs"
    pretrain_graphs_dir.mkdir(parents=True, exist_ok=True)
    
    # Get pretrain tasks from config
    tdc_config = config.get("tdc", {})
    pretrain_tasks = tdc_config.get("pretrain_tasks", PRETRAIN_TASKS)
    
    # Classification tasks only
    classification_tasks = [t for t in pretrain_tasks if t in CLASSIFICATION_TASKS]
    
    combined_metadata = {split: [] for split in ['train', 'valid', 'test']}
    graph_counter = 0
    
    for task_name in classification_tasks:
        task_split_dir = splits_dir / task_name
        task_graph_dir = graphs_dir / task_name
        
        if not task_split_dir.exists():
            logger.warning(f"Skipping {task_name} - splits not found")
            continue
        
        for split in ['train', 'valid', 'test']:
            parquet_file = task_split_dir / f"{task_name}_{split}.parquet"
            graph_split_dir = task_graph_dir / split
            
            if not parquet_file.exists():
                continue
            
            df = pd.read_parquet(parquet_file)
            
            # Process each row
            for idx, row in df.iterrows():
                # Find corresponding graph file
                old_graph_pattern = f"{task_name}_{split}_{idx:06d}.pt"
                old_graph_path = graph_split_dir / old_graph_pattern
                
                if not old_graph_path.exists():
                    # Try alternate naming
                    graph_files = list(graph_split_dir.glob(f"*_{idx:06d}.pt"))
                    if graph_files:
                        old_graph_path = graph_files[0]
                    else:
                        continue
                
                # Create new graph ID
                new_graph_id = f"pretrain_{graph_counter:08d}"
                new_graph_path = pretrain_graphs_dir / f"{new_graph_id}.pt"
                
                # Copy graph file
                shutil.copy2(old_graph_path, new_graph_path)
                
                # Add to metadata
                meta_row = {
                    'graph_id': new_graph_id,
                    'task_name': task_name,
                    'label': float(row.get('Y', row.get('y', 0))),
                    'split': split,
                    'original_drug_id': row.get('Drug_ID', ''),
                    'smiles': row.get('canonical_smiles', ''),
                }
                combined_metadata[split].append(meta_row)
                graph_counter += 1
        
        logger.info(f"  Processed {task_name}")
    
    # Save combined metadata
    results = {}
    for split in ['train', 'valid', 'test']:
        if combined_metadata[split]:
            meta_df = pd.DataFrame(combined_metadata[split])
            meta_path = output_dir / f"pretrain_{split}.parquet"
            meta_df.to_parquet(meta_path, index=False)
            results[split] = {
                'rows': len(meta_df),
                'tasks': meta_df['task_name'].nunique(),
                'path': str(meta_path),
            }
            logger.info(f"  Saved {split}: {len(meta_df)} samples")
    
    results['total_graphs'] = graph_counter
    results['graphs_dir'] = str(pretrain_graphs_dir)
    
    return results


def step_create_graphs(
    config: Dict,
    input_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Step 4: Create PyTorch Geometric graphs.
    
    Creates graph representations with fragment assignments for all splits.
    """
    logger = get_logger("create_graphs")
    logger.info("Creating PyTorch Geometric graphs...")
    
    import torch
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    graph_config = config.get("graph", {})
    atom_config = graph_config.get("atom_features", {})
    
    # Initialize featurizer
    featurizer = GraphFeaturizer(
        y_column="Y",
        smiles_col="canonical_smiles",
        include_optional=atom_config.get("include_optional", True),
        include_numeric=atom_config.get("include_numeric", True),
        include_gasteiger=atom_config.get("include_gasteiger", True),
        store_fragments=graph_config.get("store_fragments", True),
    )
    
    results = {}
    
    # Process each task directory
    for task_dir in sorted(input_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        
        task_name = task_dir.name
        logger.info(f"Creating graphs for {task_name}...")
        
        try:
            task_results = {}
            
            # Find all splits
            parquet_files = list(task_dir.glob("*.parquet"))
            if not parquet_files:
                continue
            
            # Compute normalization stats from training set
            train_file = task_dir / f"{task_name}_train.parquet"
            if train_file.exists():
                train_df = pd.read_parquet(train_file)
                mean = float(train_df["Y"].mean()) if "Y" in train_df.columns else 0.0
                std = float(train_df["Y"].std()) if "Y" in train_df.columns else 1.0
                if std == 0:
                    std = 1.0
            else:
                mean, std = 0.0, 1.0
            
            stats = {"mean": mean, "std": std}
            
            # Create output directory
            graph_task_dir = output_dir / task_name
            graph_task_dir.mkdir(parents=True, exist_ok=True)
            
            # Save normalization stats
            stats_path = graph_task_dir / "normalization_stats.json"
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            
            # Process each split
            for parquet_file in sorted(parquet_files):
                split_name = parquet_file.stem.split("_")[-1]  # train, valid, test
                logger.info(f"  Processing {split_name} split...")
                
                df = pd.read_parquet(parquet_file)
                graphs = featurizer(df, stats)
                
                # Save graphs
                split_dir = graph_task_dir / split_name
                split_dir.mkdir(parents=True, exist_ok=True)
                
                for i, graph in enumerate(graphs):
                    graph_path = split_dir / f"{task_name}_{split_name}_{i:06d}.pt"
                    torch.save(graph, graph_path)
                
                task_results[split_name] = {
                    "n_graphs": len(graphs),
                    "output_dir": str(split_dir),
                }
                
                # Add sample info
                if graphs:
                    task_results[split_name].update({
                        "sample_n_atoms": int(graphs[0].x.shape[0]),
                        "sample_n_features": int(graphs[0].x.shape[1]),
                        "sample_n_fragments": int(graphs[0].s.shape[1]),
                    })
            
            task_results["normalization"] = stats
            results[task_name] = task_results
            
            logger.info(f"  Completed {task_name}")
            
        except Exception as e:
            logger.error(f"Failed to create graphs for {task_name}: {e}")
            results[task_name] = {"error": str(e)}
    
    return results


# =============================================================================
# Main pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SEAL-ADME Data Preparation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  load_tdc     - Load ADME datasets from TDC
  load_aurora  - Load Aurora kinase data from ChEMBL
  validate     - Validate SMILES and deduplicate
  split        - Split into train/valid/test
  graphs       - Create PyTorch Geometric graphs

Examples:
  %(prog)s --steps all
  %(prog)s --steps load_tdc,validate,split,graphs
  %(prog)s --steps graphs --data-dir data/processed/
        """,
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_config.yaml"),
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help="Comma-separated list of steps to run (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Base data directory (default: data/)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to log file (optional)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file, args.log_level)
    logger = get_logger("main")
    
    logger.info("=" * 70)
    logger.info("SEAL-ADME Data Preparation Pipeline")
    logger.info("=" * 70)
    
    # Load configuration
    config = get_default_config()
    if args.config.exists():
        user_config = load_config(args.config)
        # Deep merge user config
        for key, value in user_config.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        logger.info(f"Loaded configuration from {args.config}")
    else:
        logger.info(f"Using default configuration (config file not found: {args.config})")
    
    # Setup directories
    data_dir = args.data_dir
    raw_dir = data_dir / "raw"
    preprocessed_dir = data_dir / "preprocessed"
    splits_dir = data_dir / "splits"
    graphs_dir = data_dir / "graphs"
    
    # Determine steps to run
    all_steps = ["load_tdc", "load_aurora", "validate", "split", "graphs", "combine_pretrain"]
    if args.steps.lower() == "all":
        steps = all_steps
    else:
        steps = [s.strip().lower() for s in args.steps.split(",")]
        invalid = set(steps) - set(all_steps)
        if invalid:
            logger.error(f"Invalid steps: {invalid}. Valid steps: {all_steps}")
            sys.exit(1)
    
    logger.info(f"Steps to run: {steps}")
    logger.info(f"Data directory: {data_dir}")
    
    # Initialize manifest
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "config_file": str(args.config),
        "data_dir": str(data_dir),
        "steps_requested": steps,
        "results": {},
    }
    
    # Run pipeline steps
    try:
        if "load_tdc" in steps:
            logger.info("")
            logger.info("=" * 70)
            logger.info("STEP: Loading TDC data")
            logger.info("=" * 70)
            manifest["results"]["load_tdc"] = step_load_tdc(config, raw_dir)
        
        if "load_aurora" in steps:
            logger.info("")
            logger.info("=" * 70)
            logger.info("STEP: Loading Aurora kinase data")
            logger.info("=" * 70)
            manifest["results"]["load_aurora"] = step_load_aurora(config, raw_dir)
        
        if "validate" in steps:
            logger.info("")
            logger.info("=" * 70)
            logger.info("STEP: Validating and preprocessing")
            logger.info("=" * 70)
            manifest["results"]["validate"] = step_validate(
                config, raw_dir, preprocessed_dir
            )
        
        if "split" in steps:
            logger.info("")
            logger.info("=" * 70)
            logger.info("STEP: Creating train/valid/test splits")
            logger.info("=" * 70)
            manifest["results"]["split"] = step_split(
                config, preprocessed_dir, splits_dir
            )
        
        if "graphs" in steps:
            logger.info("")
            logger.info("=" * 70)
            logger.info("STEP: Creating PyTorch Geometric graphs")
            logger.info("=" * 70)
            manifest["results"]["graphs"] = step_create_graphs(
                config, splits_dir, graphs_dir
            )
        
        if "combine_pretrain" in steps:
            logger.info("")
            logger.info("=" * 70)
            logger.info("STEP: Combining tasks for pretraining")
            logger.info("=" * 70)
            pretrain_dir = data_dir / "pretrain"
            manifest["results"]["combine_pretrain"] = step_combine_pretrain(
                config, splits_dir, graphs_dir, pretrain_dir
            )
        
        manifest["status"] = "completed"
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        manifest["status"] = "interrupted"
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        manifest["status"] = "failed"
        manifest["error"] = str(e)
    
    # Save manifest
    manifest["completed_at"] = datetime.now().isoformat()
    manifest_path = data_dir / "pipeline_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Pipeline {manifest['status']}")
    logger.info(f"Manifest saved to: {manifest_path}")
    logger.info("=" * 70)
    
    # Exit with appropriate code
    sys.exit(0 if manifest["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
