#!/usr/bin/env python3
"""
Data preparation pipeline for SEAL-ADME.

Pipeline steps:
1. Load data from TDC (classification tasks for pretraining, regression for finetuning)
2. Load Aurora kinase data from ChEMBL (optional)
3. Validate and canonicalize SMILES
4. Deduplicate by label consistency
5. Apply scaffold-based splitting
6. Build PyTorch Geometric graphs

Usage:
    python scripts/prepare_data.py --config configs/data_config.yaml
    python scripts/prepare_data.py --steps load,preprocess,split,graphs
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    TDCLoader,
    ChEMBLAuroraLoader,
    filter_aurora_data,
    convert_aurora_to_standard,
    standardize_dataframe,
    create_scaffold_split,
    GraphBuilder,
    save_graphs,
    META_COLUMNS,
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


def step_load_tdc(config, output_dir):
    """Load TDC datasets."""
    logger = logging.getLogger("load_tdc")
    logger.info("Loading TDC datasets...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loader = TDCLoader(output_dir)
    
    # Load pretrain (classification only)
    pretrain_df = loader.load_pretrain(classification_only=True)
    loader.save(pretrain_df, "raw_pretrain.csv")
    
    # Load finetune
    finetune_df = loader.load_finetune()
    loader.save(finetune_df, "raw_finetune.csv")
    
    logger.info(f"Pretrain: {len(pretrain_df)} rows, {pretrain_df['task_name'].nunique()} tasks")
    logger.info(f"Finetune: {len(finetune_df)} rows, {finetune_df['task_name'].nunique()} tasks")
    
    return {
        "pretrain_rows": len(pretrain_df),
        "finetune_rows": len(finetune_df),
        "pretrain_tasks": list(pretrain_df['task_name'].unique()),
        "finetune_tasks": list(finetune_df['task_name'].unique())
    }


def step_load_aurora(config, output_dir):
    """Load Aurora kinase data from ChEMBL."""
    logger = logging.getLogger("load_aurora")
    logger.info("Loading Aurora kinase data from ChEMBL...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loader = ChEMBLAuroraLoader(output_dir)
    
    target_ids = loader.find_aurora_targets()
    logger.info(f"Found {len(target_ids)} Aurora targets")
    
    df = loader.fetch_activities(target_ids)
    loader.save(df, "raw_aurora.csv")
    
    # Apply filters
    chembl_config = config.get("chembl", {}).get("aurora_filters", {})
    df_filtered = filter_aurora_data(
        df,
        min_target_count=chembl_config.get("min_target_count", 200),
        organism=chembl_config.get("organism", "Homo sapiens"),
        assay_type=chembl_config.get("assay_type", "B"),
        require_pchembl=chembl_config.get("require_pchembl", True)
    )
    loader.save(df_filtered, "aurora_filtered.csv")
    
    # Convert to standard format
    df_standard = convert_aurora_to_standard(df_filtered)
    loader.save(df_standard, "aurora_standard.csv")
    
    return {
        "raw_rows": len(df),
        "filtered_rows": len(df_filtered),
        "targets": target_ids
    }


def step_preprocess(config, input_dir, output_dir):
    """Validate SMILES and deduplicate."""
    logger = logging.getLogger("preprocess")
    logger.info("Preprocessing data...")
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for csv_file in input_dir.glob("raw_*.csv"):
        name = csv_file.stem.replace("raw_", "")
        logger.info(f"Processing {name}...")
        
        try:
            df = pd.read_csv(csv_file)
            df, stats = standardize_dataframe(
                df,
                smiles_col="original_smiles",
                drop_invalid=True,
                deduplicate=True
            )
            
            # Save
            out_path = output_dir / f"{name}_clean.csv"
            df.to_csv(out_path, index=False)
            logger.info(f"Saved {len(df)} rows to {out_path}")
            
            results[name] = stats
        except Exception as e:
            logger.error(f"Failed to process {name}: {e}")
            results[name] = {"error": str(e)}
    
    return results


def step_split(config, input_dir, output_dir):
    """Apply scaffold-based splitting."""
    logger = logging.getLogger("split")
    logger.info("Applying scaffold splits...")
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    split_config = config.get("splitting", {})
    seed = split_config.get("seed", 42)
    frac = split_config.get("fractions", [0.8, 0.1, 0.1])
    
    results = {}
    
    for csv_file in input_dir.glob("*_clean.csv"):
        name = csv_file.stem.replace("_clean", "")
        logger.info(f"Splitting {name}...")
        
        try:
            df = pd.read_csv(csv_file)
            
            # Split by task
            tasks = df['task_name'].unique()
            all_splits = {'train': [], 'valid': [], 'test': []}
            
            for task in tasks:
                task_df = df[df['task_name'] == task].copy()
                
                splits = create_scaffold_split(
                    task_df,
                    seed=seed,
                    frac=frac,
                    entity="canonical_smiles"
                )
                
                for split_name, split_df in splits.items():
                    split_df = split_df.copy()
                    split_df['split'] = split_name
                    all_splits[split_name].append(split_df)
            
            # Combine and save
            for split_name, dfs in all_splits.items():
                if dfs:
                    combined = pd.concat(dfs, ignore_index=True)
                    out_path = output_dir / f"{name}_{split_name}.parquet"
                    combined.to_parquet(out_path, index=False)
                    logger.info(f"Saved {len(combined)} rows to {out_path}")
            
            results[name] = {
                "tasks": list(tasks),
                "train": sum(len(d) for d in all_splits['train']),
                "valid": sum(len(d) for d in all_splits['valid']),
                "test": sum(len(d) for d in all_splits['test'])
            }
            
        except Exception as e:
            logger.error(f"Failed to split {name}: {e}")
            results[name] = {"error": str(e)}
    
    return results


def step_build_graphs(config, input_dir, output_dir):
    """Build PyTorch Geometric graphs."""
    logger = logging.getLogger("build_graphs")
    logger.info("Building PyG graphs...")
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    graph_config = config.get("graph", {})
    normalize_y = graph_config.get("normalize_y", False)
    store_fragments = graph_config.get("store_fragments", True)
    
    builder = GraphBuilder(
        y_column="Y",
        smiles_col="canonical_smiles",
        drug_id_col="Drug_ID",
        task_name_col="task_name",
        store_fragments=store_fragments,
        normalize_y=normalize_y
    )
    
    results = {}
    
    # Process each parquet file
    for parquet_file in input_dir.glob("*.parquet"):
        name = parquet_file.stem
        logger.info(f"Building graphs for {name}...")
        
        try:
            df = pd.read_parquet(parquet_file)
            
            # Compute normalization stats from training data if needed
            y_mean, y_std = 0.0, 1.0
            if normalize_y and 'train' in name:
                y_mean = df['Y'].mean()
                y_std = df['Y'].std()
                if y_std == 0:
                    y_std = 1.0
            
            graphs = builder.build_from_dataframe(df, y_mean=y_mean, y_std=y_std)
            
            # Save graphs
            graph_dir = output_dir / name
            save_graphs(graphs, graph_dir, prefix=f"graph_{name}")
            
            # Save stats
            stats = {"mean": float(y_mean), "std": float(y_std), "n_graphs": len(graphs)}
            with open(graph_dir / "stats.json", 'w') as f:
                json.dump(stats, f, indent=2)
            
            results[name] = stats
            logger.info(f"Built {len(graphs)} graphs for {name}")
            
        except Exception as e:
            logger.error(f"Failed to build graphs for {name}: {e}")
            results[name] = {"error": str(e)}
    
    return results


def main():
    parser = argparse.ArgumentParser(description="SEAL-ADME Data Preparation")
    parser.add_argument("--config", type=Path, default=Path("configs/data_config.yaml"))
    parser.add_argument("--steps", type=str, default="all",
                       help="Comma-separated: load_tdc,load_aurora,preprocess,split,graphs")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
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
    
    # Setup directories
    data_dir = args.data_dir
    raw_dir = data_dir / "raw"
    clean_dir = data_dir / "clean"
    split_dir = data_dir / "splits"
    graph_dir = data_dir / "graphs"
    
    for d in [raw_dir, clean_dir, split_dir, graph_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Determine steps
    if args.steps == "all":
        steps = ["load_tdc", "preprocess", "split", "graphs"]
    else:
        steps = [s.strip() for s in args.steps.split(",")]
    
    # Run pipeline
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "config": str(args.config),
        "steps": {}
    }
    
    if "load_tdc" in steps:
        logger.info("=" * 60)
        logger.info("STEP: Load TDC data")
        manifest["steps"]["load_tdc"] = step_load_tdc(config, raw_dir)
    
    if "load_aurora" in steps:
        logger.info("=" * 60)
        logger.info("STEP: Load Aurora kinase data")
        manifest["steps"]["load_aurora"] = step_load_aurora(config, raw_dir)
    
    if "preprocess" in steps:
        logger.info("=" * 60)
        logger.info("STEP: Preprocess")
        manifest["steps"]["preprocess"] = step_preprocess(config, raw_dir, clean_dir)
    
    if "split" in steps:
        logger.info("=" * 60)
        logger.info("STEP: Scaffold split")
        manifest["steps"]["split"] = step_split(config, clean_dir, split_dir)
    
    if "graphs" in steps:
        logger.info("=" * 60)
        logger.info("STEP: Build graphs")
        manifest["steps"]["graphs"] = step_build_graphs(config, split_dir, graph_dir)
    
    # Save manifest
    manifest_path = data_dir / "pipeline_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info("=" * 60)
    logger.info(f"Pipeline complete! Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
