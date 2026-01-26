#!/usr/bin/env python3
"""
Data preparation pipeline for SEAL-ADME.

This script orchestrates the complete data processing workflow:
1. Load raw data from TDC and/or ChEMBL
2. Validate and canonicalize SMILES
3. Deduplicate by label consistency
4. Compute molecular features
5. Create PyTorch Geometric graphs

Usage:
    python scripts/prepare_data.py --config configs/data_config.yaml
    python scripts/prepare_data.py --steps load,validate,featurize
    python scripts/prepare_data.py --dataset aurora --output-dir data/
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import yaml
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    TDCLoader,
    ChEMBLAuroraLoader,
    filter_aurora_data,
    DataPreprocessor,
    MolecularFeaturizer,
    GraphFeaturizer,
    save_graphs,
    impute_missing_descriptors,
)


def setup_logging(log_file: Optional[Path] = None, level: str = "INFO"):
    """Configure logging."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode='w'))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        handlers=handlers
    )


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def step_load_tdc(config: dict, output_dir: Path) -> dict:
    """Step 1a: Load TDC ADME datasets."""
    logger = logging.getLogger("load_tdc")
    logger.info("Loading TDC datasets...")
    
    loader = TDCLoader(output_dir)
    
    # Load pretrain and finetune tasks
    pretrain_df = loader.load_pretrain()
    finetune_df = loader.load_finetune()
    
    # Save
    loader.save(pretrain_df, "raw_pretrain.csv")
    loader.save(finetune_df, "raw_finetune.csv")
    
    return {
        "pretrain_rows": len(pretrain_df),
        "finetune_rows": len(finetune_df),
        "pretrain_tasks": pretrain_df["task_name"].nunique(),
        "finetune_tasks": finetune_df["task_name"].nunique()
    }


def step_load_aurora(config: dict, output_dir: Path) -> dict:
    logger = logging.getLogger("load_aurora")
    logger.info("Loading Aurora kinase data from ChEMBL...")
    loader = ChEMBLAuroraLoader(output_dir)
    target_ids = loader.find_aurora_targets()
    logger.info("Found %d Aurora kinase targets", len(target_ids))
    df = loader.fetch_activities(target_ids)
    loader.save(df, "raw_aurora.csv")
    chembl_config = config.get("chembl", {}).get("aurora_filters", {})
    df_filtered = filter_aurora_data(
        df,
        min_target_count=chembl_config.get("min_target_count", 200),
        organism=chembl_config.get("organism", "Homo sapiens"),
        assay_type=chembl_config.get("assay_type", "B"),
        require_pchembl=chembl_config.get("require_pchembl", True)
    )
    if df_filtered is None or len(df_filtered) == 0:
        loader.save(df_filtered, "aurora_filtered.csv")
        return {"raw_rows": len(df), "filtered_rows": 0, "targets": target_ids}
    df_filtered = df_filtered.copy()
    if "molecule_chembl_id" in df_filtered.columns:
        df_filtered = df_filtered.rename(columns={"molecule_chembl_id": "Drug_ID"})
    if "canonical_smiles" in df_filtered.columns:
        df_filtered = df_filtered.rename(columns={"canonical_smiles": "original_smiles"})
    if "target_chembl_id" in df_filtered.columns:
        df_filtered["target_chembl_id"] = df_filtered["target_chembl_id"].astype(str).str.strip().fillna("")
    else:
        df_filtered["target_chembl_id"] = ""
    def _map_task(tid: str) -> str:
        if tid == "CHEMBL4722":
            return "Aurora Kinase A"
        if tid == "CHEMBL2185":
            return "Aurora Kinase B"
        if tid:
            return f"Aurora_{tid}"
        return "Aurora_Unknown"
    df_filtered["task_name"] = df_filtered["target_chembl_id"].map(_map_task)
    df_filtered["source"] = "Aurora"
    df_filtered["task"] = "regression"
    desired = [
        "Drug_ID",
        "original_smiles",
        "standard_value",
        "standard_units",
        "standard_relation",
        "target_chembl_id",
        "standard_type",
        "pchembl_value",
        "assay_chembl_id",
        "assay_description",
        "assay_type",
        "assay_organism",
        "assay_parameters",
        "task_name",
        "source",
        "task"
    ]
    existing = [c for c in desired if c in df_filtered.columns]
    others = [c for c in df_filtered.columns if c not in existing]
    df_filtered = df_filtered[existing + others]
    loader.save(df_filtered, "aurora_filtered.csv")
    return {"raw_rows": len(df), "filtered_rows": len(df_filtered), "targets": target_ids}


def step_validate(config: dict, input_dir: Path, output_dir: Path) -> dict:
    logger = logging.getLogger("validate")
    logger.info("Validating SMILES...")
    preprocessor = DataPreprocessor(input_dir=input_dir, output_dir=output_dir)
    results = {}
    paths = set()
    for p in input_dir.glob("*filtered.csv"):
        paths.add(p)
    raw_pre = input_dir / "raw_pretrain.csv"
    raw_fine = input_dir / "raw_finetune.csv"
    if raw_pre.exists():
        paths.add(raw_pre)
    if raw_fine.exists():
        paths.add(raw_fine)
    if not paths:
        logger.warning("No model-ready CSVs found in %s", input_dir)
    for csv_file in sorted(paths):
        name = csv_file.stem
        logger.info("Processing %s...", name)
        try:
            df, stats = preprocessor.process_file(csv_file.name, drop_invalid=True, deduplicate=True)
            preprocessor.save(df, f"{name}_preprocessed.csv")
            results[name] = stats
        except Exception as e:
            logger.error("Failed to process %s: %s", name, e)
            results[name] = {"error": str(e)}
    return results



def step_featurize(config: dict, input_dir: Path, output_dir: Path) -> dict:
    """Step 3: Compute molecular features."""
    logger = logging.getLogger("featurize")
    logger.info("Computing molecular features...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    feat_config = config.get("featurization", {})
    fp_config = feat_config.get("fingerprint", {})
    
    featurizer = MolecularFeaturizer(
        fp_n_bits=fp_config.get("n_bits", 2048),
        fp_radius=fp_config.get("radius", 2),
        compute_fingerprints=True,
        compute_descriptors=feat_config.get("descriptors", {}).get("compute", True)
    )
    
    results = {}
    for csv_file in input_dir.glob("*_preprocessed.csv"):
        name = csv_file.stem.replace("_preprocessed", "")
        logger.info(f"Featurizing {name}...")
        
        try:
            df = pd.read_csv(csv_file)
            df_feat = featurizer.featurize_dataframe(df)
            
            # Impute missing values
            impute_strategy = feat_config.get("descriptors", {}).get(
                "impute_strategy", "median"
            )
            df_feat = impute_missing_descriptors(df_feat, strategy=impute_strategy)
            
            # Save as parquet
            output_path = output_dir / f"features_{name}.parquet"
            df_feat.to_parquet(output_path, index=False)
            
            results[name] = {
                "rows": len(df_feat),
                "columns": len(df_feat.columns),
                "output": str(output_path)
            }
        except Exception as e:
            logger.error(f"Failed to featurize {name}: {e}")
            results[name] = {"error": str(e)}
    
    return results


def step_create_graphs(
    config: dict,
    input_dir: Path,
    output_dir: Path,
    dataset_name: str = "aurora"
) -> dict:
    """Step 4: Create PyTorch Geometric graphs."""
    logger = logging.getLogger("create_graphs")
    logger.info("Creating PyG graphs...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    graph_config = config.get("graph", {})
    atom_config = graph_config.get("atom_features", {})
    
    featurizer = GraphFeaturizer(
        y_column='Y',
        smiles_col='canonical_smiles',
        include_optional=atom_config.get("include_optional", True),
        include_numeric=atom_config.get("include_numeric", True),
        include_gasteiger=atom_config.get("include_gasteiger", True),
        store_fragments=graph_config.get("store_fragments", True)
    )
    
    results = {}
    
    # Look for parquet files
    for parquet_file in input_dir.glob("*.parquet"):
        name = parquet_file.stem.replace("features_", "")
        logger.info(f"Creating graphs for {name}...")
        
        try:
            df = pd.read_parquet(parquet_file)
            
            # Compute normalization stats
            mean = df['Y'].mean() if 'Y' in df.columns else 0.0
            std = df['Y'].std() if 'Y' in df.columns else 1.0
            if std == 0:
                std = 1.0
            
            stats = {"mean": mean, "std": std}
            
            # Create graphs
            graphs = featurizer(df, stats)
            
            # Save graphs
            graph_output_dir = output_dir / name
            save_graphs(graphs, graph_output_dir, prefix=f"graph_{name}")
            
            # Save stats for later use
            stats_path = graph_output_dir / "normalization_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            results[name] = {
                "n_graphs": len(graphs),
                "mean": mean,
                "std": std,
                "output_dir": str(graph_output_dir)
            }
            
            if graphs:
                results[name]["sample_n_atoms"] = graphs[0].x.shape[0]
                results[name]["sample_n_fragments"] = graphs[0].s.shape[1]
                
        except Exception as e:
            logger.error(f"Failed to create graphs for {name}: {e}")
            results[name] = {"error": str(e)}
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="SEAL-ADME Data Preparation Pipeline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_config.yaml"),
        help="Path to configuration file"
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help="Comma-separated list of steps: load_tdc,load_aurora,validate,featurize,graphs"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Base data directory"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Log file path"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file, args.log_level)
    logger = logging.getLogger("main")
    
    # Load config
    if args.config.exists():
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        logger.warning(f"Config file not found: {args.config}, using defaults")
        config = {}
    
    # Setup directories
    data_dir = args.data_dir
    raw_dir = data_dir / "raw"
    preprocessed_dir = data_dir / "preprocessed"
    features_dir = data_dir / "features"
    graphs_dir = data_dir / "graphs"
    
    for d in [raw_dir, preprocessed_dir, features_dir, graphs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Determine which steps to run
    if args.steps == "all":
        steps = ["load_tdc", "load_aurora", "validate", "featurize", "graphs"]
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
        logger.info("STEP: Loading TDC data")
        manifest["steps"]["load_tdc"] = step_load_tdc(config, raw_dir)
    
    if "load_aurora" in steps:
        logger.info("=" * 60)
        logger.info("STEP: Loading Aurora kinase data")
        manifest["steps"]["load_aurora"] = step_load_aurora(config, raw_dir)
    
    if "validate" in steps:
        logger.info("=" * 60)
        logger.info("STEP: Validating and preprocessing")
        manifest["steps"]["validate"] = step_validate(config, raw_dir, preprocessed_dir)
    
    if "featurize" in steps:
        logger.info("=" * 60)
        logger.info("STEP: Computing molecular features")
        manifest["steps"]["featurize"] = step_featurize(
            config, preprocessed_dir, features_dir
        )
    
    if "graphs" in steps:
        logger.info("=" * 60)
        logger.info("STEP: Creating PyG graphs")
        manifest["steps"]["graphs"] = step_create_graphs(
            config, features_dir, graphs_dir
        )
    
    # Save manifest
    manifest_path = data_dir / "pipeline_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Pipeline manifest saved to {manifest_path}")
    
    logger.info("=" * 60)
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
