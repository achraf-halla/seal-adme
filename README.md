# SEAL-ADME: Fragment-Aware Molecular Property Prediction

Learning Molecular Representations for ADME Prediction and Interpretability: A Case Study on Aurora Kinase Inhibitors

**M.Sc. Thesis - Bielefeld University, 2026**

## Overview

This repository implements an extended SEAL (Substructure-Explainable Active Learning) framework for predicting ADME (Absorption, Distribution, Metabolism, Excretion) properties of drug-like molecules. The framework provides:

- **Fragment-aware graph neural networks** using BRICS-based molecular decomposition
- **Multiple GNN backbones** (GCN, GIN) with separate intra/inter-fragment message passing
- **Multitask learning** for simultaneous prediction across ADME endpoints
- **Built-in interpretability** through fragment-level attributions

## Key Features

### SEAL-GIN Architecture
Extension of the original SEAL framework (Musial et al., 2025) with:
- Graph Isomorphism Network (GIN) backbone for increased expressiveness
- Separate MLPs for intra-fragment and inter-fragment message passing
- Decoupled encoder/head design for transfer learning

### Supported Endpoints
- Aurora A/B kinase potency (ChEMBL)
- Aqueous solubility (TDC: AqSolDB)
- Caco-2 permeability (TDC)
- Half-life (TDC: Obach)

## Installation

```bash
# Clone the repository
git clone https://github.com/username/seal-adme.git
cd seal-adme

# Create conda environment (recommended)
conda create -n seal-adme python=3.10
conda activate seal-adme

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric

# Install other dependencies
pip install -r requirements.txt
```

## Project Structure

```
seal-adme/
├── configs/
│   ├── data_config.yaml       # Data processing configuration
│   └── model_config.yaml      # Model and training configuration
├── src/
│   ├── data/
│   │   ├── constants.py       # Shared constants
│   │   ├── loaders.py         # TDC and ChEMBL data loading
│   │   ├── preprocessing.py   # SMILES validation/deduplication
│   │   ├── featurization.py   # Molecular descriptors/fingerprints
│   │   ├── fragmentation.py   # BRICS decomposition
│   │   └── graph_featurizer.py  # PyG graph creation
│   ├── models/
│   │   ├── layers.py          # SEALConv, GINConv layers
│   │   ├── encoder.py         # FragmentAwareEncoder
│   │   └── seal.py            # PretrainModel, RegressionModel
│   ├── training/
│   │   ├── datasets.py        # Dataset classes and samplers
│   │   └── trainer.py         # PretrainTrainer, RegressionTrainer
│   └── evaluation/
│       ├── explanations.py    # Fragment attribution extraction
│       └── visualization.py   # Molecule visualization
├── scripts/
│   ├── prepare_data.py        # Data preparation pipeline
│   └── train.py               # Training script
├── notebooks/                  # Analysis notebooks
└── data/                       # Data directory (not tracked)
```

## Usage

### 1. Data Preparation

```bash
# Run complete data pipeline
python scripts/prepare_data.py --config configs/data_config.yaml

# Or run specific steps
python scripts/prepare_data.py --steps load_tdc,validate,featurize,graphs
```

### 2. Model Training

```bash
# Pretraining on classification tasks
python scripts/train.py \
    --mode pretrain \
    --graph-dir data/graphs/pretrain \
    --train-meta data/pretrain_train.parquet \
    --valid-meta data/pretrain_valid.parquet \
    --output-dir results/

# Fine-tuning on regression tasks
python scripts/train.py \
    --mode finetune \
    --graph-dir data/graphs \
    --encoder-path results/pretrain/checkpoints/pretrained_encoder.pt \
    --output-dir results/ \
    --extract-explanations \
    --visualize

# Train from scratch (no pretraining)
python scripts/train.py \
    --mode finetune \
    --graph-dir data/graphs \
    --from-scratch \
    --output-dir results/
```

### 3. Using the Python API

```python
from src.data import GraphFeaturizer, canonicalize_smiles
from src.models import build_model, load_encoder
from src.training import load_task_datasets, RegressionTrainer
from src.evaluation import extract_explanations, visualize_explanations

# Create graphs from SMILES
featurizer = GraphFeaturizer(store_fragments=True)
graphs = featurizer(df, stats={"mean": 0.0, "std": 1.0})

# Build and train model
model = build_model(
    task_names=["solubility", "caco2"],
    hidden_features=256,
    conv_type="seal",  # or "gin"
    pretrained_encoder_path="encoder.pt"
)

# Extract explanations
explanations = extract_explanations(
    model=model,
    task_name="solubility",
    graphs=test_graphs
)

# Visualize
visualize_explanations(
    explanations,
    output_dir="visualizations/",
    sample_size=10
)
```

## Model Architecture

### Fragment-Aware Message Passing

The key innovation is separate linear transformations for intra-fragment and inter-fragment edges

This enables:
1. **Interpretable attributions**: Fragment contributions sum to the prediction
2. **Regularization**: L1 penalty on inter-fragment weights encourages local explanations
3. **Transfer learning**: Encoder learns general fragment representations

## Data Sources

- **TDC (Therapeutics Data Commons)**: ADME benchmark datasets
  - https://tdcommons.ai/
- **ChEMBL**: Aurora kinase bioactivity data
  - https://www.ebi.ac.uk/chembl/



## References

- Musial et al. (2025). SEAL: Substructure-Explainable Active Learning for Molecular Property Prediction.
- Liu et al. (2022). GraphBP: Generating 3D Molecules by 3D Molecular Generative Models.

## License


