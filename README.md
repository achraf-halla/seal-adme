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

### SEAL Architecture

Extension of the original SEAL framework (Musial et al., 2025) with:

- Graph Isomorphism Network (GIN) backbone for increased expressiveness
- Separate transformations for intra-fragment and inter-fragment message passing
- Fragment contributions sum directly to predictions (interpretable by design)
- Decoupled encoder/head design for transfer learning

### Supported Endpoints

**Pretraining (Classification):**
- HIA, PGP, Bioavailability, BBB penetration
- CYP450 inhibition and substrate prediction

**Finetuning (Regression):**
- Aqueous solubility (TDC: AqSolDB)
- Caco-2 permeability (TDC)
- Half-life (TDC: Obach)
- Aurora A/B kinase potency (ChEMBL)

## Installation

```bash
# Clone the repository
git clone https://github.com/achraf-halla/seal-adme.git
cd seal-adme

# Create conda environment
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
│   │   ├── splitting.py       # Scaffold-based splitting
│   │   ├── fragmentation.py   # BRICS decomposition
│   │   └── graph_builder.py   # PyG graph creation
│   ├── models/
│   │   ├── layers.py          # SEALConv, SEALGINConv layers
│   │   ├── encoder.py         # Fragment-aware encoders
│   │   └── seal.py            # Multi-task models
│   ├── training/
│   │   ├── datasets.py        # Dataset classes and samplers
│   │   ├── pretrain.py        # Pretraining trainer
│   │   └── finetune.py        # Finetuning trainer
│   └── evaluation/
│       ├── explanations.py    # Fragment attribution extraction
│       └── visualization.py   # Molecule visualization
├── scripts/
│   ├── prepare_data.py        # Data preparation pipeline
│   └── train.py               # Training script
└── data/                      # Data directory (not tracked)
```

## Usage

### 1. Data Preparation

```bash
# Run complete data pipeline
python scripts/prepare_data.py --config configs/data_config.yaml

# Or run specific steps
python scripts/prepare_data.py --steps load_tdc,preprocess,split,graphs
```

### 2. Model Training

```bash
# Pretraining only
python scripts/train.py pretrain --config configs/model_config.yaml

# Finetuning with pretrained encoder
python scripts/train.py finetune --config configs/model_config.yaml \
    --encoder outputs/pretrain/checkpoints/pretrained_encoder.pt

# Full pipeline (pretrain + finetune)
python scripts/train.py all --config configs/model_config.yaml
```

### 3. Python API

```python
from src.data import GraphBuilder, create_scaffold_split
from src.models import build_finetune_model
from src.training import FinetuneTrainer
from src.evaluation import extract_explanations, visualize_explanations

# Build graphs from SMILES
builder = GraphBuilder(store_fragments=True)
graphs = builder.build_from_dataframe(df)

# Create model
model = build_finetune_model(
    task_names=['solubility_aqsoldb', 'caco2_wang'],
    encoder_type='gin',
    encoder_checkpoint='pretrained_encoder.pt'
)

# Train
trainer = FinetuneTrainer(model, task_datasets)
results = trainer.train(epochs=150)

# Extract and visualize explanations
explanations = extract_explanations(model, 'solubility_aqsoldb', test_graphs)
visualize_explanations('solubility_aqsoldb', explanations, sample_size=10)
```

## Model Architecture

### Fragment-Aware Message Passing

The key innovation is separate linear transformations for intra-fragment and inter-fragment edges:

```
h_i^{(l+1)} = σ(W_in · Σ_{j∈N_in(i)} h_j^{(l)} + W_out · Σ_{j∈N_out(i)} h_j^{(l)} + W_root · h_i^{(l)})
```

Where:
- `N_in(i)`: Neighbors within same fragment
- `N_out(i)`: Neighbors in different fragments
- `W_in`, `W_out`, `W_root`: Learnable weight matrices

This enables:
1. **Interpretable attributions**: Fragment contributions sum to the prediction
2. **Regularization**: L1 penalty on W_out encourages local explanations
3. **Transfer learning**: Encoder learns general fragment representations

## Configuration

### Encoder Types

**GCN (default):**
```yaml
model:
  encoder_type: gcn
  hidden_features: 256
  num_layers: 4
```

**GIN (more expressive):**
```yaml
model:
  encoder_type: gin
  hidden_features: 256
  num_layers: 4
  train_eps: false
```

### Training Settings

See `configs/model_config.yaml` for full configuration options.

## Data Sources

- **TDC (Therapeutics Data Commons)**: ADME benchmark datasets
  - https://tdcommons.ai/
- **ChEMBL**: Aurora kinase bioactivity data
  - https://www.ebi.ac.uk/chembl/

## References

- Musial et al. (2025). SEAL: Substructure-Explainable Active Learning for Molecular Property Prediction.

## License

MIT License
