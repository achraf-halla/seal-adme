# SEAL-ADME v2: Fragment-Aware Molecular Property Prediction

Simplified implementation of the SEAL (Substructure-Explainable Active Learning) framework for ADME property prediction.

## Features

- **BRICS-based fragmentation** for interpretable molecular representations
- **Multi-task pretraining** on 10 classification tasks
- **Single-task finetuning** on 3 regression tasks
- **Fragment-level explanations** - contributions sum to prediction

## Installation

```bash
# Clone repository
git clone https://github.com/achraf-halla/seal-adme.git
cd seal-adme

# Create environment
conda create -n seal-adme python=3.10
conda activate seal-adme

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric

# Install other dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data

```bash
python scripts/prepare_data.py --data-dir data
```

Creates:
- `data/graphs/pretrain/` - 10 classification tasks (no split)
- `data/graphs/finetune/{task}/{split}/` - 3 regression tasks with scaffold split

### 2. Train

```bash
# Full pipeline: pretrain + finetune
python scripts/train.py all --data-dir data --output-dir outputs

# Or separately:
python scripts/train.py pretrain --data-dir data --output-dir outputs
python scripts/train.py finetune --data-dir data --encoder outputs/pretrain/checkpoints/pretrained_encoder.pt
```

### 3. Use in Python

```python
from src.training import load_pretrain_dataset, load_finetune_datasets
from src.models import build_pretrain_model, build_finetune_model

# Load data
pretrain = load_pretrain_dataset("data/graphs")
finetune = load_finetune_datasets("data/graphs")

# Build model
model = build_finetune_model(
    task_names=["Caco2_Wang"],
    encoder_checkpoint="outputs/pretrain/checkpoints/pretrained_encoder.pt",
    encoder_type="gcn",
    hidden_features=256
)
```

## Data Pipeline

### Pretraining Tasks (Classification)
| Task | Description |
|------|-------------|
| hia_hou | Human Intestinal Absorption |
| pgp_broccatelli | P-glycoprotein Inhibition |
| bioavailability_ma | Bioavailability |
| bbb_martins | Blood-Brain Barrier |
| cyp2d6_veith | CYP2D6 Inhibition |
| cyp3a4_veith | CYP3A4 Inhibition |
| cyp2c9_veith | CYP2C9 Inhibition |
| cyp2d6_substrate_carbonmangels | CYP2D6 Substrate |
| cyp3a4_substrate_carbonmangels | CYP3A4 Substrate |
| cyp2c9_substrate_carbonmangels | CYP2C9 Substrate |

**No splitting** - all data used for training.

### Finetuning Tasks (Regression)
| Task | Description | Split |
|------|-------------|-------|
| Caco2_Wang | Caco-2 Permeability | 70/10/20 |
| Solubility_AqSolDB | Aqueous Solubility | 70/10/20 |
| Lipophilicity_AstraZeneca | LogD | 70/10/20 |

**TDC scaffold split** with normalized labels (mean=0, std=1).

## Graph Structure

Each molecular graph contains:

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `x` | [N, 25] | Atom features |
| `edge_index` | [2, E] | Bond connectivity |
| `s` | [N, K] | Fragment membership matrix |
| `mask` | [E] | Edge mask (1=intra, 0=inter fragment) |
| `y` | [1] | Target value |
| `task_name` | str | Task identifier |
| `smiles` | str | Original SMILES |
| `fragment_smiles` | list | Fragment SMILES for interpretability |

## Model Architecture

```
Input Graph
    │
    ▼
┌─────────────────────────────┐
│  Fragment-Aware Encoder     │
│  (GCN or GIN with SEAL)     │
│  - Separate intra/inter     │
│    fragment message passing │
└─────────────────────────────┘
    │
    ▼
Fragment Embeddings [B, K, H]
    │
    ▼
┌─────────────────────────────┐
│  Task Head (Linear)         │
│  fragment_contributions     │
└─────────────────────────────┘
    │
    ▼
Sum over fragments → Prediction
```

## Configuration

### configs/data_config.yaml
```yaml
tdc:
  pretrain_tasks: [hia_hou, pgp_broccatelli, ...]
  finetune_tasks: [Caco2_Wang, Solubility_AqSolDB, Lipophilicity_AstraZeneca]

splitting:
  seed: 42
  method: scaffold
  fractions: [0.7, 0.1, 0.2]

normalization:
  enabled: true
  compute_from: train
```

### configs/model_config.yaml
```yaml
model:
  encoder_type: gcn
  hidden_features: 256
  num_layers: 4
  dropout: 0.1

pretrain:
  epochs: 50
  batch_size: 64
  learning_rate: 1.0e-3

finetune:
  epochs: 150
  batch_size: 64
  learning_rate: 3.0e-4
  patience: 20
```

## Project Structure

```
seal-adme/
├── configs/
│   ├── data_config.yaml
│   └── model_config.yaml
├── scripts/
│   ├── prepare_data.py      # Data preparation
│   └── train.py             # Training CLI
├── src/
│   ├── data/
│   │   ├── constants.py     # Task lists
│   │   ├── loaders.py       # TDC data loading
│   │   └── graph_featurizer.py  # BRICS graph construction
│   ├── models/
│   │   ├── layers.py        # SEALConv layers
│   │   ├── encoder.py       # GCN/GIN encoders
│   │   └── seal.py          # Multi-task models
│   ├── training/
│   │   ├── datasets.py      # Dataset classes
│   │   ├── pretrain.py      # Pretraining trainer
│   │   └── finetune.py      # Finetuning trainer
│   └── evaluation/
│       ├── explanations.py  # Fragment attribution
│       └── visualization.py # Molecule visualization
└── requirements.txt
```

## References

- Musial et al. (2025). SEAL: Substructure-Explainable Active Learning for Molecular Property Prediction.
- TDC: https://tdcommons.ai/
