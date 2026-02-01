# SEAL-ADME: Fragment-Aware Molecular Property Prediction

A PyTorch implementation of the SEAL (Substructure-Explainable Active Learning) framework for interpretable ADME property prediction. This framework uses BRICS-based molecular fragmentation combined with fragment-aware graph neural networks to provide **additive, interpretable predictions** where each fragment's contribution to the final prediction can be directly quantified.

## Key Features

- **BRICS-based fragmentation** - Chemically meaningful molecular decomposition
- **Fragment-aware message passing** - Separate intra/inter-fragment communication
- **Additive predictions** - Fragment contributions sum exactly to the final prediction
- **Multi-task learning** - Shared encoder enables cross-task fragment analysis
- **Interpretable explanations** - Identify which fragments drive predictions

## Why Shared Encoder + Fragment Contributions?

The power of SEAL lies in combining a **shared encoder** across tasks with **additive fragment contributions**:

### 1. Per-Task Fragment Attribution
Each fragment's contribution to a prediction is directly interpretable:
```
Prediction = Σ fragment_contributions
```
For a molecule with 3 fragments:
```
Caco2_pred = frag1_contrib + frag2_contrib + frag3_contrib
           = 0.8 + (-0.3) + 0.2 = 0.7
```

### 2. Cross-Task Fragment Analysis
Because all tasks share the same encoder, you can compare how the **same fragment** affects **different properties**:

| Fragment | Caco2 (Permeability) | Solubility | Lipophilicity |
|----------|---------------------|------------|---------------|
| Phenyl ring | +0.4 | -0.3 | +0.5 |
| Hydroxyl | -0.2 | +0.6 | -0.4 |
| Amide | +0.1 | +0.2 | -0.1 |

This reveals:
- **Synergistic fragments**: Contribute positively to multiple properties
- **Antagonistic fragments**: Improve one property but harm another
- **Property-specific fragments**: Strong effect on one task, neutral on others

### 3. Multi-Objective Drug Design
Identify fragments that optimize multiple ADME properties simultaneously:
```python
# Find fragments that improve both permeability AND solubility
synergistic = fragments.query("caco2_contrib > 0 and solubility_contrib > 0")

# Find permeability-solubility tradeoffs
tradeoffs = fragments.query("caco2_contrib > 0 and solubility_contrib < 0")
```

### 4. Fragment Importance Across Chemical Series
Track how a specific fragment (e.g., fluorine substitution) affects properties across your compound library.

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

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data
```bash
python scripts/prepare_data.py --data-dir data
```

### 2. Train
```bash
# Full pipeline: pretrain (classification) → finetune (regression)
python scripts/train.py all --data-dir data --output-dir outputs

# Or separately:
python scripts/train.py pretrain --data-dir data --output-dir outputs
python scripts/train.py finetune --data-dir data \
    --encoder outputs/pretrain/pretrained_encoder.pt
```

### 3. Inference & Explanations
```bash
python scripts/inference.py \
    --checkpoint outputs/finetune/Caco2_Wang/checkpoints/best_model.pt \
    --data-dir data \
    --output-dir results \
    --visualize
```

## Project Structure

```
seal-adme/
├── configs/
│   ├── data_config.yaml       # Task lists, splitting config
│   └── model_config.yaml      # Architecture, training hyperparams
├── scripts/
│   ├── prepare_data.py        # Data preparation pipeline
│   ├── train.py               # Training CLI (pretrain/finetune)
│   └── inference.py           # Inference & explanation extraction
├── src/
│   ├── data/
│   │   ├── constants.py       # Task definitions
│   │   ├── loaders.py         # TDC data loading
│   │   └── graph_featurizer.py # BRICS fragmentation & graph construction
│   ├── models/
│   │   ├── layers.py          # SEALConv, SEALGINConv layers
│   │   ├── encoder.py         # Fragment-aware GCN/GIN encoders
│   │   └── seal.py            # Multi-task models
│   ├── training/
│   │   ├── datasets.py        # Dataset classes & data loading
│   │   ├── pretrain.py        # Multi-task classification trainer
│   │   └── finetune.py        # Multi-task regression trainer
│   └── evaluation/
│       ├── explanations.py    # Fragment contribution extraction
│       └── visualization.py   # Molecule rendering with RDKit
└── requirements.txt
```

## Data Pipeline

### Pretraining Tasks (Classification)
10 ADME classification tasks from TDC - **no splitting** (all data used):

| Task | Description | Samples |
|------|-------------|---------|
| hia_hou | Human Intestinal Absorption | 578 |
| pgp_broccatelli | P-glycoprotein Inhibition | 1,218 |
| bioavailability_ma | Bioavailability | 640 |
| bbb_martins | Blood-Brain Barrier | 2,030 |
| cyp2d6_veith | CYP2D6 Inhibition | 13,130 |
| cyp3a4_veith | CYP3A4 Inhibition | 12,328 |
| cyp2c9_veith | CYP2C9 Inhibition | 12,092 |
| cyp2d6_substrate_carbonmangels | CYP2D6 Substrate | 667 |
| cyp3a4_substrate_carbonmangels | CYP3A4 Substrate | 670 |
| cyp2c9_substrate_carbonmangels | CYP2C9 Substrate | 669 |

### Finetuning Tasks (Regression)
3 ADME regression tasks - **TDC scaffold split** (70/10/20):

| Task | Description | Unit |
|------|-------------|------|
| Caco2_Wang | Caco-2 Permeability | log cm/s |
| Solubility_AqSolDB | Aqueous Solubility | log mol/L |
| Lipophilicity_AstraZeneca | Lipophilicity (LogD) | LogD |

Labels are normalized to mean=0, std=1 during training.

## Model Architecture

```
                    Input Molecule
                          │
                          ▼
                ┌─────────────────────┐
                │  BRICS Fragmentation │
                │  → Fragment membership matrix (s) │
                │  → Edge mask (intra/inter)        │
                └─────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │      Shared Fragment-Aware Encoder   │
        │  ┌─────────────────────────────────┐ │
        │  │ SEALConv / SEALGINConv Layers   │ │
        │  │ • lin_neighbours (intra-frag)   │ │
        │  │ • lin_outside (inter-frag)      │ │
        │  │ • L1 regularization on outside  │ │
        │  └─────────────────────────────────┘ │
        │               │                      │
        │               ▼                      │
        │  ┌─────────────────────────────────┐ │
        │  │ Fragment Pooling: s.T @ x       │ │
        │  │ → Fragment embeddings [B, K, H] │ │
        │  └─────────────────────────────────┘ │
        └─────────────────────────────────────┘
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
      ┌──────────┐  ┌──────────┐  ┌──────────┐
      │ Caco2    │  │Solubility│  │Lipophil. │
      │ Head     │  │ Head     │  │ Head     │
      │ (Linear) │  │ (Linear) │  │ (Linear) │
      └──────────┘  └──────────┘  └──────────┘
            │             │             │
            ▼             ▼             ▼
      Fragment      Fragment      Fragment
      Contribs      Contribs      Contribs
            │             │             │
            ▼             ▼             ▼
         Σ → Pred      Σ → Pred      Σ → Pred
```

### Key Design Choices

1. **Separate intra/inter-fragment weights**: Encourages the model to learn fragment-specific representations
2. **L1 regularization on inter-fragment weights**: Promotes interpretability by limiting cross-fragment information flow
3. **Shared encoder**: All tasks use the same fragment representations, enabling cross-task analysis
4. **Additive contributions**: Final prediction = sum of fragment contributions (no non-linear aggregation)

## Graph Structure

Each molecular graph contains:

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `x` | [N, 25] | Atom features (type, charge, hybridization, etc.) |
| `edge_index` | [2, E] | Bond connectivity |
| `s` | [N, K] | Fragment membership matrix (one-hot) |
| `mask` | [E] | Edge mask (1=intra-fragment, 0=inter-fragment) |
| `y` | [1] | Target value (normalized for regression) |
| `task_name` | str | Task identifier |
| `smiles` | str | Original SMILES |
| `fragment_smiles` | list | SMILES of each fragment |
| `fragment_atom_lists` | list | Atom indices per fragment |

## Usage Examples

### Training with Custom Configuration
```bash
# GIN encoder with larger hidden dimension
python scripts/train.py all \
    --data-dir data \
    --output-dir outputs \
    --encoder-type gin \
    --hidden-dim 512 \
    --num-layers 6 \
    --pretrain-epochs 100 \
    --finetune-epochs 200
```

### Extract Explanations
```python
from src.evaluation import extract_explanations_for_task, visualize_task_explanations
from src.training import load_finetune_datasets

# Load model and data
datasets = load_finetune_datasets("data/graphs")
test_graphs = datasets['Caco2_Wang'].test

# Extract explanations
explanations = extract_explanations_for_task(
    model=model,
    task_name="Caco2_Wang",
    graphs=test_graphs,
    device="cuda"
)

# Each explanation contains:
# - cluster_contribs: Fragment contributions (sum to prediction)
# - fragment_smiles: SMILES of each fragment
# - additivity_ok: Verification that contributions sum correctly

# Visualize
visualize_task_explanations(
    task_name="Caco2_Wang",
    explanations=explanations,
    output_dir="visualizations",
    sample_size=20
)
```

### Cross-Task Fragment Analysis
```python
import pandas as pd
import numpy as np

def analyze_fragments_across_tasks(model, graphs, task_names):
    """Compare fragment contributions across multiple tasks."""
    all_contribs = []
    
    for task_name in task_names:
        explanations = extract_explanations_for_task(model, task_name, graphs)
        
        for expl in explanations:
            for i, (frag_smiles, contrib) in enumerate(
                zip(expl['fragment_smiles'], expl['cluster_contribs'])
            ):
                all_contribs.append({
                    'mol_idx': expl['index'],
                    'fragment': frag_smiles,
                    'task': task_name,
                    'contribution': float(contrib)
                })
    
    df = pd.DataFrame(all_contribs)
    
    # Pivot to compare fragments across tasks
    pivot = df.pivot_table(
        index='fragment', 
        columns='task', 
        values='contribution',
        aggfunc='mean'
    )
    
    return pivot

# Find synergistic fragments (positive for all tasks)
pivot = analyze_fragments_across_tasks(model, test_graphs, 
    ['Caco2_Wang', 'Solubility_AqSolDB', 'Lipophilicity_AstraZeneca'])

synergistic = pivot[(pivot > 0).all(axis=1)]
antagonistic = pivot[(pivot.iloc[:, 0] > 0) & (pivot.iloc[:, 1] < 0)]
```

## Output Files

After training and inference:
```
outputs/
├── pretrain/
│   ├── pretrained_encoder.pt      # Shared encoder weights
│   ├── best_checkpoint.pt         # Full model checkpoint
│   └── training_results.json      # Metrics history
├── finetune/
│   ├── overall_results.json       # Summary metrics
│   └── {task_name}/
│       ├── checkpoints/
│       │   └── best_model.pt
│       ├── history.json
│       ├── y_train.npy            # Ground truth
│       ├── y_pred_train.npy       # Predictions
│       └── results.json           # Task metrics

inference_results/
├── summary.json
└── {task_name}/
    ├── predictions_test.csv       # Ranked predictions
    ├── metrics_test.json          # Spearman, RMSE, Pearson
    ├── explanations_test.pt       # Fragment contributions
    └── visualizations_test/
        ├── explanation_0.svg      # Molecule with highlighted atoms
        └── explanation_0.colorbar.png
```

## Configuration

### configs/data_config.yaml
```yaml
tdc:
  pretrain_tasks:
    - hia_hou
    - pgp_broccatelli
    - bioavailability_ma
    - bbb_martins
    - cyp2d6_veith
    - cyp3a4_veith
    - cyp2c9_veith
    - cyp2d6_substrate_carbonmangels
    - cyp3a4_substrate_carbonmangels
    - cyp2c9_substrate_carbonmangels
  finetune_tasks:
    - Caco2_Wang
    - Solubility_AqSolDB
    - Lipophilicity_AstraZeneca

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
  encoder_type: gcn          # gcn or gin
  input_features: 25
  hidden_features: 256
  num_layers: 4
  dropout: 0.1

regularization:
  encoder: 1.0e-4            # L1 on inter-fragment weights
  contribution: 0.5          # L1 on fragment contributions

pretrain:
  epochs: 50
  batch_size: 64
  learning_rate: 1.0e-3

finetune:
  epochs: 120
  batch_size: 64
  learning_rate: 3.0e-4
  patience: 25
  task_sampling: round_robin  # or proportional
```

## References

- Musial et al. (2025). SEAL: Substructure-Explainable Active Learning for Molecular Property Prediction.
- Therapeutics Data Commons (TDC): https://tdcommons.ai/
- BRICS Fragmentation: Degen et al. (2008). On the Art of Compiling and Using 'Drug-Like' Chemical Fragment Spaces.

## License

MIT License

## Citation

If you use this code, please cite:
```bibtex
@article{musial2025seal,
  title={SEAL: Substructure-Explainable Active Learning for Molecular Property Prediction},
  author={Musial, et al.},
  year={2025}
}
```
