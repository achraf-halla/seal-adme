# SEAL-ADME: Fragment-Aware Molecular Property Prediction

A PyTorch implementation extending the SEAL (Substructure Explanation via Attribution Learning) framework for interpretable multi-task ADME property prediction.

## Overview

This implementation builds on the **SEAL framework** (Musial et al., 2025), which provides intrinsic interpretability through BRICS-based molecular fragmentation and fragment-aware message passing. Fragment contributions are **additive by design**—they sum exactly to the final prediction, enabling direct quantification of each substructure's impact.

### Original SEAL Framework
The original SEAL implementation introduced:
- BRICS-based molecular decomposition into chemically meaningful fragments
- Fragment-aware GCN layers with separate weights for intra-fragment and inter-fragment edges
- L1 regularization on inter-fragment weights to promote interpretability
- Single-task prediction with additive fragment contributions

### This Implementation: Key Extensions

I extended the framework in two significant ways:

| Aspect | Original SEAL | This Implementation |
|--------|--------------|---------------------|
| **Backbone** | GCN only | GCN + **GIN** with separate MLPs for intra/inter-fragment message passing |
| **Learning** | Single-task | **Multi-task** with shared encoder and task-specific heads |
| **Transfer** | Train from scratch | **Transfer learning** via pretrained encoder |

#### 1. GIN Backbone with Fragment-Aware MLPs
```python
# Original SEAL: Linear transformations only
h_intra = W_intra @ h_neighbor
h_inter = W_inter @ h_neighbor

# This implementation: MLP transformations for GIN
h_intra = MLP_intra(h_neighbor)  # 2-layer MLP
h_inter = MLP_inter(h_neighbor)  # 2-layer MLP with L1 regularization
```
The GIN backbone with MLPs increases representational expressiveness while maintaining the fragment-aware intra/inter separation that enables interpretability.

#### 2. Decoupled Encoder for Multi-Task Learning
```
┌─────────────────────────────────────────────────────────────┐
│                   SHARED ENCODER                             │
│  (Pretrained on 10 classification tasks)                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Fragment-Aware GCN/GIN → Fragment Embeddings [B, K, H] │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
    ┌──────────┐      ┌──────────┐      ┌──────────┐
    │  Caco-2  │      │Solubility│      │Lipophil. │
    │   Head   │      │   Head   │      │   Head   │
    └────┬─────┘      └────┬─────┘      └────┬─────┘
         │                 │                 │
    Σ contribs        Σ contribs        Σ contribs
         │                 │                 │
         ▼                 ▼                 ▼
      Prediction       Prediction       Prediction
```

Decoupling the encoder from task-specific heads enables:
- **Multi-task learning**: Train on multiple endpoints simultaneously
- **Transfer learning**: Pretrain encoder on classification tasks, finetune heads on regression
- **Cross-task analysis**: Compare fragment contributions across different properties

## Why Cross-Task Fragment Analysis Matters

Because all tasks share the same encoder, you can directly compare how the **same fragment** affects **different properties**:

| Fragment | Caco-2 (Permeability) | Solubility | Lipophilicity |
|----------|----------------------|------------|---------------|
| Phenyl ring | +0.4 | -0.3 | +0.5 |
| Hydroxyl | -0.2 | +0.6 | -0.4 |
| Amide | +0.1 | +0.2 | -0.1 |

This reveals:
- **Synergistic fragments**: Hydroxyl improves solubility (+0.6) without catastrophic permeability loss (-0.2)
- **Antagonistic fragments**: Phenyl improves permeability (+0.4) but harms solubility (-0.3)
- **Trade-off analysis**: Quantify the permeability-solubility trade-off for each fragment

### Multi-Objective Drug Design
```python
# Find fragments that improve both permeability AND solubility
synergistic = fragments.query("caco2_contrib > 0 and solubility_contrib > 0")

# Identify permeability-solubility trade-offs
tradeoffs = fragments.query("caco2_contrib > 0 and solubility_contrib < 0")

# Guide optimization: replace antagonistic fragments with synergistic ones
```

## Installation

```bash
git clone https://github.com/achraf-halla/seal-adme.git
cd seal-adme

conda create -n seal-adme python=3.10
conda activate seal-adme

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data
```bash
python scripts/prepare_data.py --data-dir data
```

### 2. Train
```bash
# Full pipeline: pretrain (10 classification tasks) → finetune (3 regression tasks)
python scripts/train.py all --data-dir data --output-dir outputs

# Or use GIN backbone
python scripts/train.py all --data-dir data --output-dir outputs --encoder-type gin
```

### 3. Inference & Explanations
```bash
python scripts/inference.py \
    --checkpoint outputs/finetune/best_checkpoint.pt \
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
│   │   ├── layers.py          # SEALConv (GCN), SEALGINConv (GIN)
│   │   ├── encoder.py         # FragmentAwareGCNEncoder, FragmentAwareGINEncoder
│   │   └── seal.py            # MultiTaskPretrainModel, MultiTaskRegressionModel
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
10 ADME classification tasks from TDC — **no splitting** (all data used for encoder pretraining):

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
3 ADME regression tasks — **TDC scaffold split** (70/10/20):

| Task | Description | Unit |
|------|-------------|------|
| Caco2_Wang | Caco-2 Permeability | log cm/s |
| Solubility_AqSolDB | Aqueous Solubility | log mol/L |
| Lipophilicity_AstraZeneca | Lipophilicity (LogD) | LogD |

Labels are normalized to mean=0, std=1 during training.

## Model Architecture

### SEALConv (GCN Backbone)
```python
class SEALConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        self.lin_neighbours = Linear(in_channels, out_channels)  # Intra-fragment
        self.lin_outside = Linear(in_channels, out_channels)     # Inter-fragment
        self.lin_root = Linear(in_channels, out_channels)        # Self-loop
```

### SEALGINConv (GIN Backbone) — New in This Implementation
```python
class SEALGINConv(MessagePassing):
    def __init__(self, in_channels, out_channels, train_eps=True):
        # 2-layer MLPs instead of linear transforms
        self.mlp_neighbours = Sequential(
            Linear(in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels)
        )
        self.mlp_outside = Sequential(
            Linear(in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels)
        )
        self.eps = Parameter(torch.zeros(1))  # Learnable epsilon
```

### Fragment Aggregation
```python
# Aggregate node features to fragment embeddings
# s: [N, K] fragment membership matrix (one-hot)
# x: [N, H] node features after message passing
x_frag = torch.matmul(s.T, x)  # [K, H] fragment embeddings
```

## Graph Structure

Each molecular graph contains:

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `x` | [N, 25] | Atom features (type, charge, hybridization, etc.) |
| `edge_index` | [2, E] | Bond connectivity |
| `s` | [N, K] | Fragment membership matrix (one-hot) |
| `edge_brics_mask` | [E] | Edge mask (1=intra-fragment, 0=inter-fragment) |
| `y` | [1] | Target value (normalized for regression) |
| `smiles` | str | Original SMILES |
| `fragment_smiles` | list | SMILES of each BRICS fragment |

## Usage Examples

### Extract and Compare Fragment Contributions
```python
from src.evaluation import extract_explanations_for_task
from src.training import load_finetune_datasets

datasets = load_finetune_datasets("data/graphs")
test_graphs = datasets['Caco2_Wang'].test

# Extract for multiple tasks
tasks = ['Caco2_Wang', 'Solubility_AqSolDB', 'Lipophilicity_AstraZeneca']
all_explanations = {}

for task in tasks:
    all_explanations[task] = extract_explanations_for_task(
        model=model,
        task_name=task,
        graphs=test_graphs,
        device="cuda"
    )

# Compare fragment contributions across tasks
mol_idx = 0
print(f"Molecule: {test_graphs[mol_idx].smiles}")
print(f"Fragments: {test_graphs[mol_idx].fragment_smiles}")
print()
for task in tasks:
    expl = all_explanations[task][mol_idx]
    print(f"{task}:")
    for frag, contrib in zip(expl['fragment_smiles'], expl['cluster_contribs']):
        print(f"  {frag}: {contrib:+.3f}")
    print(f"  Prediction: {expl['pred']:.3f}")
```

### Cross-Task Fragment Analysis
```python
import pandas as pd

def build_fragment_contribution_table(model, graphs, task_names):
    """Build a table comparing fragment contributions across tasks."""
    records = []
    
    for task in task_names:
        explanations = extract_explanations_for_task(model, task, graphs)
        for expl in explanations:
            for frag, contrib in zip(expl['fragment_smiles'], expl['cluster_contribs']):
                records.append({
                    'mol_idx': expl['index'],
                    'fragment': frag,
                    'task': task,
                    'contribution': float(contrib)
                })
    
    df = pd.DataFrame(records)
    
    # Average contribution per fragment per task
    pivot = df.pivot_table(
        index='fragment',
        columns='task', 
        values='contribution',
        aggfunc='mean'
    )
    return pivot

# Analyze
pivot = build_fragment_contribution_table(model, test_graphs, tasks)

# Find synergistic fragments (good for all properties)
synergistic = pivot[(pivot > 0).all(axis=1)]
print("Synergistic fragments:")
print(synergistic)

# Find antagonistic fragments (permeability vs solubility trade-off)
antagonistic = pivot[
    (pivot['Caco2_Wang'] > 0.1) & 
    (pivot['Solubility_AqSolDB'] < -0.1)
]
print("\nPermeability-Solubility trade-offs:")
print(antagonistic)
```

## Output Files

```
outputs/
├── pretrain/
│   ├── pretrained_encoder.pt      # Shared encoder (for transfer learning)
│   ├── best_checkpoint.pt         # Full pretrain model
│   └── training_results.json
├── finetune/
│   ├── best_checkpoint.pt         # Best multi-task model
│   ├── overall_results.json
│   └── {task_name}/
│       ├── results.json           # Task metrics
│       ├── y_test.npy             # Ground truth
│       └── y_pred_test.npy        # Predictions

inference_results/
└── {task_name}/
    ├── predictions_test.csv       # Ranked predictions with Drug_ID
    ├── metrics_test.json          # Spearman, Pearson, RMSE
    ├── explanations_test.pt       # Fragment contributions
    └── visualizations_test/
        └── explanation_*.svg      # Molecules with importance highlighting
```

## Configuration

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

- **Musial et al. (2025)**. *Fragment-Wise Interpretability in Graph Neural Networks via Molecule Decomposition and Contribution Analysis.* — Original SEAL framework *https://github.com/gmum/SEAL/tree/main*
- **Therapeutics Data Commons (TDC)**: https://tdcommons.ai/
- **BRICS Fragmentation**: Degen et al. (2008). *On the Art of Compiling and Using 'Drug-Like' Chemical Fragment Spaces.*

## License

MIT License
