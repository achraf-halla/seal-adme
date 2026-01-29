"""
Constants for SEAL-ADME data processing and model configuration.
"""

# TDC pretraining tasks (classification only)
PRETRAIN_TASKS = [
    'hia_hou',
    'pgp_broccatelli',
    'bioavailability_ma',
    'bbb_martins',
    'cyp2d6_veith',
    'cyp3a4_veith',
    'cyp2c9_veith',
    'cyp2d6_substrate_carbonmangels',
    'cyp3a4_substrate_carbonmangels',
    'cyp2c9_substrate_carbonmangels',
]

# TDC finetuning tasks (regression)
FINETUNE_TASKS = [
    'solubility_aqsoldb',
    'caco2_wang',
    'half_life_obach',
]

# Atom type vocabulary for node features
ATOM_TYPE_SET = ["C", "N", "O", "F", "Cl", "Br", "P", "S", "B", "I", "Unk"]

# Hybridization types
HYBRIDIZATION_SET = ["SP", "SP2", "SP3", "Other"]

# Pauling electronegativity values
PAULING_ELECTRONEGATIVITY = {
    "H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
    "P": 2.19, "S": 2.58, "Cl": 3.16, "Br": 2.96, "I": 2.66, "B": 2.04
}
DEFAULT_ELECTRONEGATIVITY = 2.50

# Halogen atomic numbers for BRICS decomposition
HALOGEN_ATOMIC_NUMS = {9, 17, 35, 53, 85, 117}

# Standard column names for data processing
META_COLUMNS = [
    'Drug_ID', 'original_smiles', 'Y', 'task_name', 'source', 'task', 'canonical_smiles'
]

STANDARD_COLUMNS = [
    'Drug_ID', 'original_smiles', 'Y', 'task_name', 'source', 'task'
]

# Aurora kinase target search terms
AURORA_SEARCH_TERMS = ["AURKA", "AURKB", "AURKC", "Aurora kinase"]

# Default split fractions
DEFAULT_SPLIT_FRACTIONS = [0.8, 0.1, 0.1]  # train, valid, test

# Model defaults
DEFAULT_HIDDEN_DIM = 256
DEFAULT_NUM_LAYERS = 4
DEFAULT_DROPOUT = 0.1
DEFAULT_INPUT_FEATURES = 25
