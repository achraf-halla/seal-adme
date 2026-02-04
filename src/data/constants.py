"""
Constants for SEAL-ADME data processing.
"""

# TDC pretraining tasks (classification) - no splitting needed
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

# TDC finetuning tasks (regression) - use TDC scaffold split
FINETUNE_TASKS = [
    'Caco2_Wang',
    'Solubility_AqSolDB', 
    'Lipophilicity_AstraZeneca',
]

# Atom type vocabulary for node features
ATOM_TYPE_SET = ["C", "N", "O", "F", "Cl", "Br", "P", "S", "B", "I", "Unk"]

# Hybridization types
HYBRIDIZATION_SET = ["SP", "SP2", "SP3", "Other"]

# Halogen atomic numbers (for feature computation)
HALOGEN_ATOMIC_NUMS = {9, 17, 35, 53}  # F, Cl, Br, I

# Pauling electronegativity values
PAULING_ELECTRONEGATIVITY = {
    "H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
    "P": 2.19, "S": 2.58, "Cl": 3.16, "Br": 2.96, "I": 2.66, "B": 2.04
}
DEFAULT_ELECTRONEGATIVITY = 2.50

# Standard column names
STANDARD_COLUMNS = ['Drug_ID', 'Drug', 'Y', 'task_name']
META_COLUMNS = ['Drug_ID', 'Drug', 'task_name', 'split']

# Default splitting configuration
DEFAULT_SPLIT_FRACTIONS = [0.7, 0.1, 0.2]  # train, valid, test

# Default model hyperparameters
DEFAULT_INPUT_FEATURES = 25
DEFAULT_HIDDEN_DIM = 256
DEFAULT_NUM_LAYERS = 4
DEFAULT_DROPOUT = 0.1

