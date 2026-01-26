"""
Constants for molecular featurization and data processing.
"""

# TDC task definitions
PRETRAIN_TASKS = [
    'hia_hou', 'pgp_broccatelli', 'bioavailability_ma', 'lipophilicity_astrazeneca',
    'bbb_martins', 'ppbr_az', 'vdss_lombardo', 'cyp2d6_veith', 'cyp3a4_veith',
    'cyp2c9_veith', 'cyp2d6_substrate_carbonmangels', 'cyp3a4_substrate_carbonmangels',
    'cyp2c9_substrate_carbonmangels', 'clearance_microsome_az', 'clearance_hepatocyte_az'
]

FINETUNE_TASKS = ['solubility_aqsoldb', 'caco2_wang', 'half_life_obach']

# Atom featurization
DEFAULT_ATOM_TYPE_SET = ["C", "N", "O", "F", "Cl", "Br", "P", "S", "B", "I", "Unk"]
DEFAULT_HYBRIDIZATION_SET = ["SP", "SP2", "SP3", "Other"]

# Pauling electronegativity values
PAULING_ELECTRONEGATIVITY = {
    "H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
    "P": 2.19, "S": 2.58, "Cl": 3.16, "Br": 2.96, "I": 2.66, "B": 2.04
}
DEFAULT_ELECTRONEGATIVITY = 2.50

# Halogen atomic numbers for BRICS decomposition
HALOGEN_ATOMIC_NUMS = {9, 17, 35, 53, 85, 117}

# Morgan fingerprint defaults
FP_NBITS = 2048
FP_RADIUS = 2

# Standard column names
STANDARD_COLUMNS = [
    'Drug_ID', 'original_smiles', 'Y', 'task_name', 'source', 'task'
]

GRAPH_COLUMNS = [
    'Drug_ID', 'task_name', 'source', 'task', 'canonical_smiles', 'Y'
]

# Aurora kinase target search terms
AURORA_SEARCH_TERMS = ["AURKA", "AURKB", "AURKC", "Aurora kinase"]
