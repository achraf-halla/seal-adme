"""
Shared constants for SEAL-ADME data processing.

This module centralizes configuration values used across the data pipeline.
"""

from typing import List, Dict

# =============================================================================
# Column names (standardized schema)
# =============================================================================
STANDARD_COLUMNS = [
    "Drug_ID",
    "original_smiles", 
    "Y",
    "task_name",
    "source",
    "task",
    "canonical_smiles",
]

REQUIRED_COLUMNS = ["Drug_ID", "original_smiles", "Y"]

# =============================================================================
# Task definitions for TDC datasets
# =============================================================================
PRETRAIN_TASKS: List[str] = [
    # Absorption
    "hia_hou",
    "pgp_broccatelli",
    "bioavailability_ma",
    # Distribution
    "bbb_martins",
    "ppbr_az",
    "vdss_lombardo",
    # Metabolism (CYP enzymes)
    "cyp2d6_veith",
    "cyp3a4_veith",
    "cyp2c9_veith",
    "cyp2d6_substrate_carbonmangels",
    "cyp3a4_substrate_carbonmangels",
    "cyp2c9_substrate_carbonmangels",
    # Excretion
    "clearance_microsome_az",
    "clearance_hepatocyte_az",
    # Physicochemical
    "lipophilicity_astrazeneca",
]

FINETUNE_TASKS: List[str] = [
    "solubility_aqsoldb",
    "caco2_wang",
    "half_life_obach",
]

# Classification vs regression task mapping
CLASSIFICATION_TASKS: List[str] = [
    "hia_hou",
    "pgp_broccatelli",
    "bioavailability_ma",
    "bbb_martins",
    "cyp2d6_veith",
    "cyp3a4_veith",
    "cyp2c9_veith",
    "cyp2d6_substrate_carbonmangels",
    "cyp3a4_substrate_carbonmangels",
    "cyp2c9_substrate_carbonmangels",
]

REGRESSION_TASKS: List[str] = [
    "lipophilicity_astrazeneca",
    "ppbr_az",
    "vdss_lombardo",
    "clearance_microsome_az",
    "clearance_hepatocyte_az",
    "solubility_aqsoldb",
    "caco2_wang",
    "half_life_obach",
]

# =============================================================================
# Atom featurization constants
# =============================================================================
DEFAULT_ATOM_TYPE_SET: List[str] = [
    "C", "N", "O", "F", "Cl", "Br", "P", "S", "B", "I", "Unk"
]

DEFAULT_HYBRIDIZATION_SET: List[str] = [
    "SP", "SP2", "SP3", "Other"
]

# Pauling electronegativity values
PAULING_ELECTRONEGATIVITY: Dict[str, float] = {
    "H": 2.20,
    "C": 2.55,
    "N": 3.04,
    "O": 3.44,
    "F": 3.98,
    "P": 2.19,
    "S": 2.58,
    "Cl": 3.16,
    "Br": 2.96,
    "I": 2.66,
    "B": 2.04,
}
DEFAULT_ELECTRONEGATIVITY: float = 2.50

# =============================================================================
# ChEMBL Aurora kinase targets
# =============================================================================
AURORA_TARGET_SYNONYMS: List[str] = [
    "AURKA",
    "AURKB", 
    "AURKC",
    "Aurora kinase",
]

AURORA_ACTIVITY_TYPES: List[str] = [
    "IC50",
    "Ki",
]

# =============================================================================
# Data quality thresholds
# =============================================================================
MIN_CLASSIFICATION_THRESHOLD: int = 10  # Max unique values to be considered classification
SMILES_ENCODINGS: List[str] = ["utf-8", "utf-8-sig", "latin1"]

