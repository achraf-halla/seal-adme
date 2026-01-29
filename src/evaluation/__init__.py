"""
SEAL-ADME Evaluation Module.

Provides utilities for extracting and visualizing model explanations:
- Fragment-level attribution extraction
- Molecular visualization with importance coloring
"""

from .explanations import (
    extract_explanations,
    extract_explanations_all_tasks,
    load_explanations,
    explanations_to_dataframe,
    get_top_fragments,
)

from .visualization import (
    get_smiles_from_explanation,
    prepare_atom_colors,
    save_colorbar,
    draw_molecule_svg,
    visualize_explanation,
    visualize_explanations,
    visualize_all_tasks,
)

__all__ = [
    # Explanations
    "extract_explanations",
    "extract_explanations_all_tasks",
    "load_explanations",
    "explanations_to_dataframe",
    "get_top_fragments",
    # Visualization
    "get_smiles_from_explanation",
    "prepare_atom_colors",
    "save_colorbar",
    "draw_molecule_svg",
    "visualize_explanation",
    "visualize_explanations",
    "visualize_all_tasks",
]
