"""
SEAL-ADME Explanations Module.

This module provides tools for extracting and visualizing
fragment-level explanations from trained SEAL models.

Extraction:
    MoleculeExplanation: Container for explanation data
    extract_explanations: Extract explanations from model
    load_explanations: Load explanations from file
    extract_all_task_explanations: Extract for all tasks

Visualization:
    visualize_explanation: Visualize single explanation
    visualize_explanations: Visualize multiple explanations
    draw_molecule_svg: Draw molecule with colored atoms
    create_summary_figure: Create multi-molecule summary
"""

from .extract import (
    MoleculeExplanation,
    extract_explanations,
    load_explanations,
    extract_all_task_explanations,
)

from .visualize import (
    prepare_atom_colors,
    save_colorbar,
    draw_molecule_svg,
    visualize_explanation,
    visualize_explanations,
    create_summary_figure,
)


__all__ = [
    # Extraction
    "MoleculeExplanation",
    "extract_explanations",
    "load_explanations",
    "extract_all_task_explanations",
    # Visualization
    "prepare_atom_colors",
    "save_colorbar",
    "draw_molecule_svg",
    "visualize_explanation",
    "visualize_explanations",
    "create_summary_figure",
]
