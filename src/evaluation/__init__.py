"""
SEAL-ADME Evaluation Module.

Provides:
- Fragment-level explanation extraction
- Molecule visualization with importance highlighting
- Inference utilities for trained models
"""

from .explanations import (
    extract_explanations_for_task,
    predict_task_on_graphs,
    extract_drug_ids_from_graphs,
    save_predictions_csv,
    run_inference_and_save,
    run_inference_all_tasks,
    spearman_scorer,
)

from .visualization import (
    visualize_explanation,
    visualize_task_explanations,
    create_summary_figure,
)

__all__ = [
    # Explanations
    "extract_explanations_for_task",
    "predict_task_on_graphs",
    "extract_drug_ids_from_graphs",
    "save_predictions_csv",
    "run_inference_and_save",
    "run_inference_all_tasks",
    "spearman_scorer",
    # Visualization
    "visualize_explanation",
    "visualize_task_explanations",
    "create_summary_figure",
]
