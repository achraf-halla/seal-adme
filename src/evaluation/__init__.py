"""
Evaluation and interpretability utilities for SEAL models.
"""

from .explanations import (
    extract_explanations,
    save_explanations,
    load_explanations,
    aggregate_fragment_importance,
    get_top_fragments,
)

from .visualization import (
    prepare_colors,
    draw_molecule_svg,
    visualize_explanation,
    visualize_explanations,
    create_summary_figure,
)

from .analysis import (
    TASK_LABELS,
    TASK_UNITS,
    load_predictions,
    load_smiles_mapping,
    compute_task_correlations,
    compare_correlations,
    compute_sa_score,
    compute_sa_scores_batch,
    find_pareto_front,
    pareto_rank,
    ParetoAnalyzer,
    compare_models,
    ensemble_predictions,
)

from .inference import (
    predict_task,
    predict_all_tasks,
    evaluate_predictions,
    extract_drug_ids,
    save_predictions_csv,
    InferenceRunner,
    load_graphs_from_directory,
)

__all__ = [
    # Explanations
    "extract_explanations",
    "save_explanations",
    "load_explanations",
    "aggregate_fragment_importance",
    "get_top_fragments",
    # Visualization
    "prepare_colors",
    "draw_molecule_svg",
    "visualize_explanation",
    "visualize_explanations",
    "create_summary_figure",
    # Analysis
    "TASK_LABELS",
    "TASK_UNITS",
    "load_predictions",
    "load_smiles_mapping",
    "compute_task_correlations",
    "compare_correlations",
    "compute_sa_score",
    "compute_sa_scores_batch",
    "find_pareto_front",
    "pareto_rank",
    "ParetoAnalyzer",
    "compare_models",
    "ensemble_predictions",
    # Inference
    "predict_task",
    "predict_all_tasks",
    "evaluate_predictions",
    "extract_drug_ids",
    "save_predictions_csv",
    "InferenceRunner",
    "load_graphs_from_directory",
]
