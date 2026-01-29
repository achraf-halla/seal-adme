"""
Visualization utilities for SEAL explanations.

Renders molecules with atoms colored by fragment contributions,
enabling visual interpretation of model predictions.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_smiles_from_explanation(
    explanation: Dict,
    smiles_keys: Tuple[str, ...] = ('canonical_smiles', 'Canonical_Smiles', 'smiles', 'SMILES')
) -> Optional[str]:
    """
    Extract SMILES string from explanation dict.
    
    Args:
        explanation: Explanation dict
        smiles_keys: Keys to search for SMILES
        
    Returns:
        SMILES string or None
    """
    # Check metadata
    meta = explanation.get('meta')
    if isinstance(meta, dict):
        for key in smiles_keys:
            if key in meta and meta[key]:
                return meta[key]
    
    # Check top-level
    for key in smiles_keys:
        if key in explanation and explanation[key]:
            return explanation[key]
    
    return None


def prepare_atom_colors(
    values: np.ndarray,
    cmap_name: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center_zero: bool = True
) -> Tuple[List[Tuple[float, float, float]], Dict]:
    """
    Convert importance values to RGB colors.
    
    Args:
        values: Array of importance values per atom
        cmap_name: Matplotlib colormap name
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        center_zero: Whether to center colormap at zero
        
    Returns:
        Tuple of (list of RGB tuples, colormap info dict)
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    vals = np.asarray(values, dtype=float).squeeze()
    
    if vals.size == 0:
        cmap = plt.colormaps.get_cmap(cmap_name)
        norm = mcolors.Normalize(-1.0, 1.0)
        return [], {'vmin': -1.0, 'vmax': 1.0, 'cmap': cmap, 'norm': norm}
    
    # Determine range
    if center_zero:
        if vmin is None or vmax is None:
            max_abs = np.nanmax(np.abs(vals))
            if max_abs == 0 or np.isnan(max_abs):
                max_abs = 1.0
            vmin, vmax = -max_abs, max_abs
    else:
        if vmin is None:
            vmin = np.nanmin(vals)
        if vmax is None:
            vmax = np.nanmax(vals)
    
    # Clip and normalize
    vals_clipped = np.clip(vals, vmin, vmax)
    
    cmap = plt.colormaps.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    rgba = cmap(norm(vals_clipped))
    rgb_list = [tuple(map(float, rgba[i][:3])) for i in range(len(rgba))]
    
    return rgb_list, {'vmin': vmin, 'vmax': vmax, 'cmap': cmap, 'norm': norm}


def save_colorbar(
    path: str,
    cmap,
    norm,
    vmin: float,
    vmax: float,
    label: str = "Contribution",
    figsize: Tuple[float, float] = (4, 0.4),
    dpi: int = 150
):
    """
    Save a standalone colorbar image.
    
    Args:
        path: Output file path
        cmap: Matplotlib colormap
        norm: Matplotlib normalizer
        vmin: Minimum value
        vmax: Maximum value
        label: Colorbar label
        figsize: Figure size
        dpi: Output DPI
    """
    import matplotlib.pyplot as plt
    from matplotlib.colorbar import ColorbarBase
    
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(bottom=0.3, top=0.8, left=0.05, right=0.95)
    
    cb = ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.set_label(label)
    
    ticks = np.linspace(vmin, vmax, 5)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.2f}" for t in ticks])
    
    fig.savefig(str(path), dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def draw_molecule_svg(
    smiles: str,
    atom_colors: Dict[int, Tuple[float, float, float]],
    size: Tuple[int, int] = (400, 400)
) -> str:
    """
    Draw molecule as SVG with highlighted atoms.
    
    Args:
        smiles: SMILES string
        atom_colors: Dict mapping atom index -> RGB tuple
        size: Image size (width, height)
        
    Returns:
        SVG string
    """
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        pass
    
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    opts = drawer.drawOptions()
    opts.padding = 0.02
    opts.bondLineWidth = 1.6
    
    # Build highlight dict
    highlight_atoms = []
    highlight_atom_colors = {}
    n_atoms = mol.GetNumAtoms()
    
    for atom_idx, color in atom_colors.items():
        if isinstance(atom_idx, (int, np.integer)) and 0 <= atom_idx < n_atoms:
            if color is not None and len(color) >= 3:
                highlight_atoms.append(int(atom_idx))
                highlight_atom_colors[int(atom_idx)] = (
                    float(color[0]), float(color[1]), float(color[2])
                )
    
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_atom_colors
    )
    drawer.FinishDrawing()
    
    return drawer.GetDrawingText()


def visualize_explanation(
    explanation: Dict,
    output_dir: str,
    filename_prefix: str = "explanation",
    normalize: bool = True,
    cmap: str = "RdBu_r",
    size: Tuple[int, int] = (400, 400),
    include_colorbar: bool = True
) -> Optional[str]:
    """
    Visualize a single explanation and save to file.
    
    Args:
        explanation: Explanation dict from extract_explanations
        output_dir: Directory to save files
        filename_prefix: Prefix for output files
        normalize: Whether to normalize importance values
        cmap: Colormap name
        size: Image size
        include_colorbar: Whether to save colorbar separately
        
    Returns:
        Path to saved SVG or None if failed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get SMILES
    smiles = get_smiles_from_explanation(explanation)
    if smiles is None:
        logger.warning("No SMILES found in explanation")
        return None
    
    # Get node importance
    node_importance = explanation.get('node_importance')
    if node_importance is None:
        node_importance = explanation.get('node_importance_raw', [])
    
    node_importance = np.asarray(node_importance, dtype=float).squeeze()
    if node_importance.ndim != 1:
        node_importance = node_importance.flatten()
    
    # Normalize if requested
    if normalize and len(node_importance) > 0:
        max_abs = np.nanmax(np.abs(node_importance))
        if max_abs > 0 and not np.isnan(max_abs):
            node_importance = node_importance / max_abs
    
    # Get colors
    rgb_colors, cmap_info = prepare_atom_colors(
        node_importance, cmap_name=cmap, vmin=-1.0, vmax=1.0
    )
    
    # Get molecule
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning(f"Could not parse SMILES: {smiles}")
        return None
    
    n_atoms = mol.GetNumAtoms()
    
    # Extend colors if needed
    if len(rgb_colors) < n_atoms:
        gray = (0.8, 0.8, 0.8)
        rgb_colors = list(rgb_colors) + [gray] * (n_atoms - len(rgb_colors))
    
    # Map node indices to atom indices
    node_to_atom = explanation.get('node_to_atom_map')
    atom_colors = {}
    
    if node_to_atom is not None:
        for node_idx, atom_idx in enumerate(node_to_atom):
            try:
                aidx = int(atom_idx)
                if 0 <= aidx < n_atoms and node_idx < len(rgb_colors):
                    atom_colors[aidx] = rgb_colors[node_idx]
            except (ValueError, TypeError):
                continue
    else:
        for i in range(min(n_atoms, len(rgb_colors))):
            atom_colors[i] = rgb_colors[i]
    
    # Draw and save
    try:
        svg = draw_molecule_svg(smiles, atom_colors, size=size)
    except Exception as e:
        logger.warning(f"Drawing error: {e}")
        return None
    
    svg_path = output_dir / f"{filename_prefix}.svg"
    svg_path.write_text(svg)
    
    # Save colorbar
    if include_colorbar:
        cb_path = output_dir / f"{filename_prefix}_colorbar.png"
        save_colorbar(
            str(cb_path),
            cmap_info['cmap'],
            cmap_info['norm'],
            cmap_info['vmin'],
            cmap_info['vmax']
        )
    
    return str(svg_path)


def visualize_explanations(
    task_name: str,
    explanations: List[Dict] = None,
    explanations_path: str = None,
    output_dir: str = None,
    sample_size: int = 10,
    indices: List[int] = None,
    seed: int = 42,
    normalize: bool = True,
    cmap: str = "RdBu_r",
    size: Tuple[int, int] = (400, 400),
    display_inline: bool = False
) -> List[str]:
    """
    Visualize multiple explanations for a task.
    
    Args:
        task_name: Name of task
        explanations: List of explanation dicts (or load from path)
        explanations_path: Path to saved explanations file
        output_dir: Directory to save visualizations
        sample_size: Number of molecules to visualize
        indices: Specific indices to visualize (overrides sample_size)
        seed: Random seed for sampling
        normalize: Whether to normalize importance values
        cmap: Colormap name
        size: Image size
        display_inline: Whether to display in Jupyter
        
    Returns:
        List of saved file paths
    """
    if output_dir is None:
        output_dir = f"visualizations/{task_name}"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load explanations if needed
    if explanations is None:
        if explanations_path is None:
            raise ValueError("Provide explanations list or explanations_path")
        data = torch.load(explanations_path, map_location='cpu', weights_only=False)
        explanations = data.get('explanations', [])
    
    n = len(explanations)
    if n == 0:
        logger.warning("No explanations to visualize")
        return []
    
    # Select indices
    rng = np.random.default_rng(seed)
    if indices is None:
        if sample_size is None or sample_size >= n:
            indices = list(range(n))
        else:
            indices = rng.choice(n, size=sample_size, replace=False).tolist()
    
    indices = list(dict.fromkeys(indices))  # Remove duplicates
    
    logger.info(f"Visualizing {len(indices)} explanations for {task_name}")
    logger.info(f"Saving to: {output_dir}")
    
    saved_paths = []
    
    for idx in indices:
        if idx < 0 or idx >= n:
            continue
        
        expl = explanations[idx]
        
        # Get SMILES
        smiles = get_smiles_from_explanation(expl)
        if smiles is None:
            continue
        
        # Get node importance
        node_importance = expl.get('node_importance', expl.get('node_importance_raw', []))
        node_importance = np.asarray(node_importance, dtype=float).squeeze()
        if node_importance.ndim != 1:
            node_importance = node_importance.flatten()
        
        if len(node_importance) == 0:
            continue
        
        # Normalize
        if normalize:
            max_abs = np.nanmax(np.abs(node_importance))
            if max_abs > 0 and not np.isnan(max_abs):
                node_importance = node_importance / max_abs
        
        # Get colors
        rgb_colors, cmap_info = prepare_atom_colors(
            node_importance, cmap_name=cmap, vmin=-1.0, vmax=1.0
        )
        
        # Get molecule
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        
        n_atoms = mol.GetNumAtoms()
        
        # Extend colors
        if len(rgb_colors) < n_atoms:
            gray = (0.8, 0.8, 0.8)
            rgb_colors = list(rgb_colors) + [gray] * (n_atoms - len(rgb_colors))
        
        # Map to atoms
        node_to_atom = expl.get('node_to_atom_map')
        atom_colors = {}
        
        if node_to_atom is not None:
            for node_idx, atom_idx in enumerate(node_to_atom):
                try:
                    aidx = int(atom_idx)
                    if 0 <= aidx < n_atoms and node_idx < len(rgb_colors):
                        atom_colors[aidx] = rgb_colors[node_idx]
                except Exception:
                    continue
        else:
            for i in range(min(n_atoms, len(rgb_colors))):
                atom_colors[i] = rgb_colors[i]
        
        # Draw
        try:
            svg = draw_molecule_svg(smiles, atom_colors, size=size)
        except Exception as e:
            logger.warning(f"Skipping idx {idx} - drawing error: {e}")
            continue
        
        # Save
        svg_path = output_dir / f"explanation_{idx}.svg"
        svg_path.write_text(svg)
        saved_paths.append(str(svg_path))
        
        colorbar_path = output_dir / f"explanation_{idx}_colorbar.png"
        save_colorbar(
            str(colorbar_path),
            cmap_info['cmap'],
            cmap_info['norm'],
            cmap_info['vmin'],
            cmap_info['vmax']
        )
        
        # Display inline if requested
        if display_inline:
            try:
                from IPython.display import SVG, display, Image, HTML
                
                drug_id = expl.get('meta', {}).get('Drug_ID') if isinstance(expl.get('meta'), dict) else None
                y_val = expl.get('y', 'N/A')
                pred_val = expl.get('pred', 'N/A')
                
                info = f"<div style='font-family: sans-serif; margin:6px 0;'>"
                info += f"<b>Task:</b> {task_name} &nbsp;&nbsp; <b>Index:</b> {idx}"
                if drug_id:
                    info += f" &nbsp;&nbsp; <b>Drug_ID:</b> {drug_id}"
                if isinstance(y_val, (int, float)):
                    info += f" &nbsp;&nbsp; <b>Y:</b> {y_val:.4f}"
                if isinstance(pred_val, (int, float)):
                    info += f" &nbsp;&nbsp; <b>Pred:</b> {pred_val:.4f}"
                info += "</div>"
                
                display(HTML(info))
                display(SVG(str(svg_path)))
                display(Image(filename=str(colorbar_path)))
            except ImportError:
                pass
    
    logger.info(f"Saved {len(saved_paths)} visualizations to {output_dir}")
    
    return saved_paths


def visualize_all_tasks(
    task_explanations: Dict[Tuple[str, str], List[Dict]],
    output_dir: str,
    sample_size: int = 10,
    **kwargs
) -> Dict[str, List[str]]:
    """
    Visualize explanations for all tasks.
    
    Args:
        task_explanations: Dict mapping (task_name, split) -> explanations
        output_dir: Base output directory
        sample_size: Samples per task
        **kwargs: Additional arguments for visualize_explanations
        
    Returns:
        Dict mapping task_name -> list of saved paths
    """
    output_dir = Path(output_dir)
    all_paths = {}
    
    for (task_name, split), explanations in task_explanations.items():
        task_output = output_dir / task_name / f"visualizations_{split}"
        
        paths = visualize_explanations(
            task_name=task_name,
            explanations=explanations,
            output_dir=str(task_output),
            sample_size=sample_size,
            **kwargs
        )
        
        all_paths[f"{task_name}_{split}"] = paths
    
    return all_paths
