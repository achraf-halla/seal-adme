"""
Visualization utilities for SEAL model explanations.

This module provides functions for visualizing fragment-level
explanations as color-coded molecular structures.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def prepare_atom_colors(
    values: np.ndarray,
    cmap_name: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    clip: bool = True
) -> Tuple[List[Tuple[float, float, float]], dict]:
    """
    Convert importance values to RGB colors.
    
    Args:
        values: Array of importance values
        cmap_name: Matplotlib colormap name
        vmin: Minimum value for normalization
        vmax: Maximum value for normalization
        clip: Whether to clip values to [vmin, vmax]
        
    Returns:
        Tuple of (list of RGB tuples, colormap info dict)
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib import colormaps
    
    vals = np.asarray(values, dtype=float).squeeze()
    
    if vals.size == 0:
        cmap = colormaps.get_cmap(cmap_name)
        norm = mcolors.Normalize(-1.0, 1.0)
        return [], {'vmin': -1.0, 'vmax': 1.0, 'cmap': cmap, 'norm': norm}
    
    if vmin is None or vmax is None:
        m = np.nanmax(np.abs(vals))
        if m == 0 or np.isnan(m):
            vmin, vmax = -1.0, 1.0
        else:
            vmin, vmax = -m, m
    
    if clip:
        vals = np.clip(vals, vmin, vmax)
    
    cmap = colormaps.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(vals))
    
    rgbs = [tuple(map(float, rgba[i][:3])) for i in range(len(rgba))]
    
    return rgbs, {'vmin': vmin, 'vmax': vmax, 'cmap': cmap, 'norm': norm}


def save_colorbar(
    path: Union[str, Path],
    cmap,
    norm,
    vmin: float,
    vmax: float,
    label: str = "importance",
    figsize: Tuple[float, float] = (4, 0.4),
    dpi: int = 150
) -> None:
    """Save a standalone colorbar image."""
    import matplotlib.pyplot as plt
    from matplotlib.colorbar import ColorbarBase
    
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(bottom=0.3, top=0.8, left=0.05, right=0.95)
    
    cb = ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.set_label(label)
    
    ticks = np.linspace(vmin, vmax, 5)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.2f}" for t in ticks])
    
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def draw_molecule_svg(
    smiles: str,
    atom_colors: Dict[int, Tuple[float, float, float]],
    size: Tuple[int, int] = (400, 400)
) -> str:
    """
    Draw a molecule with colored atoms as SVG.
    
    Args:
        smiles: SMILES string
        atom_colors: Dictionary mapping atom indices to RGB tuples
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
    
    highlight_atoms = []
    highlight_colors = {}
    n_atoms = mol.GetNumAtoms()
    
    for ai, col in atom_colors.items():
        if ai < 0 or ai >= n_atoms:
            continue
        if col is None:
            continue
        rgb = (float(col[0]), float(col[1]), float(col[2]))
        highlight_atoms.append(ai)
        highlight_colors[ai] = rgb
    
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors
    )
    drawer.FinishDrawing()
    
    return drawer.GetDrawingText()


def visualize_explanation(
    explanation,
    output_dir: Union[str, Path],
    normalize: bool = True,
    cmap: str = "RdBu_r",
    size: Tuple[int, int] = (400, 400),
    save_colorbar_: bool = True
) -> Optional[str]:
    """
    Visualize a single explanation as SVG.
    
    Args:
        explanation: MoleculeExplanation object or dict
        output_dir: Directory to save visualization
        normalize: Whether to normalize importance values
        cmap: Colormap name
        size: Image size
        save_colorbar_: Whether to save colorbar image
        
    Returns:
        Path to saved SVG file, or None if failed
    """
    from rdkit import Chem
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if hasattr(explanation, 'to_dict'):
        expl = explanation
        idx = expl.index
        smiles = expl.smiles
        node_importance = expl.node_importance
        node_to_atom_map = expl.node_to_atom_map
    else:
        expl = explanation
        idx = expl.get('index', 0)
        smiles = expl.get('smiles') or expl.get('canonical_smiles')
        if smiles is None:
            meta = expl.get('meta', {})
            smiles = meta.get('canonical_smiles') or meta.get('Canonical_Smiles')
        node_importance = expl.get('node_importance', expl.get('node_importance_raw'))
        node_to_atom_map = expl.get('node_to_atom_map')
    
    if smiles is None:
        logger.warning(f"No SMILES for explanation {idx}")
        return None
    
    if node_importance is None:
        logger.warning(f"No importance values for explanation {idx}")
        return None
    
    vals = np.asarray(node_importance, dtype=float).squeeze()
    if vals.ndim != 1:
        vals = vals.flatten()
    
    if normalize and vals.size > 0:
        max_abs = np.nanmax(np.abs(vals))
        if max_abs > 0 and not np.isnan(max_abs):
            vals = vals / max_abs
    
    rgbs, cmap_info = prepare_atom_colors(vals, cmap_name=cmap, vmin=-1.0, vmax=1.0)
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning(f"Could not parse SMILES for explanation {idx}")
        return None
    
    n_atoms = mol.GetNumAtoms()
    
    if len(rgbs) < n_atoms:
        pad = [(0.8, 0.8, 0.8)] * (n_atoms - len(rgbs))
        rgbs = rgbs + pad
    
    atom_colors = {}
    if node_to_atom_map is not None:
        for node_idx, atom_idx in enumerate(node_to_atom_map):
            try:
                aidx = int(atom_idx)
            except (TypeError, ValueError):
                continue
            if 0 <= aidx < n_atoms and node_idx < len(rgbs):
                atom_colors[aidx] = rgbs[node_idx]
    else:
        for i in range(min(n_atoms, len(rgbs))):
            atom_colors[i] = rgbs[i]
    
    try:
        svg = draw_molecule_svg(smiles, atom_colors, size=size)
    except Exception as e:
        logger.warning(f"Drawing error for explanation {idx}: {e}")
        return None
    
    svg_path = output_dir / f"explanation_{idx}.svg"
    svg_path.write_text(svg)
    
    if save_colorbar_:
        cb_path = output_dir / f"explanation_{idx}_colorbar.png"
        save_colorbar(
            cb_path,
            cmap_info['cmap'],
            cmap_info['norm'],
            cmap_info['vmin'],
            cmap_info['vmax']
        )
    
    return str(svg_path)


def visualize_explanations(
    task_name: str,
    explanations: List = None,
    explanations_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    sample_size: int = 10,
    indices: Optional[List[int]] = None,
    seed: int = 42,
    normalize: bool = True,
    cmap: str = "RdBu_r",
    size: Tuple[int, int] = (400, 400),
    display_inline: bool = False
) -> List[str]:
    """
    Visualize multiple explanations.
    
    Args:
        task_name: Name of the task
        explanations: List of explanations (or None to load from path)
        explanations_path: Path to saved explanations file
        output_dir: Output directory for visualizations
        sample_size: Number of explanations to visualize
        indices: Specific indices to visualize (overrides sample_size)
        seed: Random seed for sampling
        normalize: Whether to normalize importance values
        cmap: Colormap name
        size: Image size
        display_inline: Whether to display in Jupyter notebook
        
    Returns:
        List of paths to saved SVG files
    """
    import torch
    
    if output_dir is None:
        output_dir = f"visualizations/{task_name}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if explanations is None:
        if explanations_path is None:
            raise ValueError("Provide explanations list or explanations_path")
        data = torch.load(explanations_path, map_location='cpu', weights_only=False)
        explanations = data.get('explanations', [])
    
    n = len(explanations)
    if n == 0:
        raise ValueError("No explanations found")
    
    rng = np.random.default_rng(seed)
    if indices is None:
        if sample_size is None or sample_size >= n:
            indices = list(range(n))
        else:
            indices = rng.choice(n, size=sample_size, replace=False).tolist()
    
    indices = list(dict.fromkeys(indices))
    
    logger.info(f"Visualizing {len(indices)} explanations for {task_name}")
    
    saved_paths = []
    for idx in indices:
        if idx < 0 or idx >= n:
            continue
        
        expl = explanations[idx]
        svg_path = visualize_explanation(
            expl,
            output_dir=output_dir,
            normalize=normalize,
            cmap=cmap,
            size=size
        )
        
        if svg_path:
            saved_paths.append(svg_path)
            
            if display_inline:
                try:
                    from IPython.display import SVG, display, Image, HTML
                    
                    if hasattr(expl, 'drug_id'):
                        drug_id = expl.drug_id
                        y_val = expl.y
                        pred_val = expl.pred
                    else:
                        drug_id = expl.get('drug_id') or expl.get('meta', {}).get('Drug_ID')
                        y_val = expl.get('y', 'N/A')
                        pred_val = expl.get('pred', 'N/A')
                    
                    info = (
                        f"<div style='font-family: sans-serif; margin:6px 0;'>"
                        f"<b>Task:</b> {task_name} &nbsp;&nbsp; "
                        f"<b>Index:</b> {idx} &nbsp;&nbsp; "
                        f"<b>Drug_ID:</b> {drug_id}"
                    )
                    if isinstance(y_val, (int, float)):
                        info += f" &nbsp;&nbsp; <b>Y:</b> {y_val:.4f}"
                    if isinstance(pred_val, (int, float)):
                        info += f" &nbsp;&nbsp; <b>Pred:</b> {pred_val:.4f}"
                    info += "</div>"
                    
                    display(HTML(info))
                    display(SVG(filename=svg_path))
                    
                    cb_path = str(svg_path).replace('.svg', '_colorbar.png')
                    if Path(cb_path).exists():
                        display(Image(filename=cb_path))
                        
                except ImportError:
                    pass
    
    logger.info(f"Saved {len(saved_paths)} visualizations to {output_dir}")
    return saved_paths


def create_summary_figure(
    explanations: List,
    task_name: str,
    output_path: Union[str, Path],
    n_cols: int = 4,
    n_rows: int = 3,
    figsize: Tuple[float, float] = None,
    dpi: int = 150
) -> None:
    """
    Create a summary figure with multiple molecule visualizations.
    
    Args:
        explanations: List of explanations
        task_name: Name of the task
        output_path: Output file path
        n_cols: Number of columns
        n_rows: Number of rows
        figsize: Figure size (auto-calculated if None)
        dpi: Resolution
    """
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    import tempfile
    import os
    
    n_samples = min(len(explanations), n_cols * n_rows)
    
    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(n_samples):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            svg_path = visualize_explanation(
                explanations[i],
                output_dir=tmpdir,
                save_colorbar_=False
            )
            
            if svg_path:
                try:
                    from cairosvg import svg2png
                    png_path = svg_path.replace('.svg', '.png')
                    svg2png(url=svg_path, write_to=png_path, dpi=100)
                    img = imread(png_path)
                    ax.imshow(img)
                except ImportError:
                    ax.text(0.5, 0.5, 'cairosvg required',
                           ha='center', va='center', transform=ax.transAxes)
            
            if hasattr(explanations[i], 'drug_id'):
                drug_id = explanations[i].drug_id
                pred = explanations[i].pred
            else:
                drug_id = explanations[i].get('drug_id', f'mol_{i}')
                pred = explanations[i].get('pred', 0)
            
            ax.set_title(f"{drug_id}\npred: {pred:.3f}", fontsize=9)
            ax.axis('off')
        
        for i in range(n_samples, n_cols * n_rows):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis('off')
    
    fig.suptitle(f"{task_name} - Fragment Importance", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved summary figure to {output_path}")
