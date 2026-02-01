"""
Molecule visualization with fragment-level importance highlighting.

Uses RDKit to render molecules with atoms colored by their importance scores.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _get_smiles_from_explanation(expl: Dict, keys: Tuple[str, ...] = ("canonical_smiles", "smiles", "SMILES")) -> Optional[str]:
    """Extract SMILES from explanation dict."""
    # Check meta first
    meta = expl.get("meta")
    if isinstance(meta, dict):
        for k in keys:
            v = meta.get(k)
            if v is not None:
                return v
    
    # Check top-level
    for k in keys:
        v = expl.get(k)
        if v is not None:
            return v
    
    return None


def _prepare_colors(
    values: np.ndarray,
    cmap_name: str = "RdBu_r",
    vmin: float = None,
    vmax: float = None,
    clip: bool = True
) -> Tuple[List, Tuple]:
    """
    Prepare RGB colors from importance values.
    
    Args:
        values: Array of importance scores
        cmap_name: Matplotlib colormap name
        vmin: Minimum value for normalization
        vmax: Maximum value for normalization
        clip: Whether to clip values to [vmin, vmax]
        
    Returns:
        Tuple of (rgb_colors, (vmin, vmax, cmap, norm))
    """
    import matplotlib.colors as mcolors
    from matplotlib import colormaps
    
    vals = np.asarray(values, dtype=float).squeeze()
    
    if vals.size == 0:
        cmap = colormaps.get_cmap(cmap_name)
        norm = mcolors.Normalize(-1.0, 1.0)
        return [], (-1.0, 1.0, cmap, norm)
    
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
    
    return rgbs, (vmin, vmax, cmap, norm)


def _save_colorbar(
    path: str,
    cmap,
    norm,
    vmin: float,
    vmax: float,
    label: str = "importance",
    figsize: Tuple[float, float] = (4, 0.4),
    dpi: int = 150
):
    """Save a colorbar image."""
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


def _draw_molecule_svg(
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
    
    highlight_atoms = []
    highlight_atom_colors = {}
    n_atoms = mol.GetNumAtoms()
    
    for k, col in atom_colors.items():
        try:
            ai = int(k)
        except Exception:
            continue
        
        if ai < 0 or ai >= n_atoms:
            continue
        if col is None:
            continue
        
        if len(col) >= 3:
            rgb = (float(col[0]), float(col[1]), float(col[2]))
        else:
            rgb = (float(col),) * 3
        
        highlight_atoms.append(ai)
        highlight_atom_colors[ai] = rgb
    
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_atom_colors
    )
    drawer.FinishDrawing()
    
    return drawer.GetDrawingText()


def visualize_explanation(
    explanation: Dict,
    output_path: str,
    normalize: bool = True,
    cmap: str = "RdBu_r",
    size: Tuple[int, int] = (400, 400),
    save_colorbar: bool = True
) -> bool:
    """
    Visualize a single explanation.
    
    Args:
        explanation: Explanation dict from extract_explanations_for_task
        output_path: Path for SVG output
        normalize: Whether to normalize importance values
        cmap: Colormap name
        size: Image size
        save_colorbar: Whether to save colorbar image
        
    Returns:
        True if successful
    """
    # Get importance values
    arr = None
    for key in ("node_importance", "node_importance_raw"):
        if key in explanation and explanation[key] is not None:
            arr = explanation[key]
            break
    
    if arr is None:
        logger.warning("No importance values found")
        return False
    
    vals = np.asarray(arr, dtype=float).squeeze()
    if vals.ndim != 1:
        vals = vals.flatten()
    
    # Normalize
    if normalize:
        max_abs = np.nanmax(np.abs(vals)) if vals.size > 0 else 1.0
        if max_abs == 0 or np.isnan(max_abs):
            max_abs = 1.0
        vals = vals / max_abs
    
    # Get SMILES
    smiles = _get_smiles_from_explanation(explanation)
    if smiles is None:
        logger.warning("No SMILES found in explanation")
        return False
    
    # Prepare colors
    from rdkit import Chem
    rgbs, (vmin, vmax, cmap_obj, norm) = _prepare_colors(vals, cmap_name=cmap, vmin=-1.0, vmax=1.0)
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning(f"Invalid SMILES: {smiles}")
        return False
    
    n_atoms = mol.GetNumAtoms()
    
    # Extend colors if needed
    if len(rgbs) < n_atoms:
        pad = [(0.8, 0.8, 0.8)] * (n_atoms - len(rgbs))
        rgbs_extended = rgbs + pad
    else:
        rgbs_extended = rgbs
    
    # Map node importance to atom colors
    atom_map = explanation.get("node_to_atom_map")
    atom_colors = {}
    
    if atom_map is not None:
        for node_idx, atom_idx in enumerate(atom_map):
            try:
                aidx = int(atom_idx)
            except Exception:
                continue
            if aidx < 0 or aidx >= n_atoms:
                continue
            if node_idx < len(rgbs_extended):
                atom_colors[aidx] = rgbs_extended[node_idx]
    else:
        for i in range(min(n_atoms, len(rgbs_extended))):
            atom_colors[i] = rgbs_extended[i]
    
    # Draw molecule
    try:
        svg = _draw_molecule_svg(smiles, atom_colors, size=size)
    except Exception as e:
        logger.warning(f"Drawing error: {e}")
        return False
    
    # Save
    output_pathp = Path(output_path)
    output_pathp.parent.mkdir(parents=True, exist_ok=True)
    output_pathp.write_text(svg)
    
    # Save colorbar
    if save_colorbar:
        colorbar_path = output_pathp.with_suffix('.colorbar.png')
        _save_colorbar(str(colorbar_path), cmap_obj, norm, vmin, vmax)
    
    return True


def visualize_task_explanations(
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
    display_inline: bool = True
):
    """
    Visualize multiple explanations for a task.
    
    Args:
        task_name: Name of the task
        explanations: List of explanation dicts
        explanations_path: Path to saved explanations (.pt file)
        output_dir: Output directory for visualizations
        sample_size: Number of molecules to visualize
        indices: Specific indices to visualize
        seed: Random seed for sampling
        normalize: Whether to normalize importance values
        cmap: Colormap name
        size: Image size
        display_inline: Whether to display in Jupyter
    """
    if output_dir is None:
        output_dir = f"visualizations/{task_name}"
    
    outp = Path(output_dir)
    outp.mkdir(parents=True, exist_ok=True)
    
    # Load explanations if needed
    if explanations is None:
        if explanations_path is None:
            raise ValueError("Provide explanations list or explanations_path")
        data = torch.load(explanations_path, map_location="cpu", weights_only=False)
        explanations = data.get("explanations") or []
    
    n = len(explanations)
    if n == 0:
        raise ValueError("No explanations found")
    
    # Select indices
    rng = np.random.default_rng(seed)
    if indices is None:
        if sample_size is None:
            indices = list(range(n))
        else:
            k = min(sample_size, n)
            indices = list(range(n)) if k == n else rng.choice(n, size=k, replace=False).tolist()
    
    indices = list(dict.fromkeys(indices))  # Remove duplicates
    
    logger.info(f"Visualizing {len(indices)} explanations for {task_name}")
    logger.info(f"Saving to: {outp}")
    
    for idx in indices:
        if idx < 0 or idx >= n:
            continue
        
        expl = explanations[idx]
        svg_path = outp / f"explanation_{idx}.svg"
        
        success = visualize_explanation(
            explanation=expl,
            output_path=str(svg_path),
            normalize=normalize,
            cmap=cmap,
            size=size,
            save_colorbar=True
        )
        
        if not success:
            continue
        
        # Display in Jupyter
        if display_inline:
            try:
                from IPython.display import SVG, display, Image, HTML
                
                drug_id = expl.get('meta', {}).get('Drug_ID') if isinstance(expl.get('meta'), dict) else None
                y_val = expl.get('y', 'N/A')
                pred_val = expl.get('pred', 'N/A')
                
                info_html = f"<div style='font-family: sans-serif; margin:6px 0;'>"
                info_html += f"<b>Task:</b> {task_name} &nbsp;&nbsp; <b>Index:</b> {idx}"
                if drug_id:
                    info_html += f" &nbsp;&nbsp; <b>Drug_ID:</b> {drug_id}"
                if isinstance(y_val, (int, float)):
                    info_html += f" &nbsp;&nbsp; <b>Y:</b> {y_val:.4f}"
                if isinstance(pred_val, (int, float)):
                    info_html += f" &nbsp;&nbsp; <b>Pred:</b> {pred_val:.4f}"
                info_html += "</div>"
                
                display(HTML(info_html))
                display(SVG(filename=str(svg_path)))
                
                colorbar_path = svg_path.with_suffix('.colorbar.png')
                if colorbar_path.exists():
                    display(Image(filename=str(colorbar_path)))
                    
            except ImportError:
                pass
    
    logger.info(f"Visualizations saved to {outp}")


def create_summary_figure(
    explanations: List[Dict],
    task_name: str,
    output_path: str,
    n_cols: int = 5,
    n_rows: int = 2,
    figsize: Tuple[float, float] = (20, 8),
    dpi: int = 150
):
    """
    Create a summary figure with multiple molecules.
    
    Args:
        explanations: List of explanations
        task_name: Task name for title
        output_path: Output path for figure
        n_cols: Number of columns
        n_rows: Number of rows
        figsize: Figure size
        dpi: DPI for saving
    """
    import matplotlib.pyplot as plt
    from rdkit import Chem
    from rdkit.Chem import Draw
    
    n_mols = min(n_cols * n_rows, len(explanations))
    
    mols = []
    legends = []
    
    for i, expl in enumerate(explanations[:n_mols]):
        smiles = _get_smiles_from_explanation(expl)
        if smiles is None:
            continue
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        
        mols.append(mol)
        
        pred = expl.get('pred', 'N/A')
        y = expl.get('y', 'N/A')
        
        if isinstance(pred, (int, float)) and isinstance(y, (int, float)):
            legends.append(f"Y={y:.2f}, Pred={pred:.2f}")
        else:
            legends.append(f"idx={i}")
    
    if not mols:
        logger.warning("No valid molecules to display")
        return
    
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=n_cols,
        subImgSize=(300, 300),
        legends=legends
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"Sample Predictions - {task_name}", fontsize=14)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Summary figure saved to {output_path}")
