"""
Visualization utilities for SEAL explanations.

Provides functions to visualize fragment-level attributions on
molecular structures using RDKit drawing.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def prepare_colors(
    values: np.ndarray,
    cmap_name: str = "RdBu_r",
    vmin: float = None,
    vmax: float = None,
    clip: bool = True
) -> Tuple[List[Tuple[float, float, float]], Dict]:
    """
    Prepare colors for importance values.
    
    Args:
        values: Array of importance values
        cmap_name: Matplotlib colormap name
        vmin, vmax: Value range (None = symmetric around 0)
        clip: Whether to clip values to range
        
    Returns:
        Tuple of (RGB colors list, colormap info dict)
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib import colormaps
    
    vals = np.asarray(values, dtype=float).squeeze()
    
    if vals.size == 0:
        cmap = colormaps.get_cmap(cmap_name)
        norm = mcolors.Normalize(-1.0, 1.0)
        return [], {'vmin': -1.0, 'vmax': 1.0, 'cmap': cmap, 'norm': norm}
    
    # Set symmetric range if not specified
    if vmin is None or vmax is None:
        max_abs = np.nanmax(np.abs(vals))
        if max_abs == 0 or np.isnan(max_abs):
            vmin, vmax = -1.0, 1.0
        else:
            vmin, vmax = -max_abs, max_abs
    
    if clip:
        vals = np.clip(vals, vmin, vmax)
    
    cmap = colormaps.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    rgba = cmap(norm(vals))
    rgbs = [tuple(map(float, rgba[i][:3])) for i in range(len(rgba))]
    
    return rgbs, {'vmin': vmin, 'vmax': vmax, 'cmap': cmap, 'norm': norm}


def draw_molecule_svg(
    smiles: str,
    atom_colors: Dict[int, Tuple[float, float, float]],
    size: Tuple[int, int] = (400, 400)
) -> str:
    """
    Draw molecule with colored atoms.
    
    Args:
        smiles: SMILES string
        atom_colors: Dict mapping atom index to RGB color
        size: Image size (width, height)
        
    Returns:
        SVG string
    """
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Try to kekulize for better drawing
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        pass
    
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    opts = drawer.drawOptions()
    opts.padding = 0.02
    opts.bondLineWidth = 1.6
    
    # Prepare highlighting
    n_atoms = mol.GetNumAtoms()
    highlight_atoms = []
    highlight_atom_colors = {}
    
    for atom_idx, color in atom_colors.items():
        if 0 <= atom_idx < n_atoms and color is not None:
            highlight_atoms.append(atom_idx)
            highlight_atom_colors[atom_idx] = color
    
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_atom_colors
    )
    
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def save_colorbar(
    path: Path,
    cmap,
    norm,
    vmin: float,
    vmax: float,
    label: str = "Fragment Contribution",
    figsize: Tuple[float, float] = (4, 0.4),
    dpi: int = 150
):
    """Save a colorbar as an image."""
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


def visualize_explanation(
    explanation: Dict[str, Any],
    output_dir: Path,
    index: int = None,
    normalize: bool = True,
    cmap: str = "RdBu_r",
    size: Tuple[int, int] = (400, 400)
) -> Optional[Path]:
    """
    Visualize a single explanation.
    
    Args:
        explanation: Explanation dictionary
        output_dir: Directory to save visualization
        index: Index for filename (uses explanation['index'] if None)
        normalize: Whether to normalize importance values
        cmap: Colormap name
        size: Image size
        
    Returns:
        Path to saved SVG or None if visualization failed
    """
    from rdkit import Chem
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if index is None:
        index = explanation.get('index', 0)
    
    # Get importance values
    node_importance = explanation.get('node_importance')
    if node_importance is None:
        logger.warning(f"No node_importance in explanation {index}")
        return None
    
    vals = np.asarray(node_importance).squeeze()
    
    if normalize:
        max_abs = np.nanmax(np.abs(vals)) if vals.size > 0 else 1.0
        if max_abs > 0:
            vals = vals / max_abs
    
    # Get SMILES
    smiles = None
    meta = explanation.get('meta', {})
    if isinstance(meta, dict):
        smiles = meta.get('canonical_smiles') or meta.get('Canonical_Smiles')
    if smiles is None:
        smiles = explanation.get('canonical_smiles') or explanation.get('smiles')
    
    if smiles is None:
        logger.warning(f"No SMILES found for explanation {index}")
        return None
    
    # Validate molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning(f"Invalid SMILES for explanation {index}")
        return None
    
    n_atoms = mol.GetNumAtoms()
    
    # Prepare colors
    rgbs, cmap_info = prepare_colors(vals, cmap_name=cmap, vmin=-1.0, vmax=1.0)
    
    # Map nodes to atoms
    node_to_atom = explanation.get('node_to_atom_map')
    atom_colors = {}
    
    if node_to_atom is not None:
        for node_idx, atom_idx in enumerate(node_to_atom):
            if 0 <= atom_idx < n_atoms and node_idx < len(rgbs):
                atom_colors[atom_idx] = rgbs[node_idx]
    else:
        # Direct mapping
        for i in range(min(n_atoms, len(rgbs))):
            atom_colors[i] = rgbs[i]
    
    # Draw molecule
    try:
        svg = draw_molecule_svg(smiles, atom_colors, size=size)
    except Exception as e:
        logger.warning(f"Drawing failed for explanation {index}: {e}")
        return None
    
    # Save SVG
    svg_path = output_dir / f"explanation_{index}.svg"
    svg_path.write_text(svg)
    
    # Save colorbar
    colorbar_path = output_dir / f"explanation_{index}_colorbar.png"
    save_colorbar(
        colorbar_path,
        cmap_info['cmap'],
        cmap_info['norm'],
        cmap_info['vmin'],
        cmap_info['vmax']
    )
    
    return svg_path


def visualize_explanations(
    explanations: List[Dict],
    output_dir: Path,
    task_name: str = "",
    sample_size: int = 10,
    indices: List[int] = None,
    seed: int = 42,
    **kwargs
) -> List[Path]:
    """
    Visualize multiple explanations.
    
    Args:
        explanations: List of explanation dictionaries
        output_dir: Output directory
        task_name: Task name for logging
        sample_size: Number of samples if indices not specified
        indices: Specific indices to visualize
        seed: Random seed for sampling
        **kwargs: Additional args for visualize_explanation
        
    Returns:
        List of saved SVG paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n = len(explanations)
    if n == 0:
        logger.warning("No explanations to visualize")
        return []
    
    # Select indices
    if indices is None:
        rng = np.random.default_rng(seed)
        k = min(sample_size, n)
        indices = rng.choice(n, size=k, replace=False).tolist()
    
    logger.info(f"Visualizing {len(indices)} explanations for {task_name}")
    
    paths = []
    for idx in indices:
        if 0 <= idx < n:
            path = visualize_explanation(
                explanations[idx],
                output_dir,
                index=idx,
                **kwargs
            )
            if path:
                paths.append(path)
    
    logger.info(f"Saved {len(paths)} visualizations to {output_dir}")
    return paths


def create_summary_figure(
    explanations: List[Dict],
    output_path: Path,
    task_name: str = "",
    n_cols: int = 4,
    n_rows: int = 3,
    figsize: Tuple[float, float] = None
):
    """
    Create a summary figure with multiple molecules.
    
    Args:
        explanations: List of explanations
        output_path: Path to save figure
        task_name: Task name for title
        n_cols: Number of columns
        n_rows: Number of rows
        figsize: Figure size (auto if None)
    """
    import matplotlib.pyplot as plt
    from rdkit import Chem
    from rdkit.Chem import Draw
    
    n_mols = n_cols * n_rows
    sample = explanations[:n_mols]
    
    mols = []
    legends = []
    
    for expl in sample:
        meta = expl.get('meta', {})
        smiles = meta.get('canonical_smiles') if isinstance(meta, dict) else None
        
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mols.append(mol)
                y = expl.get('y', 'N/A')
                pred = expl.get('pred', 'N/A')
                if isinstance(y, float):
                    y = f"{y:.2f}"
                if isinstance(pred, float):
                    pred = f"{pred:.2f}"
                legends.append(f"y={y}, pred={pred}")
    
    if not mols:
        logger.warning("No valid molecules for summary figure")
        return
    
    if figsize is None:
        figsize = (n_cols * 3, n_rows * 3)
    
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=n_cols,
        subImgSize=(300, 300),
        legends=legends
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"Sample Predictions - {task_name}")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved summary figure to {output_path}")
