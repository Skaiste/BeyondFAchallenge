#!/usr/bin/env python3
"""
Plot brain scans for FA, MD, AD, and RD metrics.

This script visualizes diffusion tensor imaging (DTI) metrics from NIfTI files.
It can display brain scans in axial, coronal, or sagittal views, showing all
four metrics (FA, MD, AD, RD) side by side.

Usage:
    # View a specific slice interactively
    python plot_brain_scan.py --metric-dir tmp/tractseg_fa_output/sub-cIIIs01_ses-s1Bx1_acq-b1000n3r21x21x22peAPA_run-104/metric --slice 27 --view axial

    # Save figure to file
    python plot_brain_scan.py --metric-dir <path> --slice 27 --view axial --output scan.png

    # View middle slice of all views
    python plot_brain_scan.py --metric-dir <path> --view all
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path


def load_metric_nifti(metric_path):
    """Load a NIfTI file and return as numpy array."""
    if not os.path.exists(metric_path):
        raise FileNotFoundError(f"Metric file not found: {metric_path}")
    
    img = sitk.ReadImage(metric_path)
    array = sitk.GetArrayFromImage(img)
    return array, img


def get_slice(data, slice_idx, view='axial'):
    """
    Extract a 2D slice from 3D volume.
    
    Parameters:
    -----------
    data : np.ndarray
        3D array of shape (depth, height, width) or (z, y, x)
    slice_idx : int
        Index of the slice to extract
    view : str
        'axial' (z-axis), 'coronal' (y-axis), or 'sagittal' (x-axis)
    
    Returns:
    --------
    np.ndarray
        2D slice
    """
    if view == 'axial':
        # Axial view: slice along z-axis (first dimension)
        return data[slice_idx, :, :]
    elif view == 'coronal':
        # Coronal view: slice along y-axis (second dimension)
        return data[:, slice_idx, :]
    elif view == 'sagittal':
        # Sagittal view: slice along x-axis (third dimension)
        return data[:, :, slice_idx]
    else:
        raise ValueError(f"Unknown view: {view}. Must be 'axial', 'coronal', or 'sagittal'")


def plot_metrics(metric_dir, slice_idx=None, view='axial', output_path=None, 
                 figsize=(16, 4), dpi=100, cmap='hot'):
    """
    Plot FA, MD, AD, and RD metrics side by side.
    
    Parameters:
    -----------
    metric_dir : str
        Path to directory containing fa.nii.gz, md.nii.gz, ad.nii.gz, rd.nii.gz
    slice_idx : int, optional
        Index of slice to display. If None, displays middle slice.
    view : str
        'axial', 'coronal', 'sagittal', or 'all' to show all three views
    output_path : str, optional
        Path to save the figure. If None, displays interactively.
    figsize : tuple
        Figure size (width, height) in inches
    dpi : int
        Resolution for saved figures
    cmap : str
        Colormap to use ('hot', 'viridis', 'gray', 'jet', etc.)
    """
    metric_dir = Path(metric_dir)
    
    # Define metric files
    metric_files = {
        'FA': metric_dir / 'fa.nii.gz',
        'MD': metric_dir / 'md.nii.gz',
        'AD': metric_dir / 'ad.nii.gz',
        'RD': metric_dir / 'rd.nii.gz'
    }
    
    # Check if all files exist
    missing_files = [name for name, path in metric_files.items() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing metric files: {missing_files}")
    
    # Load all metrics
    metrics_data = {}
    metrics_info = {}
    for name, path in metric_files.items():
        array, img = load_metric_nifti(str(path))
        metrics_data[name] = array
        metrics_info[name] = img
        print(f"Loaded {name}: shape {array.shape}, spacing {img.GetSpacing()}")
    
    # Get data shape (should be same for all metrics)
    data_shape = list(metrics_data['FA'].shape)
    print(f"Data shape: {data_shape} (z, y, x)")
    
    # Determine slice indices if not provided
    if view == 'all':
        views = ['axial', 'coronal', 'sagittal']
        if slice_idx is None:
            slice_idx = {
                'axial': data_shape[0] // 2,
                'coronal': data_shape[1] // 2,
                'sagittal': data_shape[2] // 2
            }
        else:
            slice_idx = {
                'axial': slice_idx,
                'coronal': slice_idx,
                'sagittal': slice_idx
            }
    else:
        views = [view]
        if slice_idx is None:
            if view == 'axial':
                slice_idx = data_shape[0] // 2
            elif view == 'coronal':
                slice_idx = data_shape[1] // 2
            elif view == 'sagittal':
                slice_idx = data_shape[2] // 2
    
    # Create figure
    if view == 'all':
        n_rows = len(views)
        n_cols = len(metric_files)
        figsize = (figsize[0], figsize[1] * n_rows)
    else:
        n_rows = 1
        n_cols = len(metric_files)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
    
    # Handle single row/column case
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Define colormaps for each metric (optional - can use same for all)
    metric_cmaps = {
        'FA': 'hot',
        'MD': 'viridis',
        'AD': 'plasma',
        'RD': 'inferno'
    }
    
    # Plot each metric
    for row_idx, current_view in enumerate(views):
        if view == 'all':
            current_slice_idx = slice_idx[current_view]
        else:
            current_slice_idx = slice_idx
        
        for col_idx, (metric_name, metric_data) in enumerate(metrics_data.items()):
            ax = axes[row_idx, col_idx]
            
            # Extract slice
            slice_data = get_slice(metric_data, current_slice_idx, current_view)
            
            # Get colormap
            current_cmap = metric_cmaps.get(metric_name, cmap)
            
            # Display slice
            im = ax.imshow(slice_data, cmap=current_cmap, origin='lower', aspect='auto')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Set title
            if row_idx == 0:
                title = f"{metric_name}"
            else:
                title = ""
            if col_idx == 0:
                view_label = current_view.capitalize()
                ax.set_ylabel(f"{view_label}\nSlice {current_slice_idx}", fontsize=10)
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            # Add slice info in corner
            ax.text(0.02, 0.98, f"Slice {current_slice_idx}/{data_shape[0 if current_view=='axial' else 1 if current_view=='coronal' else 2]-1}",
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Add overall title
    metric_dir_name = metric_dir.parent.name if metric_dir.name == 'metric' else metric_dir.name
    fig.suptitle(f'Brain Scan Visualization: {metric_dir_name}', fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot brain scans for FA, MD, AD, and RD metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View middle axial slice
  python plot_brain_scan.py --metric-dir tmp/tractseg_fa_output/sub-cIIIs01_ses-s1Bx1_acq-b1000n3r21x21x22peAPA_run-104/metric

  # View specific slice in coronal view
  python plot_brain_scan.py --metric-dir <path> --slice 30 --view coronal

  # View all three views
  python plot_brain_scan.py --metric-dir <path> --view all

  # Save figure to file
  python plot_brain_scan.py --metric-dir <path> --slice 27 --output scan.png
        """
    )
    
    parser.add_argument('--metric-dir', type=str, required=True,
                       help='Path to directory containing fa.nii.gz, md.nii.gz, ad.nii.gz, rd.nii.gz')
    parser.add_argument('--slice', type=int, default=None,
                       help='Slice index to display (default: middle slice)')
    parser.add_argument('--view', type=str, default='axial',
                       choices=['axial', 'coronal', 'sagittal', 'all'],
                       help='View orientation: axial, coronal, sagittal, or all (default: axial)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save figure (default: display interactively)')
    parser.add_argument('--figsize', type=float, nargs=2, default=[16, 4],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Figure size in inches (default: 16 4)')
    parser.add_argument('--dpi', type=int, default=100,
                       help='Resolution for saved figures (default: 100)')
    parser.add_argument('--cmap', type=str, default='hot',
                       help='Colormap to use (default: hot)')
    
    args = parser.parse_args()
    
    # Validate metric directory
    if not os.path.exists(args.metric_dir):
        parser.error(f"Metric directory does not exist: {args.metric_dir}")
    
    # Plot metrics
    try:
        plot_metrics(
            metric_dir=args.metric_dir,
            slice_idx=args.slice,
            view=args.view,
            output_path=args.output,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            cmap=args.cmap
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

