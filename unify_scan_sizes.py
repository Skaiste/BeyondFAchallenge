#!/usr/bin/env python3
"""
Unify scan sizes by centering brains and cropping/padding to a common size.

This script processes FA, MD, RD, and AD metric files from tractseg_fa_output,
centers each brain, calculates the optimal unified size across all scans, and
outputs cropped/padded images to unified_metrics.

Usage:
    python unify_scan_sizes.py
"""

import os
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
import argparse


def get_brain_bbox(data, threshold=0.0):
    """
    Calculate bounding box of brain tissue (non-zero/non-background voxels).
    
    Parameters:
    -----------
    data : np.ndarray
        3D array of brain scan
    threshold : float
        Threshold value below which voxels are considered background
        
    Returns:
    --------
    tuple
        (min_z, max_z, min_y, max_y, min_x, max_x) bounding box coordinates
    """
    # Find non-zero voxels
    mask = data > threshold
    
    if not np.any(mask):
        # If no brain tissue found, return full volume
        return (0, data.shape[0], 0, data.shape[1], 0, data.shape[2])
    
    # Find bounding box
    z_indices, y_indices, x_indices = np.where(mask)
    
    min_z = max(0, z_indices.min())
    max_z = min(data.shape[0], z_indices.max() + 1)
    min_y = max(0, y_indices.min())
    max_y = min(data.shape[1], y_indices.max() + 1)
    min_x = max(0, x_indices.min())
    max_x = min(data.shape[2], x_indices.max() + 1)
    
    return (min_z, max_z, min_y, max_y, min_x, max_x)


def crop_or_pad_to_size(data, target_size, pad_value=0):
    """
    Crop or pad a 3D volume to target size, centering the content.
    
    Parameters:
    -----------
    data : np.ndarray
        3D array to resize
    target_size : tuple
        (z, y, x) target dimensions
    pad_value : float
        Value to use for padding
        
    Returns:
    --------
    np.ndarray
        Resized volume
    """
    current_size = np.array(data.shape)
    target_size = np.array(target_size)
    
    # Calculate padding/cropping needed
    diff = target_size - current_size
    
    # Initialize output array
    output = np.full(target_size, pad_value, dtype=data.dtype)
    
    # Calculate source and destination slices
    src_start = np.maximum(0, -diff // 2)
    src_end = current_size - np.maximum(0, diff // 2)
    dst_start = np.maximum(0, diff // 2)
    dst_end = target_size - np.maximum(0, -diff // 2)
    
    # Copy data
    output[dst_start[0]:dst_end[0], 
           dst_start[1]:dst_end[1], 
           dst_start[2]:dst_end[2]] = \
        data[src_start[0]:src_end[0],
             src_start[1]:src_end[1],
             src_start[2]:src_end[2]]
    
    return output


def process_scan(scan_dir, output_dir, unified_size, threshold=0.0):
    """
    Process a single scan: center brain and resize to unified size.
    
    Parameters:
    -----------
    scan_dir : Path
        Path to scan directory containing metric/ subdirectory
    output_dir : Path
        Output directory for processed scan
    unified_size : tuple
        (z, y, x) target size
    threshold : float
        Threshold for brain tissue detection
    """
    metric_dir = scan_dir / 'metric'
    output_metric_dir = output_dir / 'metric'
    output_metric_dir.mkdir(parents=True, exist_ok=True)
    
    # Define metric files
    metrics = ['fa', 'md', 'ad', 'rd']
    
    # Load all metrics
    metric_data = {}
    metric_images = {}
    bboxes = []
    
    for metric in metrics:
        metric_path = metric_dir / f'{metric}.nii.gz'
        if not metric_path.exists():
            print(f"Warning: {metric_path} not found, skipping scan {scan_dir.name}")
            return False
        
        # Load image
        img = sitk.ReadImage(str(metric_path))
        data = sitk.GetArrayFromImage(img)
        
        metric_data[metric] = data
        metric_images[metric] = img
        
        # Calculate bounding box
        bbox = get_brain_bbox(data, threshold)
        bboxes.append(bbox)
    
    # Use the union of all bounding boxes to ensure all metrics are processed consistently
    # This ensures we capture all brain tissue across all metrics
    min_z = min(bbox[0] for bbox in bboxes)
    max_z = max(bbox[1] for bbox in bboxes)
    min_y = min(bbox[2] for bbox in bboxes)
    max_y = max(bbox[3] for bbox in bboxes)
    min_x = min(bbox[4] for bbox in bboxes)
    max_x = max(bbox[5] for bbox in bboxes)
    
    unified_bbox = (min_z, max_z, min_y, max_y, min_x, max_x)
    
    # Process each metric
    for metric in metrics:
        data = metric_data[metric]
        img = metric_images[metric]
        
        # Extract brain region using unified bbox
        min_z, max_z, min_y, max_y, min_x, max_x = unified_bbox
        brain_region = data[min_z:max_z, min_y:max_y, min_x:max_x]
        
        # Center the brain region in the unified size volume
        # Calculate padding needed to center in unified size
        brain_z, brain_y, brain_x = brain_region.shape
        target_z, target_y, target_x = unified_size
        
        pad_z_before = (target_z - brain_z) // 2
        pad_z_after = target_z - brain_z - pad_z_before
        pad_y_before = (target_y - brain_y) // 2
        pad_y_after = target_y - brain_y - pad_y_before
        pad_x_before = (target_x - brain_x) // 2
        pad_x_after = target_x - brain_x - pad_x_before
        
        # Center the brain in unified size volume
        centered = np.pad(
            brain_region,
            ((pad_z_before, pad_z_after), (pad_y_before, pad_y_after), (pad_x_before, pad_x_after)),
            mode='constant',
            constant_values=0
        )
        
        # Create new SimpleITK image
        new_img = sitk.GetImageFromArray(centered)
        
        # Preserve metadata (spacing, origin, direction)
        new_img.SetSpacing(img.GetSpacing())
        new_img.SetOrigin(img.GetOrigin())
        new_img.SetDirection(img.GetDirection())
        
        # Save
        output_path = output_metric_dir / f'{metric}.nii.gz'
        sitk.WriteImage(new_img, str(output_path))
    
    return True


def find_all_scans(input_dir):
    """
    Find all scan directories that contain metric subdirectories.
    
    Parameters:
    -----------
    input_dir : Path
        Root directory containing scan subdirectories
        
    Returns:
    --------
    list
        List of Path objects to scan directories
    """
    scans = []
    for scan_dir in input_dir.iterdir():
        if scan_dir.is_dir():
            metric_dir = scan_dir / 'metric'
            if metric_dir.exists() and metric_dir.is_dir():
                # Check if all required metric files exist
                required_files = ['fa.nii.gz', 'md.nii.gz', 'ad.nii.gz', 'rd.nii.gz']
                if all((metric_dir / f).exists() for f in required_files):
                    scans.append(scan_dir)
    return scans


def calculate_unified_size(scans, threshold=0.0):
    """
    Calculate the optimal unified size across all scans.
    
    This finds the maximum brain size (bounding box) across all scans, then adds padding.
    
    Parameters:
    -----------
    scans : list
        List of Path objects to scan directories
    threshold : float
        Threshold for brain tissue detection
        
    Returns:
    --------
    tuple
        (z, y, x) unified size
    """
    print("Calculating unified size across all scans...")
    max_sizes = [0, 0, 0]
    
    for scan_dir in tqdm(scans, desc="Analyzing scans"):
        metric_dir = scan_dir / 'metric'
        
        # Load all metrics and find union of bounding boxes
        bboxes = []
        for metric in ['fa', 'md', 'ad', 'rd']:
            metric_path = metric_dir / f'{metric}.nii.gz'
            if not metric_path.exists():
                continue
            
            img = sitk.ReadImage(str(metric_path))
            data = sitk.GetArrayFromImage(img)
            bbox = get_brain_bbox(data, threshold)
            bboxes.append(bbox)
        
        if not bboxes:
            continue
        
        # Use union of all bounding boxes
        min_z = min(bbox[0] for bbox in bboxes)
        max_z = max(bbox[1] for bbox in bboxes)
        min_y = min(bbox[2] for bbox in bboxes)
        max_y = max(bbox[3] for bbox in bboxes)
        min_x = min(bbox[4] for bbox in bboxes)
        max_x = max(bbox[5] for bbox in bboxes)
        
        # Calculate brain size
        brain_z = max_z - min_z
        brain_y = max_y - min_y
        brain_x = max_x - min_x
        
        # Update max sizes
        max_sizes[0] = max(max_sizes[0], brain_z)
        max_sizes[1] = max(max_sizes[1], brain_y)
        max_sizes[2] = max(max_sizes[2], brain_x)
    
    # Add some padding to ensure we don't crop important tissue
    # Use 10% padding or minimum 5 voxels
    padding = [max(int(s * 0.1), 5) for s in max_sizes]
    unified_size = tuple(max_sizes[i] + 2 * padding[i] for i in range(3))
    
    print(f"Maximum brain size found: {tuple(max_sizes)} (z, y, x)")
    print(f"Unified size (with padding): {unified_size} (z, y, x)")
    return unified_size


def main():
    parser = argparse.ArgumentParser(
        description='Unify scan sizes by centering brains and resizing to common size',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input-dir', type=str, 
                       default='tmp/tractseg_fa_output',
                       help='Input directory containing scan subdirectories (default: tmp/tractseg_fa_output)')
    parser.add_argument('--output-dir', type=str,
                       default='tmp/unified_metrics',
                       help='Output directory for unified scans (default: tmp/unified_metrics)')
    parser.add_argument('--threshold', type=float, default=0.0,
                       help='Threshold for brain tissue detection (default: 0.0)')
    parser.add_argument('--target-size', type=int, nargs=3, default=None,
                       metavar=('Z', 'Y', 'X'),
                       help='Manually specify target size (z, y, x). If not provided, will be calculated automatically.')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        parser.error(f"Input directory does not exist: {input_dir}")
    
    # Find all scans
    print(f"Finding scans in {input_dir}...")
    scans = find_all_scans(input_dir)
    print(f"Found {len(scans)} scans")
    
    if len(scans) == 0:
        print("No scans found!")
        return 1
    
    # Calculate unified size
    if args.target_size:
        unified_size = tuple(args.target_size)
        print(f"Using manually specified unified size: {unified_size}")
    else:
        unified_size = calculate_unified_size(scans, args.threshold)
    
    # Process each scan
    print(f"\nProcessing scans and saving to {output_dir}...")
    successful = 0
    failed = 0
    
    for scan_dir in tqdm(scans, desc="Processing scans"):
        scan_name = scan_dir.name
        output_scan_dir = output_dir / scan_name
        
        try:
            if process_scan(scan_dir, output_scan_dir, unified_size, args.threshold):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\nError processing {scan_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\nCompleted!")
    print(f"Successfully processed: {successful} scans")
    print(f"Failed: {failed} scans")
    print(f"Unified size: {unified_size} (z, y, x)")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    exit(main())

