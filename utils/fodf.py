#!/usr/bin/env python3
"""
Script for computing fODF (fiber Orientation Distribution Function) using DIPY CSD.

This script performs Constrained Spherical Deconvolution (CSD) on DWI data
to compute fiber orientation distribution functions.
"""

import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from utils.run_cmd import relative


def compute_fodf(dwi_file, bval_file, bvec_file, frf_file, output_file, mask_file=None, sh_order=8):
    """
    Compute fODF from DWI data using Constrained Spherical Deconvolution.
    
    Args:
        dwi_file: Path to DWI NIfTI file
        bval_file: Path to b-values file
        bvec_file: Path to b-vectors file
        frf_file: Path to response function file (text file with eigenvalues and S0)
        output_file: Path to output fODF NIfTI file (spherical harmonics coefficients)
        mask_file: Optional path to mask file
        sh_order: Spherical harmonics order (default: 8, must be even)
    """
    dwi_path = Path(dwi_file)
    bval_path = Path(bval_file)
    bvec_path = Path(bvec_file)
    frf_path = Path(frf_file)
    output_path = Path(output_file)
    
    # Check if output already exists
    if output_path.exists():
        print(f"  [CACHE] fODF already exists: {relative(output_path)}, skipping computation")
        return
    
    # Verify input files exist
    for file_path, file_type in [(dwi_path, "DWI"), (bval_path, "bval"), 
                                  (bvec_path, "bvec"), (frf_path, "response function")]:
        if not file_path.exists():
            raise FileNotFoundError(f"{file_type} file does not exist: {relative(file_path)}")
    
    print(f"Computing fODF from {relative(dwi_path)}...")
    
    # Load DWI data
    img = nib.load(str(dwi_path))
    data = img.get_fdata().astype(np.float32)
    affine = img.affine
    
    # Load b-values and b-vectors
    bvals, bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))
    gtab = gradient_table(bvals, bvecs=bvecs)
    
    # Load response function
    response = np.loadtxt(str(frf_path))  # shape → [evals, S0]
    evals = response[:3]                   # eigenvalues
    S0 = response[3]                       # S0 value
    
    # Apply mask if provided
    if mask_file:
        mask_path = Path(mask_file)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file does not exist: {relative(mask_path)}")
        mask_img = nib.load(str(mask_path))
        mask = mask_img.get_fdata().astype(bool)
        
        # Ensure mask is 3D (spatial dimensions only)
        # If mask has 4 dimensions, take the first volume or squeeze if last dim is 1
        if mask.ndim == 4:
            if mask.shape[3] == 1:
                mask = mask.squeeze(axis=3)
            else:
                # If mask has multiple volumes, use the first one
                mask = mask[..., 0]
        
        # Verify mask spatial dimensions match data spatial dimensions
        if mask.shape[:3] != data.shape[:3]:
            raise ValueError(
                f"Mask spatial dimensions {mask.shape[:3]} do not match "
                f"data spatial dimensions {data.shape[:3]}"
            )
        
        # Apply mask to data: broadcast mask (H, W, D) to (H, W, D, N_volumes)
        data = data * mask[..., np.newaxis]
    
    # Build CSD model with SH order
    csd_model = ConstrainedSphericalDeconvModel(gtab, (evals, S0), sh_order=sh_order)
    
    # Fit & compute fODF
    print("  Fitting CSD model...")
    csd_fit = csd_model.fit(data)
    
    print("  Computing fODF (spherical harmonics coefficients)...")
    # Get SH coefficients instead of ODF values (required for scil_fodf_metrics.py)
    fodf_sh = csd_fit.shm_coeff.astype(np.float32)
    
    # Save as .nii.gz
    print(f"  Saving fODF SH coefficients to {relative(output_path)}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fodf_img = nib.Nifti1Image(fodf_sh, affine)
    nib.save(fodf_img, str(output_path))
    
    print(f"  ✓ fODF computation complete: {relative(output_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute fODF (fiber Orientation Distribution Function) using DIPY CSD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python compute_fodf.py \\
    --dwi dwi.nii.gz \\
    --bval dwi.bval \\
    --bvec dwi.bvec \\
    --frf frf_ssst.txt \\
    --output fodf.nii.gz \\
    --mask mask.nii.gz
        """
    )
    
    parser.add_argument("--dwi", required=True, help="Path to DWI NIfTI file")
    parser.add_argument("--bval", required=True, help="Path to b-values file (.bval)")
    parser.add_argument("--bvec", required=True, help="Path to b-vectors file (.bvec)")
    parser.add_argument("--frf", required=True, help="Path to response function file (text file with eigenvalues and S0)")
    parser.add_argument("--output", required=True, help="Path to output fODF NIfTI file (spherical harmonics coefficients)")
    parser.add_argument("--mask", help="Optional path to mask file")
    parser.add_argument("--sh_order", type=int, default=8, help="Spherical harmonics order (default: 8, must be even)")
    
    args = parser.parse_args()
    
    # Validate sh_order is even
    if args.sh_order % 2 != 0:
        print(f"ERROR: sh_order must be an even integer, got {args.sh_order}")
        exit(1)
    
    try:
        compute_fodf(
            dwi_file=args.dwi,
            bval_file=args.bval,
            bvec_file=args.bvec,
            frf_file=args.frf,
            output_file=args.output,
            mask_file=args.mask,
            sh_order=args.sh_order
        )
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)

