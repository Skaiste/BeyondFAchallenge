import os, sys
import shutil
from pathlib import Path
from utils.run_cmd import run_command, relative


def run_scilpy_dti(metric_dir, dwi_file, mask_file, bval_path, bvec_path):
    """
    Run scilpy on a metric directory using scilpy.
    """
    fa_file = metric_dir / "fa.nii.gz"
    md_file = metric_dir / "md.nii.gz"
    ad_file = metric_dir / "ad.nii.gz"
    rd_file = metric_dir / "rd.nii.gz"
    
    # Check if all metric files exist
    all_metrics_exist = all(f.exists() for f in [fa_file, md_file, ad_file, rd_file])
    if all_metrics_exist:
        print(f"  [CACHE] All DTI metric files (FA, MD, AD, RD) already exist, skipping scil_dti_metrics")
    else:
        # Convert to absolute paths for commands (but don't resolve symlinks)
        # Construct absolute paths manually to avoid following broken git-annex symlinks
        def make_absolute_no_resolve(p):
            return p if p.is_absolute() else Path.cwd() / p
        
        abs_dwi_nifti_file = make_absolute_no_resolve(dwi_file)
        abs_bval_path = make_absolute_no_resolve(bval_path)
        abs_bvec_path = make_absolute_no_resolve(bvec_path)
        
        print("Calculating DTI metrics (FA, MD, AD, RD)...")
        # Convert all paths to absolute paths for commands (without resolving symlinks)
        abs_mask_file = str(make_absolute_no_resolve(mask_file))
        abs_fa_file = str(make_absolute_no_resolve(fa_file))
        abs_md_file = str(make_absolute_no_resolve(md_file))
        abs_ad_file = str(make_absolute_no_resolve(ad_file))
        abs_rd_file = str(make_absolute_no_resolve(rd_file))
        abs_dwi_file = str(abs_dwi_nifti_file)
        abs_bval_file = str(abs_bval_path)
        abs_bvec_file = str(abs_bvec_path)
        
        run_command([
            sys.executable, "-m", "scilpy.cli.scil_dti_metrics", "--not_all",
            "--mask", abs_mask_file,
            "--fa", abs_fa_file,
            "--md", abs_md_file,
            "--ad", abs_ad_file,
            "--rd", abs_rd_file,
            abs_dwi_file, abs_bval_file, abs_bvec_file, "-f"
        ])

def estimate_frf(dwi_file, mask_file, wm_mask_file, frf_file, bval_path, bvec_path):
    """
    Estimate the fiber response function from a dwi file using scilpy.
    """
    if frf_file.exists():
        print(f"  [CACHE] Fiber response function already exists: {relative(frf_file)}, skipping estimate_frf")
    else:
        run_command([
            sys.executable, "-m", "scilpy.cli.scil_frf_ssst",
            str(dwi_file), str(bval_path), str(bvec_path),
            str(frf_file), "--mask", str(mask_file), 
            "--mask_wm", str(wm_mask_file), "-f"
        ])

def extract_fodf_metrics(metric_dir,fodf_file, mask_file):
    """
    Extract fodf metrics from a fodf file using scilpy.
    """
    afd_total_file = metric_dir / "afd_total.nii.gz"
    nufo_file = metric_dir / "nufo.nii.gz"
    if afd_total_file.exists() and nufo_file.exists():
        print(f"  [CACHE] FODF metrics already exist: {relative(afd_total_file)}, {relative(nufo_file)}, skipping extract_fodf_metrics")
    else:
        run_command([
            sys.executable, "-m", "scilpy.cli.scil_fodf_metrics", 
            str(fodf_file), "--not_all",
            "--mask", str(mask_file), 
            "--afd_total", str(afd_total_file),
            "--nufo", str(nufo_file),
            "-f"
        ])