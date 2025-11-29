import os, sys
import shutil
from pathlib import Path
from utils.run_cmd import run_command, relative

# Find scilpy CLI directory - check venv first, then try to import scilpy
venv_path = ""
for path in sys.path:
    if ".venv" in path:
        venv_path = path
        break

scilpy_dir = None
if venv_path:
    scilpy_dir = Path(venv_path) / "scilpy" / "cli"
    if not scilpy_dir.exists():
        scilpy_dir = None

# If not found in venv, try to find via import or PATH
if scilpy_dir is None or not scilpy_dir.exists():
    try:
        import scilpy
        scilpy_package_path = Path(scilpy.__file__).parent
        scilpy_dir = scilpy_package_path / "cli"
        if not scilpy_dir.exists():
            # Try finding via which command
            scilpy_script = shutil.which("scil_dti_metrics.py")
            if scilpy_script:
                scilpy_dir = Path(scilpy_script).parent
    except ImportError:
        # Try finding via which command
        scilpy_script = shutil.which("scil_dti_metrics.py")
        if scilpy_script:
            scilpy_dir = Path(scilpy_script).parent

if scilpy_dir and scilpy_dir.exists():
    os.environ["PATH"] = str(scilpy_dir) + ":" + os.environ.get("PATH", "")


def run_scilpy(metric_dir, dwi_file, mask_file, file_id, bval_path, bvec_path):
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
        # Verify input files exist and are not broken symlinks
        def check_file_not_broken_symlink(file_path, file_type):
            """Check if file exists and is not a broken symlink."""
            if not file_path.exists():
                print(f"  ERROR: {file_type} file does not exist: {relative(file_path)}")
                return False
            
            # Check if it's a broken symlink
            if file_path.is_symlink():
                try:
                    target = file_path.readlink()
                    # Try to resolve the target
                    target_path = target if target.is_absolute() else file_path.parent / target
                    if not target_path.exists():
                        print(f"  ERROR: {file_type} file is a broken symlink: {relative(file_path)}")
                        print(f"    Points to (missing): {relative(target)}")
                        print(f"    This usually means the .git/annex/objects/ directory was removed.")
                        print(f"    You need to restore the .git directory or re-download the dataset.")
                        return False
                except (OSError, RuntimeError) as e:
                    print(f"  ERROR: Cannot read {file_type} symlink: {relative(file_path)}")
                    print(f"    Error: {e}")
                    return False
            
            return True
        
        if not check_file_not_broken_symlink(dwi_file, "DWI"):
            print(f"  Skipping DTI metrics calculation for {file_id}")
            return
        if not check_file_not_broken_symlink(bval_path, "bval"):
            print(f"  Skipping DTI metrics calculation for {file_id}")
            return
        if not check_file_not_broken_symlink(bvec_path, "bvec"):
            print(f"  Skipping DTI metrics calculation for {file_id}")
            return
        
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
        
        # Use Python to call scilpy scripts to avoid permission issues
        if scilpy_dir and (scilpy_dir / "scil_dti_metrics.py").exists():
            scilpy_script = str(scilpy_dir / "scil_dti_metrics.py")
            run_command([
                sys.executable, scilpy_script, "--not_all",
                "--mask", abs_mask_file,
                "--fa", abs_fa_file,
                "--md", abs_md_file,
                "--ad", abs_ad_file,
                "--rd", abs_rd_file,
                abs_dwi_file, abs_bval_file, abs_bvec_file, "-f"
            ])
        else:
            # Fallback: try to find script in PATH or use module syntax
            scilpy_script = shutil.which("scil_dti_metrics.py")
            if scilpy_script:
                run_command([
                    sys.executable, scilpy_script, "--not_all",
                    "--mask", abs_mask_file,
                    "--fa", abs_fa_file,
                    "--md", abs_md_file,
                    "--ad", abs_ad_file,
                    "--rd", abs_rd_file,
                    abs_dwi_file, abs_bval_file, abs_bvec_file, "-f"
                ])
            else:
                # Use Python module syntax as last resort
                run_command([
                    sys.executable, "-m", "scilpy.cli.scil_dti_metrics", "--not_all",
                    "--mask", abs_mask_file,
                    "--fa", abs_fa_file,
                    "--md", abs_md_file,
                    "--ad", abs_ad_file,
                    "--rd", abs_rd_file,
                    abs_dwi_file, abs_bval_file, abs_bvec_file, "-f"
                ])