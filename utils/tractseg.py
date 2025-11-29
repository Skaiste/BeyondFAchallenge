from utils.run_cmd import run_command, relative

def run_tractseg(peaks_file, bundle_roi_dir, tractseg_dir, bval_path, bvec_path, mask_file, skip_csd=False):
    """
    Run TractSeg on a peaks file using mrtrix3.
    """
    if bundle_roi_dir.exists() and any(bundle_roi_dir.glob("*.nii.gz")):
        print(f"  [CACHE] TractSeg output already exists: {relative(bundle_roi_dir)}, skipping TractSeg")
    else:
        if skip_csd:
            print(f"  Skipping TractSeg (insufficient directions for CSD/FOD)")
        else:
            print("Running TractSeg...")
            run_command([
                "TractSeg", "-i", str(peaks_file), "-o", str(tractseg_dir),
                "--bvals", str(bval_path), "--bvecs", str(bvec_path),
                "--keep_intermediate_files",
                "--brain_mask", str(mask_file)
            ])