import os
import tempfile
from utils.run_cmd import run_command, relative

os.environ["PATH"] = "/usr/local/mrtrix3/bin:" + os.environ.get("PATH", "")


def create_mask(dwi_file, mask_file, bvec_path, bval_path):
    """
    Create a mask from a dwi file using mrtrix3.
    """
    if mask_file.exists():
        print(f"  [CACHE] Mask already exists: {relative(mask_file)}, skipping dwi2mask")
    else:
        run_command([
            "dwi2mask", str(dwi_file), str(mask_file),
            "-fslgrad", str(bvec_path), str(bval_path)
        ])


def create_response(dwi_file, response_file, bvec_path, bval_path, skip_csd=False):
    """
    Create a response from a dwi file using mrtrix3.
    """
    if response_file.exists():
        print(f"  [CACHE] Response already exists: {relative(response_file)}, skipping dwi2response")
    else:
        if skip_csd:
            print(f"  Skipping dwi2response (insufficient directions)")
        else:
            run_command([
                "dwi2response", "fa", str(dwi_file), str(response_file),
                "-fslgrad", str(bvec_path), str(bval_path)
            ])
    

def create_fod(dwi_file, fod_file, response_file, bvec_path, bval_path, skip_csd=False):
    """
    Create a FOD from a dwi file using mrtrix3.
    """
    if fod_file.exists():
        print(f"  [CACHE] FOD already exists: {relative(fod_file)}, skipping dwi2fod")
    else:
        if skip_csd:
            print(f"  Skipping dwi2fod (insufficient directions)")
        else:
            run_command([
                "dwi2fod", "csd", str(dwi_file), str(response_file), str(fod_file),
                "-fslgrad", str(bvec_path), str(bval_path)
            ])

def create_peaks(fod_file, peaks_file, mask_file, skip_peaks=False):
    """
    Create peaks from a fod file using mrtrix3.
    """
    if peaks_file.exists():
        print(f"  [CACHE] Peaks already exists: {relative(peaks_file)}, skipping sh2peaks")
    else:
        if skip_peaks:
            print(f"  Skipping sh2peaks (insufficient directions)")
        else:
            run_command([
                "sh2peaks", str(fod_file), str(peaks_file),
                "-mask", str(mask_file), "-fast"
            ])


def create_white_matter_mask(dwi_file, mask_file, threshold=0.4):
    """
    Create a white matter mask from a dwi file using mrtrix3.
    """
    force = True
    if mask_file.exists() and not force:
        print(f"  [CACHE] White matter mask already exists: {relative(mask_file)}, skipping dwi2mask")
    else:
        # Use delete=False to ensure file persists until both commands complete
        tmp_mask_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
        try:
            run_command([
                "mrthreshold", str(dwi_file), "-abs", str(threshold), tmp_mask_file.name, "-force"
            ])
            run_command([
                "mrconvert", str(tmp_mask_file.name), str(mask_file), "-datatype", "uint8", "-force"
            ])
        finally:
            tmp_mask_file.close()
            # Manually delete the temp file after both commands complete
            if os.path.exists(tmp_mask_file.name):
                os.unlink(tmp_mask_file.name)