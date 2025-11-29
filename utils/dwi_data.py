from pathlib import Path
from utils.run_cmd import relative

def get_subjects(dataset_root, output_dir, tmp_dir, subjects_to_process=None):
    """
    Get subjects from dataset root. Also print information for debugging.
    """
    # List directories for debugging
    print(f"\nDataset root: {relative(dataset_root)}")
    if dataset_root.exists():
        subjects = [d for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith("sub-")]
        print(f"Found {len(subjects)} subjects")
        if subjects_to_process:
            subjects = [s for s in subjects if s.name in subjects_to_process]
            print(f"Processing {len(subjects)} filtered subjects")
    else:
        print(f"Warning: Dataset root {relative(dataset_root)} does not exist!")
        subjects = []

    print(f"Output directory: {relative(output_dir)}")
    print(f"Temporary directory: {relative(tmp_dir)}")

    return subjects

def find_dwi_files(subjects, dataset_root):
    """
    Find all DWI nifti files in BIDS structure: sub-*/ses-*/dwi/*_dwi.nii.gz
    """
    dwi_files = []
    for subject_dir in subjects:
        for session_dir in subject_dir.glob("ses-*/dwi"):
            dwi_nifti_files = list(session_dir.glob("*_dwi.nii.gz"))
            dwi_files.extend(dwi_nifti_files)

    if not dwi_files:
        print(f"No DWI .nii.gz files found in {relative(dataset_root)}")
    else:
        print(f"Found {len(dwi_files)} DWI file(s)")
        # Verify files exist and check for broken symlinks
        verified_dwi_files = []
        broken_symlinks = []
        for f in dwi_files:
            # Check if file exists (works for symlinks too)
            if f.exists():
                # Check if it's a broken symlink
                try:
                    if f.is_symlink():
                        target = f.readlink()
                        # Check if target exists (resolve the symlink target)
                        target_path = target if target.is_absolute() else f.parent / target
                        if not target_path.exists():
                            broken_symlinks.append((f, target))
                            print(f"  WARNING: Broken symlink (will skip): {relative(f)}")
                            print(f"    Points to (missing): {target}")
                            continue
                except (OSError, RuntimeError):
                    # If we can't read the symlink, assume it's broken
                    broken_symlinks.append((f, None))
                    print(f"  WARNING: Cannot read symlink (will skip): {relative(f)}")
                    continue
                
                # Convert to absolute path without resolving symlinks
                if f.is_absolute():
                    abs_f = f
                else:
                    abs_f = Path.cwd() / f
                verified_dwi_files.append(abs_f)
            else:
                print(f"  WARNING: File not found (will skip): {relative(f)}")
        
        dwi_files = verified_dwi_files
        print(f"Verified {len(dwi_files)} existing DWI file(s)")
        
        if broken_symlinks:
            print(f"\n  ERROR: Found {len(broken_symlinks)} broken symlink(s)!")
            print(f"  These files point to git-annex objects that no longer exist.")
            print(f"  You need to restore the .git directory or re-download the dataset.")
            print(f"  The .git/annex/objects/ directory contains the actual file content.")
        
        # Show first few as examples
        for f in dwi_files[:5]:
            print(f"  - {relative(f)}")
        if len(dwi_files) > 5:
            print(f"  ... and {len(dwi_files) - 5} more")

    return dwi_files