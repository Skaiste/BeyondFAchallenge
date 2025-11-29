from pathlib import Path
from collections import defaultdict

dataset_root = Path("ds003416/derivatives/prequal-v1.0.0")

def is_file_accessible(file_path):
    """
    Check if a file actually exists and is accessible (not a broken symlink).
    
    Returns True if:
    - File exists and is a regular file
    - File exists and is a working symlink (target exists)
    Returns False if:
    - File doesn't exist
    - File is a broken symlink (target doesn't exist)
    """
    if not file_path.exists():
        return False
    
    # If it's a symlink, check if the target exists
    if file_path.is_symlink():
        try:
            target = file_path.readlink()
            # Resolve the target path
            if target.is_absolute():
                target_path = target
            else:
                target_path = file_path.parent / target
            
            # Check if target exists
            if not target_path.exists():
                return False  # Broken symlink
        except (OSError, RuntimeError):
            return False  # Cannot read symlink
    
    return True

# Dictionary to store subjects and their sessions with .nii.gz files
subjects_with_images = defaultdict(list)
broken_symlinks = []

# Walk through all subject directories
for subject_dir in sorted(dataset_root.glob("sub-*")):
    subject_id = subject_dir.name
    
    # Check each session
    for session_dir in sorted(subject_dir.glob("ses-*")):
        session_id = session_dir.name
        dwi_dir = session_dir / "dwi"
        
        if dwi_dir.exists():
            # Check if there are any .nii.gz files in this session
            all_nii_files = list(dwi_dir.glob("*.nii.gz"))
            
            # Filter to only include accessible files (not broken symlinks)
            accessible_files = []
            for nii_file in all_nii_files:
                if is_file_accessible(nii_file):
                    accessible_files.append(nii_file)
                else:
                    broken_symlinks.append(nii_file)
            
            if accessible_files:
                subjects_with_images[subject_id].append({
                    'session': session_id,
                    'count': len(accessible_files),
                    'total_found': len(all_nii_files),
                    'broken': len(all_nii_files) - len(accessible_files)
                })

# Print results
print(f"Total subjects with accessible image files: {len(subjects_with_images)}\n")

if broken_symlinks:
    print(f"WARNING: Found {len(broken_symlinks)} broken symlink(s) (not counted above)")
    print("These files point to git-annex objects that no longer exist.\n")

print("=" * 80)

for subject_id, sessions in sorted(subjects_with_images.items()):
    total_files = sum(s['count'] for s in sessions)
    total_broken = sum(s.get('broken', 0) for s in sessions)
    
    print(f"\n{subject_id}:")
    print(f"  Accessible .nii.gz files: {total_files}")
    if total_broken > 0:
        print(f"  Broken symlinks: {total_broken}")
    print(f"  Sessions with images: {len(sessions)}")
    for sess_info in sessions:
        broken_info = f" ({sess_info.get('broken', 0)} broken)" if sess_info.get('broken', 0) > 0 else ""
        print(f"    - {sess_info['session']}: {sess_info['count']} accessible files{broken_info}")

print(f"\nSubjects with accessible files: {list(subjects_with_images.keys())}")