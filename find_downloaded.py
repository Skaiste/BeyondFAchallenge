from pathlib import Path
from collections import defaultdict

dataset_root = Path("ds003416/derivatives/prequal-v1.0.0")

# Dictionary to store subjects and their sessions with .nii.gz files
subjects_with_images = defaultdict(list)

# Walk through all subject directories
for subject_dir in sorted(dataset_root.glob("sub-*")):
    subject_id = subject_dir.name
    
    # Check each session
    for session_dir in sorted(subject_dir.glob("ses-*")):
        session_id = session_dir.name
        dwi_dir = session_dir / "dwi"
        
        if dwi_dir.exists():
            # Check if there are any .nii.gz files in this session
            nii_files = list(dwi_dir.glob("*.nii.gz"))
            if nii_files:
                subjects_with_images[subject_id].append({
                    'session': session_id,
                    'count': len(nii_files)
                })

# Print results
print(f"Total subjects with image files: {len(subjects_with_images)}\n")
print("=" * 80)

for subject_id, sessions in sorted(subjects_with_images.items()):
    total_files = sum(s['count'] for s in sessions)
    print(f"\n{subject_id}:")
    print(f"  Total .nii.gz files: {total_files}")
    print(f"  Sessions with images: {len(sessions)}")
    for sess_info in sessions:
        print(f"    - {sess_info['session']}: {sess_info['count']} files")

print(list(subjects_with_images.keys()))