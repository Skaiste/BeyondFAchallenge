# Convert bval/bvec files to json in the same format as ours to debug algorithm I/O

import os
import json
import SimpleITK as sitk
import matplotlib.pyplot as plt

def bval_bvec_to_json(bval_file, bvec_file, output_json_path):
    """
    Extracts bval and bvec data and saves it in a JSON file in the specified format.

    Parameters:
        nifti_path (str): Path to the directory containing the NIfTI and corresponding bval/bvec files.
        output_json_path (str): Path to save the resulting JSON file.
    """


    
    # Read bval
    with open(bval_file, "r") as f:
        bvals = list(map(float, f.readline().strip().split()))
    
    # Read bvec
    with open(bvec_file, "r") as f:
        bvecs = [list(map(float, line.strip().split())) for line in f.readlines()]
    
    # Structure data in the new format
    data = []
    for i in range(len(bvals)):
        entry = {
            "BVAL": bvals[i] if i < len(bvals) else None,  # Add BVAL if it exists
            "BVEC": [bvecs[0][i], bvecs[1][i], bvecs[2][i]]  # Use each gradient's orientation from bvec
        }
        data.append(entry)
    
    # Write to JSON
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"JSON file saved to {output_json_path}")
