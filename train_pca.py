"""
Training script for PCA feature reduction using 432-feature input files.
This script uses Principal Component Analysis (PCA) to:
1. Takes 432 features as input from JSON files
2. Reduces them to 128 features using PCA
3. Extracts the 128-dimensional reduced features
"""

import json
import pathlib
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
import os
import pandas as pd
import pickle


def load_features_from_csv(csv_fpath, json_dir, input_features=432):
    """
    Load all features from JSON files listed in CSV.
    Returns a numpy array of shape (n_samples, input_features).
    """
    df = pd.read_csv(csv_fpath)
    features_list = []
    
    for idx, row in df.iterrows():
        json_fname = row['json']
        json_fpath = os.path.join(json_dir, json_fname)
        
        try:
            with open(json_fpath, 'r') as f:
                data = json.load(f)
            
            # Validate feature count
            if len(data) != input_features:
                print(f"Warning: {json_fname} has {len(data)} features, expected {input_features}. Skipping.")
                continue
            
            features_list.append(data)
        except Exception as e:
            print(f"Error loading {json_fname}: {e}")
            continue
    
    return np.array(features_list, dtype=np.float32)


def fit_pca(train_features, n_components=128, output_dir='results'):
    """
    Fit PCA model on training features and save it.
    
    Args:
        train_features: numpy array of shape (n_samples, n_features)
        n_components: number of components to keep (default: 128)
        output_dir: directory to save the PCA model
    
    Returns:
        Fitted PCA model
    """
    print(f"Fitting PCA: {train_features.shape[1]} → {n_components} features")
    print(f"Training samples: {train_features.shape[0]}")
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(train_features)
    
    # Calculate explained variance
    explained_variance_ratio = pca.explained_variance_ratio_.sum()
    print(f"\nPCA Summary:")
    print(f"  Input features: {train_features.shape[1]}")
    print(f"  Output features: {n_components}")
    print(f"  Explained variance ratio: {explained_variance_ratio:.4f} ({explained_variance_ratio*100:.2f}%)")
    print(f"  Top 10 components explain: {pca.explained_variance_ratio_[:10].sum():.4f} ({pca.explained_variance_ratio_[:10].sum()*100:.2f}%)")
    
    # Save PCA model
    os.makedirs(output_dir, exist_ok=True)
    pca_path = f"{output_dir}/pca_model.pkl"
    with open(pca_path, 'wb') as f:
        pickle.dump(pca, f)
    print(f"\nPCA model saved to: {pca_path}")
    
    return pca


def extract_reduced_features(pca, json_dir, output_dir, input_features=432, n_components=128):
    """
    Extract reduced features (128-dim) from all 432-feature JSON files using PCA.
    
    Args:
        pca: Fitted PCA model
        json_dir: Directory containing input JSON files with 432 features
        output_dir: Directory to save reduced feature JSON files
        input_features: Number of input features (default: 432)
        n_components: Number of output features (default: 128)
    """
    json_path = Path(json_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files with 432 features
    json_files = list(json_path.glob(f'*_features-{input_features}.json'))
    print(f"Found {len(json_files)} JSON files with {input_features} features")
    
    if len(json_files) == 0:
        print(f"Warning: No files found matching pattern '*_features-{input_features}.json' in {json_dir}")
        return
    
    processed = 0
    skipped = 0
    
    for json_file in json_files:
        try:
            # Load original features
            with open(json_file, 'r') as f:
                original_features = json.load(f)
            
            # Validate feature count
            if len(original_features) != input_features:
                print(f"Warning: {json_file.name} has {len(original_features)} features, expected {input_features}. Skipping.")
                skipped += 1
                continue
            
            # Convert to numpy array and transform using PCA
            features_array = np.array(original_features, dtype=np.float32).reshape(1, -1)
            reduced_features = pca.transform(features_array)
            reduced_features_list = reduced_features[0].tolist()
            
            # Create output filename (replace features-432 with features-128)
            output_filename = json_file.name.replace(f'_features-{input_features}.json', f'_features-{n_components}.json')
            output_file = output_path / output_filename
            
            # Save reduced features
            with open(output_file, 'w') as f:
                json.dump(reduced_features_list, f, indent=2)
            
            processed += 1
            
            if processed % 50 == 0:
                print(f"Processed {processed}/{len(json_files)} files...")
                
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            skipped += 1
            continue
    
    print(f"\nFeature extraction complete!")
    print(f"  Processed: {processed} files")
    print(f"  Skipped: {skipped} files")
    print(f"  Reduced features saved to: {output_path}")


def main():
    # Configuration
    config = {
        'input_features': 432,
        'n_components': 128,
        'train_csv_path': 'evaluation/csv/train_age.csv',
        'valid_csv_path': 'evaluation/csv/valid_age.csv',
        'json_dir': 'output',
        'output_dir': 'results_pca',
        'reduced_features_dir': 'output_pca'
    }
    
    # Convert paths to absolute
    root_dir = pathlib.Path(__file__).parent
    train_csv_path = str((root_dir / config['train_csv_path']).resolve())
    valid_csv_path = str((root_dir / config['valid_csv_path']).resolve())
    json_dir = str((root_dir / config['json_dir']).resolve())
    output_dir = str((root_dir / config['output_dir']).resolve())
    reduced_features_dir = str((root_dir / config['reduced_features_dir']).resolve())
    
    # Load training features
    print("="*60)
    print("LOADING TRAINING DATA")
    print("="*60)
    print("Loading training features...")
    train_features = load_features_from_csv(train_csv_path, json_dir, config['input_features'])
    print(f"Loaded {train_features.shape[0]} training samples with {train_features.shape[1]} features")
    
    # Load validation features (for evaluation)
    print("\nLoading validation features...")
    valid_features = load_features_from_csv(valid_csv_path, json_dir, config['input_features'])
    print(f"Loaded {valid_features.shape[0]} validation samples with {valid_features.shape[1]} features")
    
    print("\n" + "="*60)
    print("PCA CONFIGURATION")
    print("="*60)
    print(f"Input features: {config['input_features']}")
    print(f"Output features: {config['n_components']}")
    print("="*60)
    print()
    
    # Fit PCA on training data
    print("Fitting PCA model...")
    pca = fit_pca(
        train_features=train_features,
        n_components=config['n_components'],
        output_dir=output_dir
    )
    
    # Evaluate on validation set
    print("\n" + "="*60)
    print("EVALUATING ON VALIDATION SET")
    print("="*60)
    valid_reduced = pca.transform(valid_features)
    reconstruction = pca.inverse_transform(valid_reduced)
    reconstruction_error = np.mean((valid_features - reconstruction) ** 2)
    print(f"Mean squared reconstruction error: {reconstruction_error:.6f}")
    print("="*60)
    
    print(f"\nPCA fitting complete! Model saved to: {output_dir}")
    
    # Extract and save reduced features for all data
    print("\n" + "="*60)
    print("EXTRACTING REDUCED FEATURES (128-DIM)")
    print("="*60)
    extract_reduced_features(
        pca=pca,
        json_dir=json_dir,
        output_dir=reduced_features_dir,
        input_features=config['input_features'],
        n_components=config['n_components']
    )


if __name__ == "__main__":
    main()

