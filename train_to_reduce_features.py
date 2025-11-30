"""
Training script for models using 432-feature input files.
This script trains a model that:
1. Takes 432 features as input from JSON files
2. Reduces them to 128 features using a learned reduction layer
3. Uses the BFANet architecture from evaluation/bfa_dev for age prediction
"""

import sys
import json
import pathlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# Add evaluation directory to path to import modules
eval_dir = pathlib.Path(__file__).parent / 'evaluation' / 'bfa_dev'
sys.path.insert(0, str(eval_dir))

from evaluation.bfa_dev.models.network import BFANet
from evaluation.bfa_dev.data.dataset import BFADataset
from evaluation.bfa_dev.utils.trainer import train_model, evaluate_model


class FeatureReductionModel(nn.Module):
    """
    Model that reduces 432 features to 128 features, then uses BFANet for prediction.
    """
    def __init__(self, input_features=432, reduced_features=128, hidden_dim=32, task='age'):
        super().__init__()
        self.task = task
        
        # Feature reduction layer: 432 -> 128
        self.feature_reduction = nn.Sequential(
            nn.Linear(input_features, reduced_features),
            nn.LeakyReLU(0.2)
        )
        
        # Use BFANet with reduced feature dimension
        self.bfa_net = BFANet(task=task, input_dim=reduced_features, hidden_dim=hidden_dim)
    
    def forward(self, x):
        # Reduce features from 432 to 128
        reduced_features = self.feature_reduction(x)
        # Pass through BFANet
        output = self.bfa_net(reduced_features)
        return output
    
    def extract_features(self, x):
        """Extract only the reduced features (128-dim) without prediction."""
        return self.feature_reduction(x)


def extract_reduced_features(model, json_dir, output_dir, input_features=432, device='cpu'):
    """
    Extract reduced features (128-dim) from all 432-feature JSON files.
    
    Args:
        model: Trained FeatureReductionModel
        json_dir: Directory containing input JSON files with 432 features
        output_dir: Directory to save reduced feature JSON files
        input_features: Number of input features (default: 432)
        device: Device to run inference on
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
    
    model.eval()
    model.to(device)
    
    processed = 0
    skipped = 0
    
    with torch.no_grad():
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
                
                # Convert to tensor and process
                features_tensor = torch.tensor(original_features, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Extract reduced features (128-dim)
                reduced_features = model.extract_features(features_tensor)
                reduced_features_list = reduced_features.squeeze(0).cpu().tolist()
                
                # Create output filename (replace features-432 with features-128)
                output_filename = json_file.name.replace(f'_features-{input_features}.json', '_features-128.json')
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
        'num_epochs': 1000,
        'batch_size': 32,
        'lr': 0.0003,
        'gpu_id': 0,
        'task': 'age',
        'input_features': 432,
        'reduced_features': 128,
        'hidden_dim': 32,
        'train_csv_path': 'evaluation/csv/train_age.csv',
        'valid_csv_path': 'evaluation/csv/valid_age.csv',
        'eval_csv_path': 'evaluation/csv/eval_age.csv',
        'json_dir': 'output',
        'output_dir': 'results_432_to_128',
        'reduced_features_dir': 'output_reduced_128'
    }
    
    # Convert paths to absolute
    root_dir = pathlib.Path(__file__).parent
    train_csv_path = str((root_dir / config['train_csv_path']).resolve())
    valid_csv_path = str((root_dir / config['valid_csv_path']).resolve())
    eval_csv_path = str((root_dir / config['eval_csv_path']).resolve())
    json_dir = str((root_dir / config['json_dir']).resolve())
    output_dir = str((root_dir / config['output_dir']).resolve())
    reduced_features_dir = str((root_dir / config['reduced_features_dir']).resolve())
    
    # Setup device
    device = torch.device(
        f'cuda:{config["gpu_id"]}' 
        if torch.cuda.is_available() and config["gpu_id"] is not None 
        else 'cpu'
    )
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = BFADataset(train_csv_path, json_dir, task=config['task'])
    valid_dataset = BFADataset(valid_csv_path, json_dir, task=config['task'])
    eval_dataset = BFADataset(eval_csv_path, json_dir, task=config['task'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=len(eval_dataset), shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Create model with feature reduction
    model = FeatureReductionModel(
        input_features=config['input_features'],
        reduced_features=config['reduced_features'],
        hidden_dim=config['hidden_dim'],
        task=config['task']
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    print("="*60)
    print("MODEL CONFIGURATION")
    print("="*60)
    print(f"Task: {config['task']}")
    print(f"Input features: {config['input_features']}")
    print(f"Reduced features: {config['reduced_features']}")
    print(f"Hidden dimension: {config['hidden_dim']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Number of epochs: {config['num_epochs']}")
    print(f"Device: {device}")
    print("="*60)
    print()
    
    # Train the model
    print("Starting training...")
    train_model(
        model,
        train_loader,
        valid_loader,
        task=config['task'],
        optimizer=optimizer,
        num_epochs=config['num_epochs'],
        device=device,
        output_dir=output_dir
    )
    
    # Evaluate on evaluation set
    print("Evaluating on evaluation set...")
    evaluate_model(model, eval_loader, task=config['task'], device=device)
    
    print(f"\nTraining complete! Results saved to: {output_dir}")
    
    # Extract and save reduced features for all data
    print("\n" + "="*60)
    print("EXTRACTING REDUCED FEATURES")
    print("="*60)
    extract_reduced_features(
        model=model,
        json_dir=json_dir,
        output_dir=reduced_features_dir,
        input_features=config['input_features'],
        device=device
    )


if __name__ == "__main__":
    main()

