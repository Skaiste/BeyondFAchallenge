"""
Training script for autoencoder using 432-feature input files.
This script trains an autoencoder that:
1. Takes 432 features as input from JSON files
2. Encodes them through: 432 → 256 → 128 (bottleneck)
3. Decodes them through: 128 → 256 → 432 (reconstruction)
4. Extracts the 128-dimensional bottleneck features
"""

import json
import pathlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class AutoencoderDataset(Dataset):
    """
    Dataset that loads features from JSON files (ignoring targets for autoencoder training).
    """
    def __init__(self, csv_fpath, json_dir):
        self.df = pd.read_csv(csv_fpath)
        self.json_dir = json_dir

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        json_fname = row['json']
        json_fpath = os.path.join(self.json_dir, json_fname)

        with open(json_fpath, 'r') as f:
            data = json.load(f)
        feature = torch.tensor(data, dtype=torch.float32)
        
        # For autoencoder, we use the features as both input and target
        return feature


class Autoencoder(nn.Module):
    """
    Autoencoder model: 432 → 256 → 128 → 256 → 432
    """
    def __init__(self, input_features=432):
        super().__init__()
        
        # Encoder: 432 → 256 → 128
        self.encoder = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder: 128 → 256 → 432
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, input_features)
        )
    
    def forward(self, x):
        # Encode to bottleneck
        encoded = self.encoder(x)
        # Decode to reconstruction
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Extract the 128-dimensional bottleneck features."""
        return self.encoder(x)


def train_autoencoder(model, train_loader, valid_loader, optimizer, num_epochs, device, output_dir='results'):
    """
    Train the autoencoder using reconstruction loss.
    """
    writer = SummaryWriter(log_dir=f'{output_dir}/runs')
    model.to(device)
    criterion = nn.MSELoss()
    
    # Track best model
    best_valid_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss_train = 0.0
        for features in train_loader:
            features = features.to(device)
            
            optimizer.zero_grad()
            reconstructed = model(features)
            loss = criterion(reconstructed, features)
            loss.backward()
            optimizer.step()
            
            total_loss_train += loss.item()
        
        avg_loss_train = total_loss_train / len(train_loader)
        
        # Validation
        model.eval()
        total_loss_valid = 0.0
        with torch.no_grad():
            for features in valid_loader:
                features = features.to(device)
                reconstructed = model(features)
                loss = criterion(reconstructed, features)
                total_loss_valid += loss.item()
        
        avg_loss_valid = total_loss_valid / len(valid_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss_train:.6f}, Valid Loss: {avg_loss_valid:.6f}")
        
        # Log to tensorboard
        writer.add_scalars('Loss/Reconstruction', {'train': avg_loss_train, 'valid': avg_loss_valid}, epoch)
        
        # Check if this is the best model
        if avg_loss_valid < best_valid_loss:
            best_valid_loss = avg_loss_valid
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            print(f"*** NEW BEST MODEL at Epoch {best_epoch} (Valid Loss: {best_valid_loss:.6f}) ***")
    
    writer.close()
    
    # Print best model summary
    print("\n" + "="*60)
    print("BEST MODEL SUMMARY")
    print("="*60)
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation Loss: {best_valid_loss:.6f}")
    print("="*60 + "\n")
    
    # Save best model
    if best_model_state is not None:
        os.makedirs(output_dir, exist_ok=True)
        best_model_path = f"{output_dir}/best_autoencoder.pth"
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'valid_loss': best_valid_loss,
        }, best_model_path)
        print(f"Best model saved to: {best_model_path}\n")
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)


def extract_reduced_features(model, json_dir, output_dir, input_features=432, device='cpu'):
    """
    Extract reduced features (128-dim) from all 432-feature JSON files using the encoder.
    
    Args:
        model: Trained Autoencoder model
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
                
                # Extract reduced features (128-dim) using encoder
                encoded_features = model.encode(features_tensor)
                encoded_features_list = encoded_features.squeeze(0).cpu().tolist()
                
                # Create output filename (replace features-432 with features-128)
                output_filename = json_file.name.replace(f'_features-{input_features}.json', '_features-128.json')
                output_file = output_path / output_filename
                
                # Save reduced features
                with open(output_file, 'w') as f:
                    json.dump(encoded_features_list, f, indent=2)
                
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
        'input_features': 432,
        'train_csv_path': 'evaluation/csv/train_age.csv',
        'valid_csv_path': 'evaluation/csv/valid_age.csv',
        'json_dir': 'output',
        'output_dir': 'results_autoencoder',
        'reduced_features_dir': 'output_autoencoder'
    }
    
    # Convert paths to absolute
    root_dir = pathlib.Path(__file__).parent
    train_csv_path = str((root_dir / config['train_csv_path']).resolve())
    valid_csv_path = str((root_dir / config['valid_csv_path']).resolve())
    json_dir = str((root_dir / config['json_dir']).resolve())
    output_dir = str((root_dir / config['output_dir']).resolve())
    reduced_features_dir = str((root_dir / config['reduced_features_dir']).resolve())
    
    # Setup device
    device = torch.device(
        f'cuda:{config["gpu_id"]}' 
        if torch.cuda.is_available() and config["gpu_id"] is not None 
        else 'cpu'
    )
    
    # Create datasets (using AutoencoderDataset which only loads features)
    print("Loading datasets...")
    train_dataset = AutoencoderDataset(train_csv_path, json_dir)
    valid_dataset = AutoencoderDataset(valid_csv_path, json_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    
    # Create autoencoder model
    model = Autoencoder(input_features=config['input_features'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    print("="*60)
    print("AUTOENCODER CONFIGURATION")
    print("="*60)
    print(f"Architecture: {config['input_features']} → 256 → 128 → 256 → {config['input_features']}")
    print(f"Input features: {config['input_features']}")
    print(f"Bottleneck features: 128")
    print(f"Learning rate: {config['lr']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Number of epochs: {config['num_epochs']}")
    print(f"Device: {device}")
    print("="*60)
    print()
    
    # Train the autoencoder
    print("Starting training...")
    train_autoencoder(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        num_epochs=config['num_epochs'],
        device=device,
        output_dir=output_dir
    )
    
    print(f"\nTraining complete! Results saved to: {output_dir}")
    
    # Extract and save reduced features for all data
    print("\n" + "="*60)
    print("EXTRACTING REDUCED FEATURES (128-DIM BOTTLENECK)")
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

