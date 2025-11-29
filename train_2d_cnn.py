#!/usr/bin/env python3
"""
Train a 2D CNN for age prediction using FA (Fractional Anisotropy) metric.

This script:
1. Loads FA nifti files from tractseg output directories
2. Processes 3D volumes (63, 117, 114) as 2D slices with 1 channel
3. Splits data by subject with age-stratified splitting
4. Normalizes the data
5. Trains a 2D CNN for age regression

Required dependencies:
    - torch
    - numpy
    - pandas
    - SimpleITK
    - scikit-learn
    - tqdm

Usage:
    python train_2d_cnn.py --tractseg-output-dir tmp/tractseg_fa_output --participants-tsv ds003416/participants.tsv
"""

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
from collections import defaultdict


class NiftiDataset(Dataset):
    """Dataset for loading 2D slices from 3D nifti volumes."""
    
    def __init__(self, data_list, normalize_stats=None):
        """
        Parameters:
        -----------
        data_list : list of tuples
            Each tuple contains (volume_path, age, subject_id)
            volume_path is the path to the directory containing metric files
        normalize_stats : dict, optional
            Dictionary with 'mean' and 'std' for normalization
        """
        self.data_list = data_list
        self.normalize_stats = normalize_stats
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        metric_dir, age, subject_id = self.data_list[idx]
        
        # Load the FA metric file
        fa_path = os.path.join(metric_dir, 'fa.nii.gz')
        
        # Load nifti file
        fa_img = sitk.ReadImage(fa_path)
        
        # Convert to numpy array
        fa_array = sitk.GetArrayFromImage(fa_img)  # Shape: (63, 117, 114)
        
        # Add channel dimension: (1, 63, 117, 114)
        volume = np.expand_dims(fa_array, axis=0)
        
        # Convert to torch tensor and normalize
        volume = torch.from_numpy(volume).float()
        
        if self.normalize_stats is not None:
            mean = torch.tensor(self.normalize_stats['mean']).view(1, 1, 1, 1)
            std = torch.tensor(self.normalize_stats['std']).view(1, 1, 1, 1)
            volume = (volume - mean) / (std + 1e-8)
        
        age_tensor = torch.tensor(age, dtype=torch.float32)
        
        return volume, age_tensor


class AgePredictor2DCNN(nn.Module):
    """2D CNN for age prediction from FA 3D volumes."""
    
    def __init__(self, in_channels=1, num_slices=63):
        super(AgePredictor2DCNN, self).__init__()
        
        # 2D CNN layers for processing each slice
        self.conv_layers = nn.Sequential(
            # First block
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112x112 -> 56x56
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56x56 -> 28x28
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # Global average pooling: 14x14 -> 1x1
        )
        
        # Process all slices and aggregate
        # After conv layers: (batch, 256, 1, 1) per slice
        # We have num_slices slices, so we'll get (batch, num_slices, 256)
        self.slice_processor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Final regression head
        self.regressor = nn.Sequential(
            nn.Linear(128 * num_slices, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        """
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 63, 117, 114)
        
        Returns:
        --------
        torch.Tensor
            Age predictions of shape (batch,)
        """
        batch_size = x.size(0)
        num_slices = x.size(2)
        
        # Reshape to process all slices in parallel: (batch * num_slices, 1, 117, 114)
        x_reshaped = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, 63, 1, 117, 114)
        x_reshaped = x_reshaped.view(batch_size * num_slices, 1, 117, 114)
        
        # Apply 2D CNN to all slices in parallel: (batch * num_slices, 256, 1, 1)
        conv_out = self.conv_layers(x_reshaped)
        
        # Flatten: (batch * num_slices, 256)
        conv_out = conv_out.view(batch_size * num_slices, -1)
        
        # Process slice features: (batch * num_slices, 128)
        slice_features = self.slice_processor(conv_out)
        
        # Reshape back: (batch, num_slices, 128)
        slice_features = slice_features.view(batch_size, num_slices, -1)
        
        # Flatten slice features: (batch, num_slices * 128)
        combined_features = slice_features.view(batch_size, -1)
        
        # Final regression: (batch, 1)
        age_pred = self.regressor(combined_features)
        
        return age_pred.squeeze(1)  # (batch,)


def parse_metric_dir_name(dir_path):
    """Extract subject and session IDs from metric directory path."""
    # Example: tmp/tractseg_fa_output/sub-cIIIs01_ses-s1Bx1_acq-b1000n3r21x21x22peAPA_run-104/metric
    dir_name = os.path.basename(os.path.dirname(dir_path))
    match = re.match(r'(sub-[^_]+)_(ses-[^_]+)_', dir_name)
    if match:
        return match.group(1), match.group(2)
    return None, None


def load_participants(participants_tsv_path):
    """Load participants.tsv and create a lookup dictionary."""
    df = pd.read_csv(participants_tsv_path, sep='\t')
    
    # Create a dictionary keyed by (participant_id, session_id)
    participants_dict = {}
    for _, row in df.iterrows():
        key = (row['participant_id'], row['session_id'])
        participants_dict[key] = {
            'age': float(row['age']),
            'sex': row['sex']
        }
    
    return participants_dict


def create_age_bins(ages, age_threshold=10.0):
    """Create age bins: 0 for < threshold, 1 for >= threshold."""
    return np.array([0 if age < age_threshold else 1 for age in ages])


def split_with_age_coverage(subjects, subject_to_bin, subject_to_age, 
                            train_split, val_split, eval_split, random_state=42):
    """
    Split subjects ensuring age coverage and balanced mean ages across sets.
    Adapted from generate_data_csv.py
    """
    np.random.seed(random_state)
    
    # Group subjects by age bin
    subjects_by_bin = {}
    for subject in subjects:
        bin_id = subject_to_bin[subject]
        if bin_id not in subjects_by_bin:
            subjects_by_bin[bin_id] = []
        subjects_by_bin[bin_id].append(subject)
    
    # Initialize sets
    train_subjects = []
    val_subjects = []
    eval_subjects = []
    
    # First pass: ensure at least one subject from each bin in each set
    for bin_id, bin_subjects in subjects_by_bin.items():
        if len(bin_subjects) >= 3:
            shuffled = np.random.permutation(bin_subjects)
            train_subjects.append(shuffled[0])
            val_subjects.append(shuffled[1])
            eval_subjects.append(shuffled[2])
            subjects_by_bin[bin_id] = list(shuffled[3:])
        elif len(bin_subjects) == 2:
            shuffled = np.random.permutation(bin_subjects)
            train_subjects.append(shuffled[0])
            val_subjects.append(shuffled[1])
            subjects_by_bin[bin_id] = []
        elif len(bin_subjects) == 1:
            train_subjects.append(bin_subjects[0])
            subjects_by_bin[bin_id] = []
    
    # Second pass: distribute remaining subjects maintaining age balance
    remaining_by_bin = {bin_id: sorted(subs, key=lambda s: subject_to_age[s]) 
                        for bin_id, subs in subjects_by_bin.items() if len(subs) > 0}
    
    if len(remaining_by_bin) > 0:
        total_subjects = len(subjects)
        target_train = int(total_subjects * train_split)
        target_val = int(total_subjects * val_split)
        target_eval = int(total_subjects * eval_split)
        
        def get_mean_age(subject_list):
            if len(subject_list) == 0:
                return 0.0
            return np.mean([subject_to_age[s] for s in subject_list])
        
        overall_mean = np.mean([subject_to_age[s] for s in subjects])
        
        all_remaining = []
        bin_iterators = {bin_id: iter(bin_subs) for bin_id, bin_subs in remaining_by_bin.items()}
        bin_ids = sorted(remaining_by_bin.keys())
        active_bins = bin_ids.copy()
        
        while active_bins:
            for bin_id in active_bins[:]:
                try:
                    subject = next(bin_iterators[bin_id])
                    all_remaining.append(subject)
                except StopIteration:
                    active_bins.remove(bin_id)
        
        for subject in all_remaining:
            current_train = len(train_subjects)
            current_val = len(val_subjects)
            current_eval = len(eval_subjects)
            
            train_deficit = target_train - current_train
            val_deficit = target_val - current_val
            eval_deficit = target_eval - current_eval
            
            train_mean = get_mean_age(train_subjects)
            val_mean = get_mean_age(val_subjects)
            eval_mean = get_mean_age(eval_subjects)
            
            subject_age = subject_to_age[subject]
            
            candidates = []
            
            if train_deficit > 0:
                new_train_mean = (train_mean * current_train + subject_age) / (current_train + 1)
                mean_balance_score = -abs(new_train_mean - overall_mean)
                deficit_score = train_deficit
                temp_means = [new_train_mean, val_mean, eval_mean]
                variance_score = -np.var(temp_means)
                candidates.append(('train', deficit_score * 100 + mean_balance_score * 10 + variance_score))
            
            if val_deficit > 0:
                new_val_mean = (val_mean * current_val + subject_age) / (current_val + 1)
                temp_means = [train_mean, new_val_mean, eval_mean]
                mean_balance_score = -abs(new_val_mean - overall_mean)
                deficit_score = val_deficit
                variance_score = -np.var(temp_means)
                candidates.append(('val', deficit_score * 100 + mean_balance_score * 10 + variance_score))
            
            if eval_deficit > 0:
                new_eval_mean = (eval_mean * current_eval + subject_age) / (current_eval + 1)
                temp_means = [train_mean, val_mean, new_eval_mean]
                mean_balance_score = -abs(new_eval_mean - overall_mean)
                deficit_score = eval_deficit
                variance_score = -np.var(temp_means)
                candidates.append(('eval', deficit_score * 100 + mean_balance_score * 10 + variance_score))
            
            if not candidates:
                train_subjects.append(subject)
            else:
                best_set = max(candidates, key=lambda x: x[1])[0]
                if best_set == 'train':
                    train_subjects.append(subject)
                elif best_set == 'val':
                    val_subjects.append(subject)
                else:
                    eval_subjects.append(subject)
    
    return list(train_subjects), list(val_subjects), list(eval_subjects)


def find_all_metric_directories(tractseg_output_dir):
    """Find all metric directories containing FA files."""
    metric_dirs = []
    tractseg_path = Path(tractseg_output_dir)
    
    for metric_dir in tractseg_path.rglob('metric'):
        # Check if FA file exists
        fa_path = metric_dir / 'fa.nii.gz'
        
        if fa_path.exists():
            metric_dirs.append(str(metric_dir))
    
    return metric_dirs


def compute_normalization_stats(dataset):
    """Compute mean and std for normalization."""
    print("Computing normalization statistics...")
    all_data = []
    
    for idx in tqdm(range(len(dataset)), desc="Loading data"):
        volume, _ = dataset[idx]
        all_data.append(volume.numpy())

    all_data = np.stack(all_data, axis=0)  # (N, 1, 63, 117, 114)
    
    # Compute mean and std across all samples
    mean = np.mean(all_data)  # scalar
    std = np.std(all_data)    # scalar
    
    return {'mean': [mean], 'std': [std]}


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    
    for volumes, ages in tqdm(train_loader, desc="Training"):
        volumes = volumes.to(device)
        ages = ages.to(device)
        
        optimizer.zero_grad()
        predictions = model(volumes)
        loss = criterion(predictions, ages)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        mae = torch.mean(torch.abs(predictions - ages))
        total_mae += mae.item()
        num_batches += 1
    
    return total_loss / num_batches, total_mae / num_batches


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for volumes, ages in tqdm(val_loader, desc="Validating"):
            volumes = volumes.to(device)
            ages = ages.to(device)
            
            predictions = model(volumes)
            loss = criterion(predictions, ages)
            
            total_loss += loss.item()
            mae = torch.mean(torch.abs(predictions - ages))
            total_mae += mae.item()
            num_batches += 1
    
    return total_loss / num_batches, total_mae / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train 2D CNN for age prediction')
    parser.add_argument('--tractseg-output-dir', type=str,
                       default='tmp/tractseg_fa_output',
                       help='Directory containing tractseg output with metric directories')
    parser.add_argument('--participants-tsv', type=str,
                       default='ds003416/participants.tsv',
                       help='Path to participants.tsv file')
    parser.add_argument('--train-split', type=float, default=0.7,
                       help='Fraction of subjects for training')
    parser.add_argument('--val-split', type=float, default=0.15,
                       help='Fraction of subjects for validation')
    parser.add_argument('--eval-split', type=float, default=0.15,
                       help='Fraction of subjects for evaluation')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default='results_2d_cnn',
                       help='Directory to save model and results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Normalize splits
    total_split = args.train_split + args.val_split + args.eval_split
    if abs(total_split - 1.0) > 0.01:
        print(f"Warning: Splits sum to {total_split:.3f}, normalizing to sum to 1.0")
        args.train_split /= total_split
        args.val_split /= total_split
        args.eval_split /= total_split
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("2D CNN Training Script for Age Prediction")
    print("=" * 80)
    
    # Find all metric directories
    print(f"\nFinding metric directories in {args.tractseg_output_dir}...")
    metric_dirs = find_all_metric_directories(args.tractseg_output_dir)
    print(f"Found {len(metric_dirs)} metric directories")
    
    # Load participant metadata
    print(f"\nLoading participant metadata from {args.participants_tsv}...")
    participants_dict = load_participants(args.participants_tsv)
    
    # Match metric directories with participant data
    matched_data = []
    for metric_dir in metric_dirs:
        subject_id, session_id = parse_metric_dir_name(metric_dir)
        if subject_id is None or session_id is None:
            continue
        
        key = (subject_id, session_id)
        if key in participants_dict:
            age = participants_dict[key]['age']
            matched_data.append({
                'metric_dir': metric_dir,
                'age': age,
                'subject': subject_id,
                'session': session_id
            })
    
    print(f"Matched {len(matched_data)} metric directories with participant data")
    
    if len(matched_data) == 0:
        print("Error: No matched data found!")
        return
    
    # Create DataFrame for easier manipulation
    df = pd.DataFrame(matched_data)
    
    # Calculate mean age per subject for stratification
    subject_ages = df.groupby('subject')['age'].mean().reset_index()
    subject_ages.columns = ['subject', 'mean_age']
    
    # Create age bins
    age_bins = create_age_bins(subject_ages['mean_age'].values, age_threshold=10.0)
    subject_ages['age_bin'] = age_bins
    
    print(f"\nAge distribution:")
    for bin_id in [0, 1]:
        bin_subjects = subject_ages[subject_ages['age_bin'] == bin_id]
        if len(bin_subjects) > 0:
            bin_label = "< 10 years" if bin_id == 0 else ">= 10 years"
            print(f"  {bin_label}: {len(bin_subjects)} subjects, "
                  f"age range: {bin_subjects['mean_age'].min():.1f} - {bin_subjects['mean_age'].max():.1f} years")
    
    # Split by subject
    subjects = df['subject'].unique()
    subject_to_bin = dict(zip(subject_ages['subject'], subject_ages['age_bin']))
    subject_to_age = dict(zip(subject_ages['subject'], subject_ages['mean_age']))
    
    train_subjects, val_subjects, eval_subjects = split_with_age_coverage(
        subjects, subject_to_bin, subject_to_age,
        args.train_split, args.val_split, args.eval_split,
        random_state=args.random_state
    )
    
    # Create data lists for each split
    train_data = [(row['metric_dir'], row['age'], row['subject']) 
                  for _, row in df.iterrows() if row['subject'] in train_subjects]
    val_data = [(row['metric_dir'], row['age'], row['subject']) 
                for _, row in df.iterrows() if row['subject'] in val_subjects]
    eval_data = [(row['metric_dir'], row['age'], row['subject']) 
                 for _, row in df.iterrows() if row['subject'] in eval_subjects]
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_data)} samples from {len(train_subjects)} subjects")
    print(f"  Validation: {len(val_data)} samples from {len(val_subjects)} subjects")
    print(f"  Evaluation: {len(eval_data)} samples from {len(eval_subjects)} subjects")
    
    # Compute normalization statistics from training data
    print(f"\nComputing normalization statistics from training data...")
    train_dataset_temp = NiftiDataset(train_data)
    normalize_stats = compute_normalization_stats(train_dataset_temp)
    print(f"Normalization stats - Mean: {normalize_stats['mean']}, Std: {normalize_stats['std']}")
    
    # Create datasets with normalization
    train_dataset = NiftiDataset(train_data, normalize_stats=normalize_stats)
    val_dataset = NiftiDataset(val_data, normalize_stats=normalize_stats)
    eval_dataset = NiftiDataset(eval_data, normalize_stats=normalize_stats)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Create model
    device = torch.device(args.device)
    model = AgePredictor2DCNN(in_channels=1, num_slices=63).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)#, verbose=True)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print(f"\nStarting training...")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Number of epochs: {args.num_epochs}")
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_maes.append(train_mae)
        
        # Validate
        val_loss, val_mae = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'normalize_stats': normalize_stats,
            }, best_model_path)
            print(f"Saved best model (val_loss: {val_loss:.4f})")
    
    # Evaluate on evaluation set
    print(f"\nEvaluating on evaluation set...")
    eval_loss, eval_mae = validate(model, eval_loader, criterion, device)
    print(f"Eval Loss: {eval_loss:.4f}, Eval MAE: {eval_mae:.4f}")
    
    # Helper function to convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        """Convert numpy types to native Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        else:
            return obj
    
    # Save final results
    results = {
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses],
        'train_maes': [float(x) for x in train_maes],
        'val_maes': [float(x) for x in val_maes],
        'final_eval_loss': float(eval_loss),
        'final_eval_mae': float(eval_mae),
        'normalize_stats': convert_to_native(normalize_stats),
        'num_train_samples': len(train_data),
        'num_val_samples': len(val_data),
        'num_eval_samples': len(eval_data),
    }
    
    import json
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print(f"Best model saved to {os.path.join(args.output_dir, 'best_model.pth')}")
    print("\nTraining completed!")


if __name__ == '__main__':
    main()

