#!/usr/bin/env python3
"""
Generate CSV files for training/validation/evaluation from output JSON files and participants.tsv

This script:
1. Reads all JSON feature files from the output directory
2. Matches them with participant metadata from participants.tsv
3. Creates train/validation/evaluation splits with age-stratified splitting
4. Generates CSV files with columns: json, age, sex (and optionally cognitive_status)
"""

import os
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from sklearn.model_selection import train_test_split

def parse_json_filename(filename):
    """
    Parse JSON filename to extract subject and session IDs.
    
    Example: sub-cIIIs01_ses-s1Bx1_acq-b1000n40r25x25x25peAPP_run-208_features-128.json
    Returns: ('sub-cIIIs01', 'ses-s1Bx1')
    """
    # Pattern: sub-{subject}_ses-{session}_...
    match = re.match(r'(sub-[^_]+)_(ses-[^_]+)_', filename)
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
            'sex': 1 if row['sex'].lower() == 'male' else 0,  # 1=male, 0=female
            'sex_str': row['sex']  # Keep original for debugging
        }

    print(df["age"].unique())
    
    return participants_dict

def create_age_bins(ages, n_bins=2, age_threshold=10.0):
    """
    Create age bins for ensuring age coverage across splits.
    Uses 2 bins: below threshold and above threshold.
    
    Parameters:
    -----------
    ages : array-like
        Array of ages
    n_bins : int
        Number of bins (ignored, always 2)
    age_threshold : float
        Age threshold for binning (default: 10.0)
    
    Returns:
    --------
    bins : array
        Age bin labels: 0 for < threshold, 1 for >= threshold
    """
    # Create 2 bins: below and above threshold
    bins = np.array([0 if age < age_threshold else 1 for age in ages])
    return bins

def split_with_age_coverage(subjects, subject_to_bin, subject_to_age, 
                            train_split, val_split, eval_split, random_state=42):
    """
    Split subjects ensuring at least one subject from each age bin in each set.
    
    Parameters:
    -----------
    subjects : array-like
        List of subject IDs
    subject_to_bin : dict
        Mapping from subject ID to age bin
    subject_to_age : dict
        Mapping from subject ID to age
    train_split : float
        Target fraction for training
    val_split : float
        Target fraction for validation
    eval_split : float
        Target fraction for evaluation
    random_state : int
        Random seed
    
    Returns:
    --------
    train_subjects : list
        Subjects assigned to training set
    val_subjects : list
        Subjects assigned to validation set
    eval_subjects : list
        Subjects assigned to evaluation set
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
            # Enough subjects: assign one to each set
            shuffled = np.random.permutation(bin_subjects)
            train_subjects.append(shuffled[0])
            val_subjects.append(shuffled[1])
            eval_subjects.append(shuffled[2])
            # Remove assigned subjects
            subjects_by_bin[bin_id] = list(shuffled[3:])
        elif len(bin_subjects) == 2:
            # Two subjects: assign to train and val
            shuffled = np.random.permutation(bin_subjects)
            train_subjects.append(shuffled[0])
            val_subjects.append(shuffled[1])
            subjects_by_bin[bin_id] = []
        elif len(bin_subjects) == 1:
            # One subject: assign to train
            train_subjects.append(bin_subjects[0])
            subjects_by_bin[bin_id] = []
    
    # Second pass: distribute remaining subjects maintaining age balance
    # Strategy: Actively balance mean ages across sets while respecting size targets
    remaining_by_bin = {bin_id: sorted(subs, key=lambda s: subject_to_age[s]) 
                        for bin_id, subs in subjects_by_bin.items() if len(subs) > 0}
    
    if len(remaining_by_bin) > 0:
        # Calculate target sizes
        total_subjects = len(subjects)
        target_train = int(total_subjects * train_split)
        target_val = int(total_subjects * val_split)
        target_eval = int(total_subjects * eval_split)
        
        # Helper function to calculate mean age
        def get_mean_age(subject_list):
            if len(subject_list) == 0:
                return 0.0
            return np.mean([subject_to_age[s] for s in subject_list])
        
        # Calculate overall mean age for reference
        overall_mean = np.mean([subject_to_age[s] for s in subjects])
        
        # Collect all remaining subjects, interleaving from different bins
        all_remaining = []
        bin_iterators = {bin_id: iter(bin_subs) for bin_id, bin_subs in remaining_by_bin.items()}
        bin_ids = sorted(remaining_by_bin.keys())
        active_bins = bin_ids.copy()
        
        # Interleave subjects from different bins
        while active_bins:
            for bin_id in active_bins[:]:
                try:
                    subject = next(bin_iterators[bin_id])
                    all_remaining.append(subject)
                except StopIteration:
                    active_bins.remove(bin_id)
        
        # Distribute subjects one by one, choosing the set that best balances mean ages
        for subject in all_remaining:
            current_train = len(train_subjects)
            current_val = len(val_subjects)
            current_eval = len(eval_subjects)
            
            # Calculate deficits
            train_deficit = target_train - current_train
            val_deficit = target_val - current_val
            eval_deficit = target_eval - current_eval
            
            # Get current mean ages
            train_mean = get_mean_age(train_subjects)
            val_mean = get_mean_age(val_subjects)
            eval_mean = get_mean_age(eval_subjects)
            
            subject_age = subject_to_age[subject]
            
            # Calculate what the new mean would be for each set
            candidates = []
            
            if train_deficit > 0:
                new_train_mean = (train_mean * current_train + subject_age) / (current_train + 1)
                # Score: prefer sets that need more subjects AND would have mean closer to overall mean
                # Also consider how balanced the means would be across all sets
                mean_balance_score = -abs(new_train_mean - overall_mean)
                deficit_score = train_deficit
                # Also consider variance of means across sets
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
            
            # Choose the best candidate
            if not candidates:
                # All sets are full, assign to train
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

def generate_csv_files(output_dir, participants_tsv_path, output_csv_dir, 
                      train_split=0.7, val_split=0.15, eval_split=0.15, 
                      random_state=42, task='age', n_age_bins=2, features=128):
    """
    Generate train, validation, and evaluation CSV files with age-stratified splitting.
    
    Parameters:
    -----------
    output_dir : str
        Directory containing the JSON feature files
    participants_tsv_path : str
        Path to participants.tsv file
    output_csv_dir : str
        Directory to save the generated CSV files
    train_split : float
        Fraction of data to use for training (default: 0.7)
    val_split : float
        Fraction of data to use for validation (default: 0.15)
    eval_split : float
        Fraction of data to use for evaluation (default: 0.15)
    random_state : int
        Random seed for splits
    task : str
        Task type: 'age', 'sex', or 'cognitive_status'
    n_age_bins : int
        Number of age bins for stratification (default: 5)
    features : int
        Number of features
    """
    
    # Load participant metadata
    print(f"Loading participant metadata from {participants_tsv_path}...")
    participants_dict = load_participants(participants_tsv_path)
    print(f"Loaded metadata for {len(participants_dict)} participant-session pairs")
    
    # Find all JSON files with the specified feature size
    output_path = Path(output_dir)
    json_files = list(output_path.glob(f'*_features-{features}.json'))
    print(f"Found {len(json_files)} JSON feature files with {features} features")
    
    # Match JSON files with participant data
    matched_data = []
    unmatched_files = []
    skipped_files = 0
    
    for json_file in json_files:
        filename = json_file.name
        subject_id, session_id = parse_json_filename(filename)
        
        if subject_id is None or session_id is None:
            print(f"Warning: Could not parse filename: {filename}")
            unmatched_files.append(filename)
            continue

        # INSERT_YOUR_CODE
        # Check if the JSON file contains all zeros (assume JSON contains a list of numbers)
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                is_all_zero = all(float(val) == 0.0 for val in data) if data else True
            else:
                is_all_zero = True
        except Exception as e:
            print(f"Warning: Could not load JSON or parse numeric list for {filename}: {e}")
            is_all_zero = True

        if is_all_zero:
            print(f"Skipping {filename} as it contains only zeros")
            skipped_files += 1
            continue
        
        key = (subject_id, session_id)
        if key in participants_dict:
            participant_info = participants_dict[key]
            matched_data.append({
                'json': filename,
                'age': participant_info['age'],
                'sex': participant_info['sex'],
                'subject': subject_id,
                'session': session_id
            })
        else:
            print(f"Warning: No metadata found for {subject_id}, {session_id} (file: {filename})")
            unmatched_files.append(filename)
    
    print(f"\nMatched {len(matched_data)} files with participant metadata")
    if unmatched_files:
        print(f"Unmatched files: {len(unmatched_files)}")
    
    if len(matched_data) == 0:
        print("Error: No matched data found. Cannot generate CSV files.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(matched_data)
    
    # Calculate mean age per subject for stratification
    subject_ages = df.groupby('subject')['age'].mean().reset_index()
    subject_ages.columns = ['subject', 'mean_age']
    
    # Create age bins for stratification (2 bins: below and above 10)
    print(f"\nCreating age bins for stratification (threshold: 10 years)...")
    age_bins = create_age_bins(subject_ages['mean_age'].values, age_threshold=10.0)
    subject_ages['age_bin'] = age_bins
    
    # Print age distribution
    print(f"Age distribution across bins:")
    for bin_id in [0, 1]:
        bin_subjects = subject_ages[subject_ages['age_bin'] == bin_id]
        if len(bin_subjects) > 0:
            bin_label = "< 10 years" if bin_id == 0 else ">= 10 years"
            print(f"  {bin_label}: {len(bin_subjects)} subjects, age range: "
                  f"{bin_subjects['mean_age'].min():.1f} - {bin_subjects['mean_age'].max():.1f} years")
    
    # Create train/validation/evaluation split
    # Stratify by subject and age bin to avoid data leakage and ensure uniform age distribution
    subjects = df['subject'].unique()
    num_subjects = len(subjects)
    
    # Normalize splits to ensure they sum to 1
    total_split = train_split + val_split + eval_split
    if abs(total_split - 1.0) > 0.01:
        print(f"\nWarning: Splits sum to {total_split:.3f}, normalizing to sum to 1.0")
        train_split = train_split / total_split
        val_split = val_split / total_split
        eval_split = eval_split / total_split
    
    if num_subjects == 1:
        # Only one subject: split by samples instead
        print(f"\nWarning: Only 1 unique subject found. Splitting by samples instead of subjects.")
        print("Note: This may cause data leakage if multiple sessions from the same subject are in different splits.")
        # First split: train vs (val+eval)
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_split + eval_split),
            random_state=random_state
        )
        # Second split: val vs eval
        val_df, eval_df = train_test_split(
            temp_df,
            test_size=eval_split / (val_split + eval_split),
            random_state=random_state
        )
        print(f"\nTrain set: {len(train_df)} samples")
        print(f"Validation set: {len(val_df)} samples")
        print(f"Evaluation set: {len(eval_df)} samples")
    elif num_subjects < 3:
        # Very few subjects: warn but still split by subjects
        print(f"\nWarning: Only {num_subjects} unique subjects found. Splits may be imbalanced.")
        if num_subjects == 2:
            # Put 1 subject in train, 1 in val (no eval possible)
            train_subjects = [subjects[0]]
            val_subjects = [subjects[1]]
            eval_subjects = []
        else:
            # Random split without stratification
            train_subjects, temp_subjects = train_test_split(
                subjects,
                test_size=(val_split + eval_split),
                random_state=random_state
            )
            val_subjects, eval_subjects = train_test_split(
                temp_subjects,
                test_size=eval_split / (val_split + eval_split),
                random_state=random_state
            )
        train_df = df[df['subject'].isin(train_subjects)].copy()
        val_df = df[df['subject'].isin(val_subjects)].copy()
        eval_df = df[df['subject'].isin(eval_subjects)].copy() if eval_subjects else pd.DataFrame()
        print(f"\nTrain set: {len(train_df)} samples from {len(train_subjects)} subjects")
        print(f"Validation set: {len(val_df)} samples from {len(val_subjects)} subjects")
        if len(eval_subjects) > 0:
            print(f"Evaluation set: {len(eval_df)} samples from {len(eval_subjects)} subjects")
        else:
            print(f"Evaluation set: 0 samples (insufficient subjects)")
    else:
        # Normal case: split by subjects ensuring age coverage
        # Create mappings
        subject_to_bin = dict(zip(subject_ages['subject'], subject_ages['age_bin']))
        subject_to_age = dict(zip(subject_ages['subject'], subject_ages['mean_age']))
        
        # Split ensuring at least one subject from each age bin in each set
        train_subjects, val_subjects, eval_subjects = split_with_age_coverage(
            subjects,
            subject_to_bin,
            subject_to_age,
            train_split,
            val_split,
            eval_split,
            random_state=random_state
        )
        
        train_df = df[df['subject'].isin(train_subjects)].copy()
        val_df = df[df['subject'].isin(val_subjects)].copy()
        eval_df = df[df['subject'].isin(eval_subjects)].copy()
        
        # Verify age coverage
        train_bins = set([subject_to_bin[s] for s in train_subjects])
        val_bins = set([subject_to_bin[s] for s in val_subjects])
        eval_bins = set([subject_to_bin[s] for s in eval_subjects])
        all_bins = set(subject_to_bin.values())
        
        print(f"\nTrain set: {len(train_df)} samples from {len(train_subjects)} subjects")
        print(f"Validation set: {len(val_df)} samples from {len(val_subjects)} subjects")
        print(f"Evaluation set: {len(eval_df)} samples from {len(eval_subjects)} subjects")

        print(f"\nSkipped files: {skipped_files}")
        
        print(f"\nAge bin coverage:")
        print(f"  Total age bins: {len(all_bins)}")
        print(f"  Train covers {len(train_bins)}/{len(all_bins)} bins")
        print(f"  Val covers {len(val_bins)}/{len(all_bins)} bins")
        print(f"  Eval covers {len(eval_bins)}/{len(all_bins)} bins")
        
        missing_train = all_bins - train_bins
        missing_val = all_bins - val_bins
        missing_eval = all_bins - eval_bins
        
        if missing_train:
            print(f"  Warning: Train set missing bins: {sorted(missing_train)}")
        if missing_val:
            print(f"  Warning: Val set missing bins: {sorted(missing_val)}")
        if missing_eval:
            print(f"  Warning: Eval set missing bins: {sorted(missing_eval)}")
    
    # Select columns based on task
    if task == 'age':
        columns = ['json', 'age']
    elif task == 'sex':
        columns = ['json', 'sex']
    elif task == 'cognitive_status':
        # Note: cognitive_status is not in participants.tsv, so this would need to be added
        # For now, we'll create a placeholder
        print("Warning: cognitive_status not available in participants.tsv")
        print("Creating placeholder column (all zeros). Update manually if needed.")
        train_df['cognitive_status'] = 0
        val_df['cognitive_status'] = 0
        if len(eval_df) > 0:
            eval_df['cognitive_status'] = 0
        columns = ['json', 'cognitive_status']
    else:
        # Include all available columns
        columns = ['json', 'age', 'sex']
    
    # Create output directory
    os.makedirs(output_csv_dir, exist_ok=True)
    
    # Save CSV files
    train_csv_path = os.path.join(output_csv_dir, f'train_{task}.csv')
    val_csv_path = os.path.join(output_csv_dir, f'valid_{task}.csv')
    eval_csv_path = os.path.join(output_csv_dir, f'eval_{task}.csv')
    
    train_df[columns].to_csv(train_csv_path, index=False)
    val_df[columns].to_csv(val_csv_path, index=False)
    
    if len(eval_df) > 0:
        eval_df[columns].to_csv(eval_csv_path, index=False)
        print(f"\nCSV files generated:")
        print(f"  Train: {train_csv_path}")
        print(f"  Validation: {val_csv_path}")
        print(f"  Evaluation: {eval_csv_path}")
    else:
        print(f"\nCSV files generated:")
        print(f"  Train: {train_csv_path}")
        print(f"  Validation: {val_csv_path}")
        print(f"  Evaluation: Not created (insufficient subjects)")
    
    # Print statistics
    print(f"\nStatistics:")
    if 'age' in columns:
        print(f"  Age range - Train: {train_df['age'].min():.1f} to {train_df['age'].max():.1f} years (mean: {train_df['age'].mean():.1f})")
        print(f"  Age range - Val: {val_df['age'].min():.1f} to {val_df['age'].max():.1f} years (mean: {val_df['age'].mean():.1f})")
        if len(eval_df) > 0:
            print(f"  Age range - Eval: {eval_df['age'].min():.1f} to {eval_df['age'].max():.1f} years (mean: {eval_df['age'].mean():.1f})")
    if 'sex' in columns:
        print(f"  Sex distribution - Train: {train_df['sex'].sum()} male, {len(train_df) - train_df['sex'].sum()} female")
        print(f"  Sex distribution - Val: {val_df['sex'].sum()} male, {len(val_df) - val_df['sex'].sum()} female")
        if len(eval_df) > 0:
            print(f"  Sex distribution - Eval: {eval_df['sex'].sum()} male, {len(eval_df) - eval_df['sex'].sum()} female")
    
    return train_csv_path, val_csv_path, eval_csv_path if len(eval_df) > 0 else None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate CSV files for evaluation')
    parser.add_argument('--output-dir', type=str, 
                       default='output',
                       help='Directory containing JSON feature files (default: output)')
    parser.add_argument('--participants-tsv', type=str,
                       default='ds003416/participants.tsv',
                       help='Path to participants.tsv file (default: ds003416/participants.tsv)')
    parser.add_argument('--output-csv-dir', type=str,
                       default='evaluation/csv',
                       help='Directory to save generated CSV files (default: evaluation/csv)')
    parser.add_argument('--train-split', type=float, default=0.7,
                       help='Fraction of data for training (default: 0.7)')
    parser.add_argument('--val-split', type=float, default=0.15,
                       help='Fraction of data for validation (default: 0.15)')
    parser.add_argument('--eval-split', type=float, default=0.15,
                       help='Fraction of data for evaluation (default: 0.15)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for splits (default: 42)')
    parser.add_argument('--task', type=str, choices=['age', 'sex', 'cognitive_status'],
                       default='age',
                       help='Task type (default: age)')
    parser.add_argument('--n-age-bins', type=int, default=2,
                       help='Number of age bins for stratification (default: 2, threshold: 10 years)')
    parser.add_argument('--features', type=int, default=128,
                       help='Number of features (default: 128)')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    workspace_root = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    output_dir = workspace_root / args.output_dir
    participants_tsv = workspace_root / args.participants_tsv
    output_csv_dir = workspace_root / args.output_csv_dir
    
    print("=" * 80)
    print("CSV Generation Script")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Participants TSV: {participants_tsv}")
    print(f"CSV output directory: {output_csv_dir}")
    print(f"Task: {args.task}")
    print(f"Train/Val/Eval splits: {args.train_split:.2f}/{args.val_split:.2f}/{args.eval_split:.2f}")
    print(f"Age bins for stratification: {args.n_age_bins}")
    print(f"Features: {args.features}")
    print("=" * 80)
    
    generate_csv_files(
        str(output_dir),
        str(participants_tsv),
        str(output_csv_dir),
        train_split=args.train_split,
        val_split=args.val_split,
        eval_split=args.eval_split,
        random_state=args.random_state,
        task=args.task,
        n_age_bins=args.n_age_bins,
        features=args.features
    )