import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from utils.plot_utils import plot_raw_test_mean_feature_per_class
import torch
import os
import numpy as np
from typing import List
from utils.config import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def tensor_dataset_autoencoder_peaks_only(batch_size: int, X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None, indices_train=None, indices_val=None, indices_test=None):
    """
    Create tensor datasets for autoencoder (33 input features: 32 peaks + 1 chip, 32 target features: peaks only)
    """
    train_loader, val_loader, test_loader = None, None, None

    if X_train is not None and len(X_train) > 0:
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32)
        # Use all input features (33: 32 peaks + 1 chip for X, 32 peaks for y)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False, drop_last=False)

    if X_val is not None and len(X_val) > 0:
        X_val = torch.tensor(X_val.values, dtype=torch.float32)
        y_val = torch.tensor(y_val.values, dtype=torch.float32)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False, drop_last=False)

    if X_test is not None and len(X_test) > 0:
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, indices_train, indices_val, indices_test


def tensor_dataset_multitask_autoencoder(batch_size: int,
                                         X_train=None, y_train=None, class_train=None,
                                         X_val=None, y_val=None, class_val=None,
                                         X_test=None, y_test=None, class_test=None,
                                         indices_train=None, indices_val=None, indices_test=None):
    """
    Create tensor datasets for multi-task autoencoder (reconstruction + classification).

    Returns data loaders with (inputs, targets, class_labels) tuples.

    Args:
        batch_size: Batch size for data loaders
        X_train, X_val, X_test: Input features (DataFrames or arrays)
        y_train, y_val, y_test: Target features (DataFrames or arrays)
        class_train, class_val, class_test: Class labels (arrays or Series), can be None
        indices_train, indices_val, indices_test: Original indices

    Returns:
        train_loader, val_loader, test_loader, indices_train, indices_val, indices_test
    """
    train_loader, val_loader, test_loader = None, None, None

    if X_train is not None and len(X_train) > 0:
        if class_train is None:
            raise ValueError("class_train is required for multi-task autoencoder but was None")
        X_train_tensor = torch.tensor(X_train.values if hasattr(X_train, 'values') else X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values if hasattr(y_train, 'values') else y_train, dtype=torch.float32)
        class_train_tensor = torch.tensor(class_train.values if hasattr(class_train, 'values') else class_train, dtype=torch.long)
        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor, class_train_tensor),
            batch_size=batch_size, shuffle=True, drop_last=False
        )

    if X_val is not None and len(X_val) > 0:
        if class_val is None:
            raise ValueError("class_val is required for multi-task autoencoder but was None")
        X_val_tensor = torch.tensor(X_val.values if hasattr(X_val, 'values') else X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values if hasattr(y_val, 'values') else y_val, dtype=torch.float32)
        class_val_tensor = torch.tensor(class_val.values if hasattr(class_val, 'values') else class_val, dtype=torch.long)
        val_loader = DataLoader(
            TensorDataset(X_val_tensor, y_val_tensor, class_val_tensor),
            batch_size=batch_size, shuffle=False, drop_last=False
        )

    if X_test is not None and len(X_test) > 0:
        if class_test is None:
            raise ValueError("class_test is required for multi-task autoencoder but was None")
        X_test_tensor = torch.tensor(X_test.values if hasattr(X_test, 'values') else X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values if hasattr(y_test, 'values') else y_test, dtype=torch.float32)
        class_test_tensor = torch.tensor(class_test.values if hasattr(class_test, 'values') else class_test, dtype=torch.long)
        test_loader = DataLoader(
            TensorDataset(X_test_tensor, y_test_tensor, class_test_tensor),
            batch_size=batch_size, shuffle=False, drop_last=False
        )

    return train_loader, val_loader, test_loader, indices_train, indices_val, indices_test

def load_and_preprocess_data_autoencoder_denoised_to_baseline(denoised_df, baseline_file_path, random_state=seed, finetune=False):
    """
    Load denoised data as input and baseline data as target for second autoencoder training.
    Maps each denoised row to its corresponding class baseline row.

    Args:
        denoised_df: DataFrame containing denoised data with Class column
        baseline_file_path: Path to baseline CSV file (4 rows, one per class)
        random_state: Random state for splitting
        finetune: Whether to use train/val split only (no test)

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, label_encoder, indices_train, indices_val, indices_test
        where X contains denoised data and y contains corresponding baseline data matched by class
    """
    baseline_df = pd.read_csv(baseline_file_path)

    # Extract peak columns from denoised data (input)
    denoised_peak_cols = [col for col in denoised_df.columns if str(col).startswith("Peak") and "Temperature" not in str(col)]
    if not denoised_peak_cols:
        # If no "Peak" columns, assume numeric columns (0-31) are the peak columns
        denoised_peak_cols = [col for col in denoised_df.columns if isinstance(col, (int, float)) or (isinstance(col, str) and col.isdigit())]

    # Extract peak columns from baseline data (target)
    baseline_peak_cols = [col for col in baseline_df.columns if col.startswith("Peak") and "Temperature" not in col]
    if not baseline_peak_cols:
        # If no "Peak" columns, assume numeric columns (0-31) are the peak columns
        baseline_peak_cols = [col for col in baseline_df.columns if isinstance(col, (int, float)) or (isinstance(col, str) and col.isdigit())]

    # Map each denoised row to its corresponding baseline class row
    aligned_baseline_rows = []
    for idx, row in denoised_df.iterrows():
        class_label = row['Class']
        # Find the baseline row for this class
        baseline_row = baseline_df[baseline_df['Class'] == class_label][baseline_peak_cols]
        if len(baseline_row) > 0:
            aligned_baseline_rows.append(baseline_row.iloc[0].values)
        else:
            # If no match found, use zeros (shouldn't happen if data is correct)
            aligned_baseline_rows.append(np.zeros(len(baseline_peak_cols)))

    X = denoised_df[denoised_peak_cols]
    y = pd.DataFrame(aligned_baseline_rows, columns=baseline_peak_cols, index=X.index)

    # Create indices array
    indices = np.arange(len(X))

    # Split data with indices
    if finetune:
        X_train, X_val, y_train, y_val, indices_train, indices_val = train_test_split(
            X, y, indices, test_size=0.2, random_state=random_state
        )
        X_test, y_test = pd.DataFrame(), pd.DataFrame()
        indices_test = np.array([])
    else:
        X_train, X_temp, y_train, y_temp, indices_train, indices_temp = train_test_split(
            X, y, indices, test_size=0.3, random_state=random_state
        )
        X_val, X_test, y_val, y_test, indices_val, indices_test = train_test_split(
            X_temp, y_temp, indices_temp, test_size=0.333, random_state=random_state
        )

    label_encoder = LabelEncoder()

    return X_train, y_train, X_val, y_val, X_test, y_test, label_encoder, indices_train, indices_val, indices_test

def load_and_preprocess_data_autoencoder_prenormalized(file_path, random_state=seed, finetune=False):
    """
    Load and preprocess normalized data for autoencoder training with peaks + temperature.
    This function assumes normalization has already been applied to the dataset.

    Input: train_Peak columns (32 peaks) + train_Temperature (MinMax normalized)
    Target: match_Peak columns (32 peaks)

    Args:
        file_path: Path to the CSV file (should contain normalized data)
        random_state: Random state for splitting
        finetune: Whether to use train/val split only (no test)

    Returns:
        Tuple: (X_train, y_train, X_val, y_val, X_test, y_test,
                class_train, class_val, class_test, label_encoder,
                indices_train, indices_val, indices_test)
        where X contains train_Peak columns + train_Temperature_normalized,
        y contains match_Peak columns, and class_* are encoded class labels
    """
    # Load pre-normalized data
    df = pd.read_csv(file_path)

    # Determine class column name (try different variations)
    if "train_Class" in df.columns:
        class_col = "train_Class"
    elif "Class" in df.columns:
        class_col = "Class"
    else:
        raise ValueError(f"No class column found in {file_path}. Available columns: {list(df.columns)}")

    # Encode Class labels
    label_encoder = LabelEncoder()
    df[class_col] = label_encoder.fit_transform(df[class_col])

    # Detect data format: check if we have train_Peak/match_Peak or just Peak columns
    has_train_match = any(col.startswith("train_Peak") for col in df.columns)

    if has_train_match:
        # Format 1: Series training with train_Peak and match_Peak columns
        train_peak_cols = [col for col in df.columns if col.startswith("train_Peak") and "Temperature" not in col]
        train_peak_cols = sorted(train_peak_cols, key=lambda x: int(x.split()[-1]))[:32]  # First 32 peaks

        # Add MinMax normalized temperature column
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df['train_Temperature_normalized'] = scaler.fit_transform(df[['train_Temperature']])
        input_cols = train_peak_cols + ['train_Temperature_normalized']
        X_input = df[input_cols]

        # Extract target columns (match_Peak at 27°C)
        match_peak_cols = [col for col in df.columns if col.startswith("match_Peak") and "Temperature" not in col]
        match_peak_cols = sorted(match_peak_cols, key=lambda x: int(x.split()[-1]))[:32]  # First 32 peaks
        y_target = df[match_peak_cols]
    else:
        # Format 2: Parallel training with just Peak columns
        # Extract peak columns (assuming "Peak 1", "Peak 2", etc.)
        peak_cols = [col for col in df.columns if str(col).startswith("Peak") and "Temperature" not in str(col)]
        peak_cols = sorted(peak_cols, key=lambda x: int(str(x).split()[-1]))[:32]  # First 32 peaks

        # Determine temperature column name
        if "Temperature" in df.columns:
            temp_col = "Temperature"
        elif "train_Temperature" in df.columns:
            temp_col = "train_Temperature"
        else:
            raise ValueError(f"No temperature column found in {file_path}. Available columns: {list(df.columns)}")

        # Add MinMax normalized temperature column
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df['Temperature_normalized'] = scaler.fit_transform(df[[temp_col]])
        input_cols = peak_cols + ['Temperature_normalized']
        X_input = df[input_cols]

        # For parallel training, target is the same as input (peaks only, no temperature)
        y_target = df[peak_cols]

    print(f"=== Normalized Autoencoder Data Preparation ===")
    print(f"Data format: {'Series (train/match)' if has_train_match else 'Parallel (Peak only)'}")
    print(f"Input features: {X_input.shape[1]} columns")
    print(f"Target features: {y_target.shape[1]} columns")

    # Create indices array to track original row positions
    indices = np.arange(len(X_input))
    stratify_labels = df[class_col].values

    # 70-20-10 split with stratification to ensure all classes in all splits
    X_train, X_temp, y_train, y_temp, indices_train, indices_temp = train_test_split(
        X_input, y_target, indices,
        test_size=0.3,  # 30% for temp
        random_state=random_state,
        stratify=stratify_labels
    )

    # Second split: 20% val, 10% test
    stratify_labels_temp = stratify_labels[indices_temp]
    X_val, X_test, y_val, y_test, indices_val, indices_test = train_test_split(
        X_temp, y_temp, indices_temp,
        test_size=0.333,  # 10% of total
        random_state=random_state,
        stratify=stratify_labels_temp
    )

    print(f"=== Normalized Autoencoder Data Split ===")
    total_samples = len(X_input)
    print(f"Training: {len(X_train)} samples ({len(X_train)/total_samples:.1%})")
    print(f"Validation: {len(X_val)} samples ({len(X_val)/total_samples:.1%})")
    print(f"Test: {len(X_test)} samples ({len(X_test)/total_samples:.1%})")

    # Extract class labels for each split
    class_train = stratify_labels[indices_train]
    class_val = stratify_labels[indices_val]
    class_test = stratify_labels[indices_test]

    # Print class distribution
    print(f"\nClass distribution:")
    unique_classes = np.unique(stratify_labels)
    print(f"  Total classes: {len(unique_classes)} - {unique_classes}")
    print(f"  Train classes: {np.unique(class_train)}")
    print(f"  Val classes:   {np.unique(class_val)}")
    print(f"  Test classes:  {np.unique(class_test)}")

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            class_train, class_val, class_test, label_encoder,
            indices_train, indices_val, indices_test)

