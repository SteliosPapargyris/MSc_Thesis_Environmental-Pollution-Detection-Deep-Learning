"""
Utility functions for autoencoder training pipeline.

This module contains helper functions for:
- Creating baseline targets
- Saving autoencoder outputs
- Merging outputs from multiple chips
"""
import os
import torch
import pandas as pd
import numpy as np


def create_baseline_targets(indices, original_df, class_col_name, baseline_denoised_df):
    """
    Create baseline targets aligned with given indices using baseline autoencoder output.

    Args:
        indices: Array of indices to create targets for
        original_df: Original dataframe with class labels
        class_col_name: Name of the class column
        baseline_denoised_df: Denoised baseline dataframe from baseline autoencoder

    Returns:
        numpy array of aligned baseline targets
    """
    baseline_peak_cols_denoised = [col for col in baseline_denoised_df.columns if col != 'Class']

    aligned_targets = []
    for idx in indices:
        class_label = original_df.iloc[idx][class_col_name]
        # Get the baseline row for this class from the baseline autoencoder output
        baseline_row = baseline_denoised_df[baseline_denoised_df['Class'] == class_label][baseline_peak_cols_denoised]
        if len(baseline_row) > 0:
            aligned_targets.append(baseline_row.iloc[0].values)
        else:
            # Fallback: use zeros if no match found
            aligned_targets.append(np.zeros(len(baseline_peak_cols_denoised)))

    return np.array(aligned_targets)


def create_raw_baseline_targets(indices, original_df, class_col_name, baseline_full_df, temp_col_name='train_Temperature', tolerance=0.03):
    """
    Create baseline targets aligned with given indices using full normalized baseline chip data.

    This creates targets from the full normalized baseline chip data (all ~400 rows),
    matching each sample by class AND temperature. NOT passed through any autoencoder.
    Used for transfer autoencoder training.

    Args:
        indices: Array of indices to create targets for
        original_df: Original dataframe with class labels (chip data)
        class_col_name: Name of the class column
        baseline_full_df: Full normalized baseline dataframe (all ~400 rows)
        temp_col_name: Name of temperature column in original_df
        tolerance: Temperature matching tolerance (default 0.03)

    Returns:
        numpy array of aligned baseline targets
    """
    # Get peak columns (exclude Class and other metadata columns)
    baseline_peak_cols = [col for col in baseline_full_df.columns if col.startswith('Peak')]

    aligned_targets = []

    for idx in indices:
        class_label = original_df.iloc[idx][class_col_name]
        temp_value = original_df.iloc[idx][temp_col_name]

        # Match baseline rows by class AND temperature (within tolerance)
        matching_rows = baseline_full_df[
            (baseline_full_df['Class'] == class_label) &
            (abs(baseline_full_df['Temperature'] - temp_value) <= tolerance)
        ][baseline_peak_cols]

        if len(matching_rows) > 0:
            # Use the first matching row (or you could average them)
            aligned_targets.append(matching_rows.iloc[0].values)
        else:
            # Fallback: match by class only if no temperature match
            class_only_match = baseline_full_df[baseline_full_df['Class'] == class_label][baseline_peak_cols]
            if len(class_only_match) > 0:
                aligned_targets.append(class_only_match.iloc[0].values)
            else:
                # Last resort: use zeros
                aligned_targets.append(np.zeros(len(baseline_peak_cols)))

    return np.array(aligned_targets)


def save_autoencoder_output(denoised_data_splits, indices_splits, chip_id,
                           original_df, class_col, output_path, norm_name):
    """
    Save first autoencoder outputs (denoised data) to CSV.

    Args:
        denoised_data_splits: Tuple of (train, val, test) denoised tensors
        indices_splits: Tuple of (train, val, test) indices
        chip_id: Chip identifier
        original_df: Original dataframe with class labels
        class_col: Name of class column (or None)
        output_path: Base output directory
        norm_name: Normalization method name

    Returns:
        Path to saved CSV file
    """
    denoised_train, denoised_val, denoised_test = denoised_data_splits
    indices_train, indices_val, indices_test = indices_splits

    # Combine all splits
    all_denoised_data = torch.cat([denoised_train, denoised_val, denoised_test], dim=0)
    all_indices = np.concatenate([indices_train, indices_val, indices_test])

    # Create dataframe
    denoised_df = pd.DataFrame(all_denoised_data.squeeze().cpu().numpy())
    denoised_df['Chip'] = chip_id

    if class_col is not None:
        # Use indices to map back to original class labels correctly
        denoised_df['Class'] = original_df.iloc[all_indices][class_col].values

    # Save to CSV
    output_file = f"{output_path}/autoencoder_output_chip_{chip_id}_{norm_name}.csv"
    os.makedirs(output_path, exist_ok=True)
    denoised_df.to_csv(output_file, index=False)

    return output_file


def save_transfer_output(transferred_data_splits, indices_splits, chip_id,
                        original_df, class_col, output_path, norm_name):
    """
    Save second autoencoder outputs (transferred/standardized data) to CSV.

    Args:
        transferred_data_splits: Tuple of (train, val, test) transferred tensors
        indices_splits: Tuple of (train, val, test) indices
        chip_id: Chip identifier
        original_df: Original dataframe with class labels
        class_col: Name of class column (or None)
        output_path: Base output directory
        norm_name: Normalization method name

    Returns:
        Path to saved CSV file
    """
    transferred_train, transferred_val, transferred_test = transferred_data_splits
    indices_train, indices_val, indices_test = indices_splits

    # Combine all splits
    all_transferred_data = torch.cat([transferred_train, transferred_val, transferred_test], dim=0)
    all_indices = np.concatenate([indices_train, indices_val, indices_test])

    # Create dataframe
    transfer_df = pd.DataFrame(all_transferred_data.squeeze().cpu().numpy())
    transfer_df['Chip'] = chip_id

    if class_col is not None:
        # Use indices to map back to original class labels correctly
        transfer_df['Class'] = original_df.iloc[all_indices][class_col].values

    # Save to CSV
    output_file = f"{output_path}/transfer_autoencoder_output_chip_{chip_id}_to_baseline_{norm_name}.csv"
    os.makedirs(output_path, exist_ok=True)
    transfer_df.to_csv(output_file, index=False)

    return output_file


def merge_chip_outputs(num_chips, output_base_dir, norm_name, output_type='transfer'):
    """
    Merge autoencoder outputs from all chips into a single CSV file.

    Args:
        num_chips: List of chip IDs
        output_base_dir: Base output directory
        norm_name: Normalization method name
        output_type: Type of output to merge ('transfer' or 'denoised')

    Returns:
        Path to merged CSV file, or None if no files found
    """
    # Collect output files
    output_files = []
    for chip_id in num_chips:
        if output_type == 'transfer':
            file_path = f"{output_base_dir}/transfer_autoencoder_output_chip_{chip_id}_to_baseline_{norm_name}.csv"
        else:  # denoised
            file_path = f"{output_base_dir}/autoencoder_output_chip_{chip_id}_{norm_name}.csv"

        if os.path.exists(file_path):
            output_files.append(file_path)

    if not output_files:
        return None

    # Read and merge all files
    all_outputs = []
    for file_path in output_files:
        df = pd.read_csv(file_path)
        all_outputs.append(df)

    # Combine and save
    merged_outputs = pd.concat(all_outputs, ignore_index=True)

    if output_type == 'transfer':
        merged_path = f"{output_base_dir}/merged_transfer_autoencoder_outputs_{norm_name}_to_baseline.csv"
    else:  # denoised
        merged_path = f"{output_base_dir}/merged_autoencoder_outputs_{norm_name}.csv"

    merged_outputs.to_csv(merged_path, index=False)

    return merged_path


def save_baseline_denoised_output(denoised_data_splits, indices_splits,
                                  baseline_temp_file, label_encoder,
                                  norm_folder):
    """
    Save baseline autoencoder denoised outputs.

    Args:
        denoised_data_splits: Tuple of (train, val, test) denoised tensors
        indices_splits: Tuple of (train, val, test) indices
        baseline_temp_file: Path to baseline temp file
        label_encoder: Label encoder for baseline
        norm_folder: Normalization folder name

    Returns:
        Tuple of (denoised_df, output_path)
    """
    baseline_denoised_train, baseline_denoised_val, baseline_denoised_test = denoised_data_splits
    baseline_indices_train, baseline_indices_val, baseline_indices_test = indices_splits

    # Combine all denoised baseline outputs
    all_baseline_denoised = torch.cat([baseline_denoised_train, baseline_denoised_val, baseline_denoised_test], dim=0)
    all_baseline_indices = np.concatenate([baseline_indices_train, baseline_indices_val, baseline_indices_test])

    # Load the merged baseline dataframe to get class labels
    merged_baseline_df_loaded = pd.read_csv(baseline_temp_file)
    baseline_denoised_df = pd.DataFrame(all_baseline_denoised.squeeze().cpu().numpy())
    baseline_denoised_df['Class'] = merged_baseline_df_loaded.iloc[all_baseline_indices]['train_Class'].values

    # Average by class to get 4 rows (one per class)
    baseline_denoised_df = baseline_denoised_df.groupby('Class').mean().reset_index()

    # Save denoised baseline data
    baseline_denoised_path = f"data/out/{norm_folder}/baseline/baseline_chip_{norm_folder}_denoised_by_baseline_autoencoder.csv"
    os.makedirs(os.path.dirname(baseline_denoised_path), exist_ok=True)
    baseline_denoised_df.to_csv(baseline_denoised_path, index=False)

    return baseline_denoised_df, baseline_denoised_path


def collect_data_for_visualization(transferred_data, denoised_data, labels,
                                   all_transferred, all_denoised, all_labels):
    """
    Collect data across splits for combined visualization.

    Args:
        transferred_data: Tuple of (train, val, test) transferred data
        denoised_data: Tuple of (train, val, test) denoised data
        labels: Tuple of (train, val, test) labels
        all_transferred: List to append transferred data to
        all_denoised: List to append denoised data to
        all_labels: List to append labels to
    """
    transferred_train, transferred_val, transferred_test = transferred_data
    denoised_train, denoised_val, denoised_test = denoised_data
    train_labels, val_labels, test_labels = labels

    all_transferred['train'].append(transferred_train)
    all_transferred['val'].append(transferred_val)
    all_transferred['test'].append(transferred_test)

    all_denoised['train'].append(denoised_train)
    all_denoised['val'].append(denoised_val)
    all_denoised['test'].append(denoised_test)

    all_labels['train'].append(train_labels)
    all_labels['val'].append(val_labels)
    all_labels['test'].append(test_labels)
