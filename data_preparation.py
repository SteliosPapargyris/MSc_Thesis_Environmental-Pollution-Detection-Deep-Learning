"""
Data Preparation Script

This script handles all data preparation tasks for the autoencoder pipeline:
1. Loading and creating matched datasets from normalized chip data
2. Creating baseline self-match datasets
3. Saving prepared datasets as CSV files for subsequent training steps

Run this script first before running the autoencoder training scripts.

Usage:
    python data_preparation.py
"""

import pandas as pd
import numpy as np
import os
from typing import List
from utils.config import total_num_chips, num_chips, norm_folder, norm_name, extended
from utils.plot_utils import plot_normalized_data_distribution


def create_chip_matched_datasets(
    chip_indices: List[int],
    norm_method: str
) -> List[pd.DataFrame]:
    """
    Create matched datasets for each chip by pairing rows with their 27°C class representatives.

    For each chip, this function:
    1. Loads the normalized chip data
    2. Filters for 27°C temperature rows and averages by class
    3. Matches every row with its corresponding 27°C class representative
    4. Creates train/match paired dataset

    Args:
        chip_indices: List of chip IDs to process (e.g., [1, 2, 3, ...])
        norm_method: Normalization method folder name (e.g., 'mean_std', 'minmax', 'robust')

    Returns:
        List of DataFrames, one matched dataset per chip with train_ and match_ prefixed columns

    Example:
        >>> datasets = create_chip_matched_datasets([1, 2, 3], 'mean_std')
        >>> print(f"Created {len(datasets)} chip datasets")
    """
    chip_paired_datasets = []

    # Determine directory suffix based on extended flag
    dir_suffix = "_extended"

    for chip_id in chip_indices:
        # Load normalized chip data (preprocessed via apply_standard_scaler_global.py)
        chip_data_path = f"data/out/{norm_method}/{total_num_chips}chips{dir_suffix}/with_standard_scaler/chip_{chip_id}_{norm_method}.csv"
        chip_all_temps = pd.read_csv(chip_data_path)

        # Filter for 27°C temperature rows (within tolerance of ±0.03°C)
        chip_at_27c = chip_all_temps[abs(chip_all_temps['Temperature'] - 27.0) <= 0.03]

        # Average within each class to get one representative row per class at 27°C
        chip_27c_class_representatives = chip_at_27c.groupby('Class').mean().reset_index()

        # Create matched pairs: each row paired with its 27°C class representative
        matched_rows = []
        for _, train_row in chip_all_temps.iterrows():
            # Find matching 27°C representative for this row's class
            matching_rows = chip_27c_class_representatives[
                chip_27c_class_representatives['Class'] == train_row['Class']
            ]

            # Create paired row with train_ and match_ prefixes
            for _, match_row in matching_rows.iterrows():
                train_series = train_row.add_prefix("train_")
                match_series = match_row.add_prefix("match_")
                merged_row = pd.concat([train_series, match_series])
                matched_rows.append(merged_row)

        # Create and shuffle the matched dataset
        chip_paired_dataset = pd.DataFrame(matched_rows)
        chip_paired_dataset = chip_paired_dataset.sample(frac=1).reset_index(drop=True)

        # Save to file for traceability
        output_dir = f"data/out/shuffled_dataset"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{chip_id}_self_match_27C.csv")
        chip_paired_dataset.to_csv(output_path, index=False)

        chip_paired_datasets.append(chip_paired_dataset)
        print(f"✓ Created matched dataset for chip {chip_id}: {len(chip_paired_dataset)} samples → {output_path}")

    return chip_paired_datasets


def create_baseline_matched_dataset(
    norm_method: str
) -> pd.DataFrame:
    """
    Create baseline matched dataset by pairing rows with their 27°C class representatives.

    Similar to chip matching, but for the baseline chip. The baseline is used to:
    1. Train the baseline autoencoder (denoising task)
    2. Provide target reference for the transfer autoencoder

    Args:
        norm_method: Normalization method folder name (e.g., 'mean_std', 'minmax', 'robust')

    Returns:
        Matched baseline dataset with train_ and match_ prefixed columns

    Example:
        >>> baseline_paired_dataset = create_baseline_matched_dataset('mean_std')
        >>> print(f"Baseline dataset: {len(baseline_paired_dataset)} samples")
    """
    # Determine filename suffix based on extended flag
    file_suffix = "_extended" if extended else ""

    # Load baseline normalized data (preprocessed via apply_standard_scaler_global.py)
    baseline_data_path = f"data/out/{norm_method}/baseline/with_standard_scaler/baseline_chip{file_suffix}_{norm_method}.csv"
    baseline_all_temps = pd.read_csv(baseline_data_path)

    # Filter for 27°C temperature rows (within tolerance of ±0.03°C)
    baseline_at_27c = baseline_all_temps[abs(baseline_all_temps['Temperature'] - 27.0) <= 0.03]

    # Average within each class to get one representative row per class at 27°C
    baseline_27c_class_representatives = baseline_at_27c.groupby('Class').mean().reset_index()

    # Create matched pairs: each row paired with its 27°C class representative
    matched_rows = []
    for _, train_row in baseline_all_temps.iterrows():
        # Find matching 27°C representative for this row's class
        matching_rows = baseline_27c_class_representatives[
            baseline_27c_class_representatives['Class'] == train_row['Class']
        ]

        # Create paired row with train_ and match_ prefixes
        for _, match_row in matching_rows.iterrows():
            train_series = train_row.add_prefix("train_")
            match_series = match_row.add_prefix("match_")
            merged_row = pd.concat([train_series, match_series])
            matched_rows.append(merged_row)

    # Create and shuffle the matched dataset
    baseline_paired_dataset = pd.DataFrame(matched_rows)
    baseline_paired_dataset = baseline_paired_dataset.sample(frac=1).reset_index(drop=True)

    # Save to file for traceability
    output_dir = f"data/out/shuffled_dataset"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "baseline_self_match_27C.csv")
    baseline_paired_dataset.to_csv(output_path, index=False)

    print(f"✓ Created baseline matched dataset: {len(baseline_paired_dataset)} samples → {output_path}")

    return baseline_paired_dataset


def prepare_data_splits_summary(
    chip_datasets: List[pd.DataFrame],
    num_chips: int
) -> None:
    """
    Print summary statistics for prepared datasets.

    Args:
        chip_datasets: List of prepared chip datasets
        num_chips: Total number of chips
    """
    print(f"\n{'='*80}")
    print(f"DATA PREPARATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total chips processed: {num_chips}")
    print(f"Datasets created: {len(chip_datasets)}")

    if chip_datasets:
        sample_counts = [len(df) for df in chip_datasets]
        print(f"Samples per chip - Min: {min(sample_counts)}, Max: {max(sample_counts)}, Avg: {np.mean(sample_counts):.1f}")

    print(f"{'='*80}\n")


def main():
    """Main data preparation pipeline."""

    print("\n" + "="*80)
    print("DATA PREPARATION PIPELINE")
    print("="*80)
    print(f"Configuration: {total_num_chips} chips, {norm_name}")
    print(f"Normalization method: {norm_folder}")
    print(f"Extended dataset: {extended}")
    print("="*80 + "\n")

    # ====================================================================================
    # STEP 1: CREATE CHIP MATCHED DATASETS
    # ====================================================================================

    print(f"\n{'='*80}")
    print(f"STEP 1: CREATING CHIP MATCHED DATASETS")
    print(f"{'='*80}\n")

    chip_matched_datasets = create_chip_matched_datasets(num_chips, norm_method=norm_folder)

    # ====================================================================================
    # STEP 2: CREATE BASELINE MATCHED DATASET
    # ====================================================================================

    print(f"\n{'='*80}")
    print(f"STEP 2: CREATING BASELINE MATCHED DATASET")
    print(f"{'='*80}\n")

    baseline_paired_dataset = create_baseline_matched_dataset(norm_folder)
    print(f"Baseline dataset shape: {baseline_paired_dataset.shape}")

    # ====================================================================================
    # STEP 3: VISUALIZE PREPARED DATA
    # ====================================================================================

    print(f"\n{'='*80}")
    print(f"STEP 3: CREATING DATA VISUALIZATIONS")
    print(f"{'='*80}\n")

    print("Plotting normalized data distribution...")
    plot_normalized_data_distribution(chip_matched_datasets, num_chips, norm_name)

    # ====================================================================================
    # STEP 4: SUMMARY
    # ====================================================================================

    prepare_data_splits_summary(chip_matched_datasets, total_num_chips)

    print(f"\n{'='*80}")
    print(f"DATA PREPARATION COMPLETE!")
    print(f"{'='*80}")
    print(f"✓ {total_num_chips} chip datasets created")
    print(f"✓ Baseline dataset created")
    print(f"✓ All datasets saved to: data/out/shuffled_dataset/")
    print(f"✓ Visualizations saved")
    print(f"\nNext step: Run train_chip_temperature_autoencoder.py or train_chip_to_baseline_autoencoder.py")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
