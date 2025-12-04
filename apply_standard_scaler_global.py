"""
Apply StandardScaler globally to all chips using training data from all datasets for fitting.

This script:
1. Loads all chip files (chip_1 to chip_N) + baseline chip
2. Identifies training samples from baseline + all chips (using train/val/test split)
3. Fits StandardScaler on ALL training samples from ALL datasets
4. Transforms ALL samples from ALL chips using the fitted scaler
5. Saves normalized files to: data/out/robust/Nchips/with_standard_scaler/chip_X_robust.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.config import total_num_chips, seed, extended

def apply_global_standard_scaler():
    """
    Apply StandardScaler globally across all datasets.

    The scaler is fitted on ALL training samples from ALL chips + baseline,
    then applied to transform all samples from all chips.
    """

    print("\n" + "="*80)
    print("GLOBAL STANDARD SCALER APPLICATION")
    print("="*80)
    print(f"Processing {total_num_chips} chips + baseline chip")
    print("="*80 + "\n")

    # ============================================================================
    # STEP 1: Load all chip data
    # ============================================================================

    print("STEP 1: Loading all chip data...")

    # Get current normalization method from config
    from utils.config import norm_folder, norm_name

    chip_files = []
    chip_dataframes = []

    # Determine directory suffix based on extended flag
    dir_suffix = "_extended" if extended else ""
    data_dir = Path(f"data/out/{norm_folder}/{total_num_chips}chips{dir_suffix}")

    print(f"Loading from: {data_dir}")
    print(f"Normalization method: {norm_folder}")
    print(f"Extended dataset: {extended}\n")

    # Load numbered chips (1 to total_num_chips)
    for chip_num in range(1, total_num_chips + 1):
        chip_file = data_dir / f"chip_{chip_num}_{norm_folder}.csv"
        df = pd.read_csv(chip_file)
        chip_dataframes.append(df)
        chip_files.append(chip_file)
        print(f"  ✓ Loaded chip {chip_num}: {len(df)} samples")


    # Load baseline chip
    baseline_filename = f"baseline_chip{dir_suffix}_{norm_folder}.csv"
    baseline_file = Path(f"data/out/{norm_folder}/baseline/{baseline_filename}")
    baseline_df = pd.read_csv(baseline_file)
    print(f"  ✓ Loaded baseline chip: {len(baseline_df)} samples")
    print(f"\nTotal chips loaded: {len(chip_dataframes)}")
    print(f"Baseline samples: {len(baseline_df)}")

    # ============================================================================
    # STEP 2: Collect ALL training samples from ALL chips + baseline
    # ============================================================================

    print("\n" + "-"*80)
    print("STEP 2: Collecting training samples from all datasets...")
    print("-"*80 + "\n")

    # Extract Peak columns - check which format is used in the data
    # Normalized data uses "train_Peak 1" format
    sample_cols = chip_dataframes[0].columns.tolist()
    if any('train_Peak' in col for col in sample_cols):
        peak_columns = [f"train_Peak {i}" for i in range(1, 33)]
        print(f"Using train_Peak format for columns")
    else:
        peak_columns = [f"Peak {i}" for i in range(1, 33)]
        print(f"Using Peak format for columns")

    # Collect all training samples
    all_train_samples = []

    # From baseline: Apply same train/test split as chips
    class_col = 'train_Class' if 'train_Class' in baseline_df.columns else 'Class'
    baseline_indices = np.arange(len(baseline_df))
    baseline_y = baseline_df[class_col].values

    # First split: 70% train, 30% temp
    baseline_train_indices, baseline_temp_indices = train_test_split(
        baseline_indices, test_size=0.3, random_state=seed, stratify=baseline_y
    )

    baseline_peaks = baseline_df.iloc[baseline_train_indices][peak_columns].values
    all_train_samples.append(baseline_peaks)
    baseline_train_count = len(baseline_peaks)
    print(f"  ✓ Baseline training samples: {baseline_train_count}")

    # From each chip: Get training samples using same split logic as pipeline
    chip_train_count = 0
    for chip_idx, df in enumerate(chip_dataframes, start=1):
        # Use same split logic as in load_and_preprocess_data_autoencoder_prenormalized
        # Split: 70% train, 15% val, 15% test

        # Check if 'Class' or 'train_Class' column exists
        class_col = 'train_Class' if 'train_Class' in df.columns else 'Class'

        indices = np.arange(len(df))
        y = df[class_col].values

        # First split: 70% train, 30% temp
        train_indices, temp_indices = train_test_split(
            indices, test_size=0.3, random_state=seed, stratify=y
        )

        # Get training samples
        train_samples = df.iloc[train_indices][peak_columns].values
        all_train_samples.append(train_samples)
        chip_train_count += len(train_samples)
        print(f"  ✓ Chip {chip_idx} training samples: {len(train_samples)}")

    # Concatenate all training samples
    X_train_fit = np.vstack(all_train_samples)

    print(f"\nTotal training samples collected: {X_train_fit.shape[0]}")
    print(f"  - From baseline: {baseline_train_count}")
    print(f"  - From chips: {chip_train_count}")

    # ============================================================================
    # STEP 3: Fit StandardScaler on ALL training samples
    # ============================================================================

    print("\n" + "-"*80)
    print("STEP 3: Fitting StandardScaler on ALL training data...")
    print("-"*80 + "\n")

    print(f"Training data shape for fitting: {X_train_fit.shape}")
    print(f"  - Samples: {X_train_fit.shape[0]}")
    print(f"  - Features: {X_train_fit.shape[1]}")

    # Fit StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train_fit)

    print("\n✓ StandardScaler fitted successfully")
    print(f"  - Mean shape: {scaler.mean_.shape}")
    print(f"  - Std shape: {scaler.scale_.shape}")
    print(f"  - Mean (first 5 features): {scaler.mean_[:5]}")
    print(f"  - Std (first 5 features): {scaler.scale_[:5]}")

    # ============================================================================
    # STEP 4: Transform all chip samples
    # ============================================================================

    print("\n" + "-"*80)
    print("STEP 4: Transforming all chip samples...")
    print("-"*80 + "\n")

    normalized_chips = []

    for chip_idx, df in enumerate(chip_dataframes, start=1):
        print(f"Transforming chip {chip_idx}...")

        # Create a copy to avoid modifying original
        normalized_df = df.copy()

        # Extract peak values
        X_chip = df[peak_columns].values

        # Transform using fitted scaler
        X_chip_normalized = scaler.transform(X_chip)

        # Replace peak columns with normalized values
        normalized_df[peak_columns] = X_chip_normalized

        normalized_chips.append(normalized_df)

        print(f"  ✓ Chip {chip_idx} transformed: {X_chip_normalized.shape}")
        print(f"    - Original range: [{X_chip.min():.4f}, {X_chip.max():.4f}]")
        print(f"    - Normalized range: [{X_chip_normalized.min():.4f}, {X_chip_normalized.max():.4f}]")

    # Transform baseline chip as well
    print(f"\nTransforming baseline chip...")
    normalized_baseline = baseline_df.copy()
    X_baseline = baseline_df[peak_columns].values
    X_baseline_normalized = scaler.transform(X_baseline)
    normalized_baseline[peak_columns] = X_baseline_normalized

    print(f"  ✓ Baseline chip transformed: {X_baseline_normalized.shape}")
    print(f"    - Original range: [{X_baseline.min():.4f}, {X_baseline.max():.4f}]")
    print(f"    - Normalized range: [{X_baseline_normalized.min():.4f}, {X_baseline_normalized.max():.4f}]")

    # ============================================================================
    # STEP 5: Save normalized chips
    # ============================================================================

    print("\n" + "-"*80)
    print("STEP 5: Saving normalized chips...")
    print("-"*80 + "\n")

    # Create output directory with appropriate naming
    output_dir = Path(f"data/out/{norm_folder}/{total_num_chips}chips{dir_suffix}/with_standard_scaler")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each normalized chip
    for chip_idx, normalized_df in enumerate(normalized_chips, start=1):
        output_file = output_dir / f"chip_{chip_idx}_{norm_folder}.csv"
        normalized_df.to_csv(output_file, index=False)
        print(f"  ✓ Saved: {output_file}")

    # Save normalized baseline
    baseline_output_dir = Path(f"data/out/{norm_folder}/baseline/with_standard_scaler")
    baseline_output_dir.mkdir(parents=True, exist_ok=True)
    baseline_output_filename = f"baseline_chip{dir_suffix}_{norm_folder}.csv"
    baseline_output_file = baseline_output_dir / baseline_output_filename
    normalized_baseline.to_csv(baseline_output_file, index=False)
    print(f"  ✓ Saved: {baseline_output_file}")

    # ============================================================================
    # SUMMARY
    # ============================================================================

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ StandardScaler fitted on: {X_train_fit.shape[0]} training samples")
    print(f"  - From baseline: {baseline_train_count}")
    print(f"  - From all chips: {chip_train_count}")
    print(f"✓ Transformed {len(normalized_chips)} chips")
    print(f"✓ Transformed baseline chip")
    print(f"✓ Output directory: {output_dir}")
    print("="*80 + "\n")

    return normalized_chips, normalized_baseline, scaler


if __name__ == "__main__":
    normalized_chips, normalized_baseline, scaler = apply_global_standard_scaler()
    print("✓ Global StandardScaler application complete!")
