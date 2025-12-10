"""
Script to create averaged training/validation loss plots from saved JSON loss history files.

This script reads loss history JSON files for all 10 chips and creates
averaged plots showing mean loss across all chips with confidence intervals.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


def load_loss_history(json_path):
    """Load loss history from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return np.array(data['train_loss']), np.array(data['val_loss'])


def interpolate_to_common_length(losses_list, target_length):
    """
    Interpolate all loss curves to the same length.

    Args:
        losses_list: List of loss arrays (different lengths)
        target_length: Target length for all arrays

    Returns:
        numpy array: Shape (num_curves, target_length)
    """
    interpolated = []

    for losses in losses_list:
        if len(losses) == 0:
            continue

        # Create interpolation function
        x_old = np.linspace(0, 1, len(losses))
        x_new = np.linspace(0, 1, target_length)

        f = interpolate.interp1d(x_old, losses, kind='linear', fill_value='extrapolate')
        interpolated.append(f(x_new))

    return np.array(interpolated)


def create_averaged_plot_from_json(json_paths, output_path, title):
    """
    Create an averaged loss plot from multiple JSON loss history files.

    Args:
        json_paths: List of paths to JSON loss history files
        output_path: Path where the averaged plot will be saved
        title: Title for the averaged plot
    """
    print(f"\nCreating averaged plot: {title}")
    print(f"Processing {len(json_paths)} JSON files...")

    all_train_losses = []
    all_val_losses = []
    max_length = 0

    # Load all loss histories
    for i, json_path in enumerate(json_paths, 1):
        if not os.path.exists(json_path):
            print(f"  [{i}/{len(json_paths)}] WARNING: File not found: {json_path}")
            continue

        try:
            print(f"  [{i}/{len(json_paths)}] Loading: {os.path.basename(json_path)}")
            train_loss, val_loss = load_loss_history(json_path)

            all_train_losses.append(train_loss)
            all_val_losses.append(val_loss)
            max_length = max(max_length, len(train_loss), len(val_loss))

        except Exception as e:
            print(f"    ERROR: Could not load {json_path}: {e}")
            continue

    if len(all_train_losses) == 0:
        print(f"  ERROR: No valid loss histories loaded!")
        return False

    print(f"  Loaded {len(all_train_losses)} loss histories")
    print(f"  Target length for interpolation: {max_length}")

    # Interpolate all curves to the same length
    train_interpolated = interpolate_to_common_length(all_train_losses, max_length)
    val_interpolated = interpolate_to_common_length(all_val_losses, max_length)

    # Compute mean and std
    train_mean = np.mean(train_interpolated, axis=0)
    train_std = np.std(train_interpolated, axis=0)

    val_mean = np.mean(val_interpolated, axis=0)
    val_std = np.std(val_interpolated, axis=0)

    # Create epochs array
    epochs = np.arange(max_length)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot mean lines
    plt.plot(epochs, train_mean, 'b-', linewidth=2, label=f'Training Loss (mean, n={len(all_train_losses)})')
    plt.plot(epochs, val_mean, 'orange', linewidth=2, label=f'Validation Loss (mean, n={len(all_val_losses)})')

    # Add confidence intervals (mean ± std)
    plt.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                     color='blue', alpha=0.2, label='Training Loss ± 1 std')
    plt.fill_between(epochs, val_mean - val_std, val_mean + val_std,
                     color='orange', alpha=0.2, label='Validation Loss ± 1 std')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)

    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved averaged plot to: {output_path}")
    return True


def main():
    """Main function to create all averaged plots."""

    base_dir = "hyperparameter_search_results/ae_p10_es30_clf_p5_es15_bs256"

    print("="*80)
    print("Creating Averaged Loss Plots from JSON Loss Histories")
    print("="*80)

    # ===== 1. chip_to_temperature_autoencoder - Autoencoder plots =====
    temp_ae_dir = f"{base_dir}/chip_to_temperature_autoencoder/train_val_plots"
    temp_ae_json_files = [
        f"{temp_ae_dir}/autoencoder_robust_normalized_chip_{i}_loss_history.json"
        for i in range(1, 11)
    ]
    temp_ae_output = f"{temp_ae_dir}/autoencoder_robust_normalized_AVERAGED.png"

    create_averaged_plot_from_json(
        temp_ae_json_files,
        temp_ae_output,
        "Averaged Temperature Autoencoder Training Loss (Chips 1-10)"
    )

    # ===== 2. chip_to_temperature_autoencoder - Classifier plots =====
    temp_clf_json_files = [
        f"{temp_ae_dir}/classifier_after_1st_autoencoder_robust_normalized_chip_{i}_loss_history.json"
        for i in range(1, 11)
    ]
    temp_clf_output = f"{temp_ae_dir}/classifier_after_1st_autoencoder_robust_normalized_AVERAGED.png"

    create_averaged_plot_from_json(
        temp_clf_json_files,
        temp_clf_output,
        "Averaged Classifier Training Loss after 1st Autoencoder (Chips 1-10)"
    )

    # ===== 3. chip_to_baseline_autoencoder - Transfer Autoencoder plots =====
    baseline_ae_dir = f"{base_dir}/chip_to_baseline_autoencoder/train_val_plots"
    baseline_ae_json_files = [
        f"{baseline_ae_dir}/transfer_autoencoder_robust_normalized_chip_{i}_to_baseline_loss_history.json"
        for i in range(1, 11)
    ]
    baseline_ae_output = f"{baseline_ae_dir}/transfer_autoencoder_robust_normalized_AVERAGED.png"

    create_averaged_plot_from_json(
        baseline_ae_json_files,
        baseline_ae_output,
        "Averaged Transfer Autoencoder Training Loss (Chips 1-10 to Baseline)"
    )

    # ===== 4. chip_to_baseline_autoencoder - Classifier plots =====
    baseline_clf_json_files = [
        f"{baseline_ae_dir}/classifier_after_2nd_autoencoder_robust_normalized_chip_{i}_loss_history.json"
        for i in range(1, 11)
    ]
    baseline_clf_output = f"{baseline_ae_dir}/classifier_after_2nd_autoencoder_robust_normalized_AVERAGED.png"

    create_averaged_plot_from_json(
        baseline_clf_json_files,
        baseline_clf_output,
        "Averaged Classifier Training Loss after 2nd Autoencoder (Chips 1-10)"
    )

    print("\n" + "="*80)
    print("✓ All averaged plots created successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
