"""
Master script to run complete training pipeline for multiple seeds.

This script runs:
1. train_chip_temperature_autoencoder.py
2. train_chip_to_baseline_autoencoder.py
3. inference_with_training.py
4. create_averaged_plots_from_json.py

For each of the specified seeds: [0, 42, 2023]

Results are saved to:
hyperparameter_search_results/ae_p10_es30_clf_p5_es15_bs256/seed_<SEED>/
"""

import os
import sys
import subprocess
import json
from datetime import datetime

# TODO -> Change seed to previous list
# Configuration
SEEDS = [0, 42, 123, 777, 1234, 3407, 5555, 8888, 12345]
BASE_OUTPUT_DIR = "hyperparameter_search_results/ae_p10_es30_clf_p5_es15_bs256"

# Hyperparameters
AUTOENCODER_PATIENCE = 10
AUTOENCODER_EARLY_STOPPING = 30
CLASSIFIER_PATIENCE = 5
CLASSIFIER_EARLY_STOPPING = 15
BATCH_SIZE = 256


def run_command(cmd, env_vars, description):
    """Run a command with environment variables and log output."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")

    # Merge environment variables
    env = os.environ.copy()
    env.update(env_vars)

    # Run the command
    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ {description} failed with error: {e}")
        return False


def run_training_for_seed(seed):
    """Run complete training pipeline for a single seed."""
    print("\n" + "="*80)
    print(f"STARTING TRAINING FOR SEED {seed}")
    print("="*80)

    # Create output directory for this seed
    seed_output_dir = os.path.join(BASE_OUTPUT_DIR, f"seed_{seed}")

    # Check if this seed has already been completed
    joint_results_dir = os.path.join(seed_output_dir, "joint_training_results")
    if os.path.exists(joint_results_dir) and os.path.exists(os.path.join(joint_results_dir, f"joint_training_loss_plot_robust_normalized.png")):
        print(f"✓ SEED {seed} ALREADY COMPLETED - SKIPPING")
        print(f"  Results found in: {seed_output_dir}")
        print("="*80)
        return True

    os.makedirs(seed_output_dir, exist_ok=True)

    # Common environment variables
    env_vars = {
        'SEED': str(seed),
        'AUTOENCODER_PATIENCE': str(AUTOENCODER_PATIENCE),
        'AUTOENCODER_EARLY_STOPPING': str(AUTOENCODER_EARLY_STOPPING),
        'CLASSIFIER_PATIENCE': str(CLASSIFIER_PATIENCE),
        'CLASSIFIER_EARLY_STOPPING': str(CLASSIFIER_EARLY_STOPPING),
        'BATCH_SIZE': str(BATCH_SIZE),
        'OUTPUT_BASE_DIR': seed_output_dir,
        'PYTHONUNBUFFERED': '1'  # For real-time output
    }

    # Store configuration
    config = {
        'seed': seed,
        'autoencoder_patience': AUTOENCODER_PATIENCE,
        'autoencoder_early_stopping': AUTOENCODER_EARLY_STOPPING,
        'classifier_patience': CLASSIFIER_PATIENCE,
        'classifier_early_stopping': CLASSIFIER_EARLY_STOPPING,
        'batch_size': BATCH_SIZE,
        'start_time': datetime.now().isoformat()
    }

    config_path = os.path.join(seed_output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Saved configuration to: {config_path}")

    # Step 1: Train chip temperature autoencoders
    success = run_command(
        ['python', 'train_chip_temperature_autoencoder.py'],
        env_vars,
        f"[SEED {seed}] Step 1/3: Training chip temperature autoencoders"
    )
    if not success:
        print(f"✗ Training failed for seed {seed} at step 1")
        return False

    # Step 2: Train chip-to-baseline autoencoders
    success = run_command(
        ['python', 'train_chip_to_baseline_autoencoder.py'],
        env_vars,
        f"[SEED {seed}] Step 2/3: Training chip-to-baseline autoencoders"
    )
    if not success:
        print(f"✗ Training failed for seed {seed} at step 2")
        return False

    # Step 3: Run joint training with inference
    success = run_command(
        ['python', 'inference_with_training.py'],
        env_vars,
        f"[SEED {seed}] Step 3/3: Running joint training with inference"
    )
    if not success:
        print(f"✗ Training failed for seed {seed} at step 3")
        return False

    # Step 4: Generate averaged plots
    print(f"\n{'='*80}")
    print(f"[SEED {seed}] Generating averaged loss plots from JSON histories")
    print(f"{'='*80}")

    # Update the averaging script to use the correct seed directory
    create_averaged_plots_for_seed(seed, seed_output_dir)

    # Update config with completion time
    config['end_time'] = datetime.now().isoformat()
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✓ TRAINING COMPLETED FOR SEED {seed}")
    print(f"  Results saved to: {seed_output_dir}")
    print(f"{'='*80}")

    return True


def create_averaged_plots_for_seed(seed, seed_output_dir):
    """Create averaged plots for a specific seed."""
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import interpolate

    def load_loss_history(json_path):
        """Load loss history from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return np.array(data['train_loss']), np.array(data['val_loss'])

    def interpolate_to_common_length(losses_list, target_length):
        """Interpolate all loss curves to the same length."""
        interpolated = []
        for losses in losses_list:
            if len(losses) == 0:
                continue
            x_old = np.linspace(0, 1, len(losses))
            x_new = np.linspace(0, 1, target_length)
            f = interpolate.interp1d(x_old, losses, kind='linear', fill_value='extrapolate')
            interpolated.append(f(x_new))
        return np.array(interpolated)

    def create_averaged_plot(json_paths, output_path, title):
        """Create an averaged loss plot from multiple JSON files."""
        print(f"  Creating: {os.path.basename(output_path)}")

        all_train_losses = []
        all_val_losses = []
        max_length = 0

        for json_path in json_paths:
            if not os.path.exists(json_path):
                continue
            try:
                train_loss, val_loss = load_loss_history(json_path)
                all_train_losses.append(train_loss)
                all_val_losses.append(val_loss)
                max_length = max(max_length, len(train_loss), len(val_loss))
            except Exception as e:
                print(f"    Warning: Could not load {json_path}: {e}")
                continue

        if len(all_train_losses) == 0:
            print(f"    ERROR: No valid loss histories loaded!")
            return False

        # Interpolate and compute statistics
        train_interpolated = interpolate_to_common_length(all_train_losses, max_length)
        val_interpolated = interpolate_to_common_length(all_val_losses, max_length)

        train_mean = np.mean(train_interpolated, axis=0)
        train_std = np.std(train_interpolated, axis=0)
        val_mean = np.mean(val_interpolated, axis=0)
        val_std = np.std(val_interpolated, axis=0)

        epochs = np.arange(max_length)

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_mean, 'b-', linewidth=2, label=f'Training Loss (mean, n={len(all_train_losses)})')
        plt.plot(epochs, val_mean, 'orange', linewidth=2, label=f'Validation Loss (mean, n={len(all_val_losses)})')
        plt.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                         color='blue', alpha=0.2, label='Training Loss ± 1 std')
        plt.fill_between(epochs, val_mean - val_std, val_mean + val_std,
                         color='orange', alpha=0.2, label='Validation Loss ± 1 std')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"    ✓ Saved to: {output_path}")
        return True

    # Generate all 4 averaged plots

    # 1. Temperature autoencoder
    temp_ae_dir = f"{seed_output_dir}/chip_to_temperature_autoencoder/train_val_plots"
    temp_ae_json_files = [
        f"{temp_ae_dir}/autoencoder_robust_normalized_chip_{i}_loss_history.json"
        for i in range(1, 11)
    ]
    temp_ae_output = f"{temp_ae_dir}/autoencoder_robust_normalized_AVERAGED.png"
    create_averaged_plot(temp_ae_json_files, temp_ae_output,
                        f"Averaged Temperature Autoencoder Loss (Seed {seed})")

    # 2. Temperature classifier
    temp_clf_json_files = [
        f"{temp_ae_dir}/classifier_after_1st_autoencoder_robust_normalized_chip_{i}_loss_history.json"
        for i in range(1, 11)
    ]
    temp_clf_output = f"{temp_ae_dir}/classifier_after_1st_autoencoder_robust_normalized_AVERAGED.png"
    create_averaged_plot(temp_clf_json_files, temp_clf_output,
                        f"Averaged Classifier Loss after 1st Autoencoder (Seed {seed})")

    # 3. Baseline transfer autoencoder
    baseline_ae_dir = f"{seed_output_dir}/chip_to_baseline_autoencoder/train_val_plots"
    baseline_ae_json_files = [
        f"{baseline_ae_dir}/transfer_autoencoder_robust_normalized_chip_{i}_to_baseline_loss_history.json"
        for i in range(1, 11)
    ]
    baseline_ae_output = f"{baseline_ae_dir}/transfer_autoencoder_robust_normalized_AVERAGED.png"
    create_averaged_plot(baseline_ae_json_files, baseline_ae_output,
                        f"Averaged Transfer Autoencoder Loss (Seed {seed})")

    # 4. Baseline classifier
    baseline_clf_json_files = [
        f"{baseline_ae_dir}/classifier_after_2nd_autoencoder_robust_normalized_chip_{i}_loss_history.json"
        for i in range(1, 11)
    ]
    baseline_clf_output = f"{baseline_ae_dir}/classifier_after_2nd_autoencoder_robust_normalized_AVERAGED.png"
    create_averaged_plot(baseline_clf_json_files, baseline_clf_output,
                        f"Averaged Classifier Loss after 2nd Autoencoder (Seed {seed})")

    print(f"  ✓ All averaged plots created for seed {seed}")


def main():
    """Main function to run training for all seeds."""
    print("\n" + "="*80)
    print("MULTI-SEED TRAINING PIPELINE")
    print("="*80)
    print(f"Configuration: ae_p10_es30_clf_p5_es15_bs256")
    print(f"Seeds: {SEEDS}")
    print(f"Base output directory: {BASE_OUTPUT_DIR}")
    print("="*80)

    results = {}

    for seed in SEEDS:
        success = run_training_for_seed(seed)
        results[seed] = success

        if not success:
            print(f"\n⚠️  Training failed for seed {seed}")
            print(f"   Continuing with remaining seeds...")

    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    for seed, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"Seed {seed:4d}: {status}")
    print("="*80)

    successful_seeds = [seed for seed, success in results.items() if success]
    if len(successful_seeds) == len(SEEDS):
        print(f"\n✓ All training runs completed successfully!")
    else:
        print(f"\n⚠️  {len(successful_seeds)}/{len(SEEDS)} training runs succeeded")

    print(f"\nResults saved to: {BASE_OUTPUT_DIR}/seed_<SEED>/")


if __name__ == "__main__":
    main()
