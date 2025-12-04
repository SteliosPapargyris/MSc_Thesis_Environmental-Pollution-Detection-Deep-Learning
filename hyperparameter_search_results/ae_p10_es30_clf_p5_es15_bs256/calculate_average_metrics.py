"""
Calculate average metrics across all seeds for different stages of the pipeline.

This script computes mean and standard deviation of accuracy metrics across
multiple random seeds for:
1. Temperature autoencoder classifier (after 1st autoencoder)
2. Baseline autoencoder classifier (after 2nd autoencoder)
3. Global classifier (joint training)
"""

import json
import numpy as np
from pathlib import Path


def calculate_averages():
    # Define seeds
    seeds = [0, 42, 123, 456, 789, 1234, 5678, 9012, 11111, 12345]

    base_path = Path(__file__).parent

    # Storage for all metrics
    temp_ae_metrics = {"train_acc": [], "val_acc": [], "test_acc": []}
    baseline_ae_metrics = {"train_acc": [], "val_acc": [], "test_acc": []}
    global_clf_metrics = {"train_acc": [], "val_acc": [], "test_acc": []}

    for seed in seeds:
        seed_path = base_path / f"seed_{seed}"

        # 1. Temperature autoencoder classifier metrics
        temp_ae_file = seed_path / "chip_to_temperature_autoencoder/metrics/classifier_after_1st_autoencoder_metrics_robust_normalized.json"
        if temp_ae_file.exists():
            with open(temp_ae_file, 'r') as f:
                data = json.load(f)
                temp_ae_metrics["train_acc"].append(data["averages"]["train"]["accuracy"])
                temp_ae_metrics["val_acc"].append(data["averages"]["val"]["accuracy"])
                temp_ae_metrics["test_acc"].append(data["averages"]["test"]["accuracy"])

        # 2. Baseline autoencoder classifier metrics
        baseline_ae_file = seed_path / "chip_to_baseline_autoencoder/metrics/classifier_after_2nd_autoencoder_metrics_robust_normalized.json"
        if baseline_ae_file.exists():
            with open(baseline_ae_file, 'r') as f:
                data = json.load(f)
                baseline_ae_metrics["train_acc"].append(data["averages"]["train"]["accuracy"])
                baseline_ae_metrics["val_acc"].append(data["averages"]["val"]["accuracy"])
                baseline_ae_metrics["test_acc"].append(data["averages"]["test"]["accuracy"])

        # 3. Global classifier metrics (separate files for train/val/test)
        train_file = seed_path / "joint_training_results/train_metrics_robust_normalized.json"
        val_file = seed_path / "joint_training_results/val_metrics_robust_normalized.json"
        test_file = seed_path / "joint_training_results/test_metrics_robust_normalized.json"

        if train_file.exists() and val_file.exists() and test_file.exists():
            with open(train_file, 'r') as f:
                data = json.load(f)
                global_clf_metrics["train_acc"].append(data["accuracy"])

            with open(val_file, 'r') as f:
                data = json.load(f)
                global_clf_metrics["val_acc"].append(data["accuracy"])

            with open(test_file, 'r') as f:
                data = json.load(f)
                global_clf_metrics["test_acc"].append(data["accuracy"])

    # Calculate averages and std
    results = {
        "temperature_autoencoder_classifier": {
            "train_accuracy": {
                "mean": float(np.mean(temp_ae_metrics["train_acc"])) * 100,
                "std": float(np.std(temp_ae_metrics["train_acc"])) * 100
            },
            "val_accuracy": {
                "mean": float(np.mean(temp_ae_metrics["val_acc"])) * 100,
                "std": float(np.std(temp_ae_metrics["val_acc"])) * 100
            },
            "test_accuracy": {
                "mean": float(np.mean(temp_ae_metrics["test_acc"])) * 100,
                "std": float(np.std(temp_ae_metrics["test_acc"])) * 100
            }
        },
        "baseline_autoencoder_classifier": {
            "train_accuracy": {
                "mean": float(np.mean(baseline_ae_metrics["train_acc"])) * 100,
                "std": float(np.std(baseline_ae_metrics["train_acc"])) * 100
            },
            "val_accuracy": {
                "mean": float(np.mean(baseline_ae_metrics["val_acc"])) * 100,
                "std": float(np.std(baseline_ae_metrics["val_acc"])) * 100
            },
            "test_accuracy": {
                "mean": float(np.mean(baseline_ae_metrics["test_acc"])) * 100,
                "std": float(np.std(baseline_ae_metrics["test_acc"])) * 100
            }
        },
        "global_classifier": {
            "train_accuracy": {
                "mean": float(np.mean(global_clf_metrics["train_acc"])) * 100,
                "std": float(np.std(global_clf_metrics["train_acc"])) * 100
            },
            "val_accuracy": {
                "mean": float(np.mean(global_clf_metrics["val_acc"])) * 100,
                "std": float(np.std(global_clf_metrics["val_acc"])) * 100
            },
            "test_accuracy": {
                "mean": float(np.mean(global_clf_metrics["test_acc"])) * 100,
                "std": float(np.std(global_clf_metrics["test_acc"])) * 100
            }
        },
        "num_seeds": len(seeds)
    }

    return results


def main():
    results = calculate_averages()

    # Print results
    print(json.dumps(results, indent=2))

    # Save to file
    output_file = Path(__file__).parent / "averaged_metrics_across_seeds.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
