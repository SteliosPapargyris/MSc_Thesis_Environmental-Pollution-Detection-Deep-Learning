"""
Calculate average metrics for reverse order training across first 3 seeds.

This script computes mean and standard deviation of accuracy metrics across
the first 3 random seeds (0, 42, 123) for the reverse order training approach.
"""

import json
import numpy as np
from pathlib import Path


def calculate_averages():
    # Define seeds - only first 3 seeds for reverse order
    seeds = [0, 42, 123, 777, 1234, 2024, 3407, 5555, 8888, 12345]

    base_path = Path(__file__).parent

    # Storage for reverse order metrics only
    reverse_order_metrics = {"train_acc": [], "val_acc": [], "test_acc": []}

    for seed in seeds:
        seed_path = base_path / f"seed_{seed}"

        # Reverse order metrics (separate files for train/val/test)
        train_file = seed_path / "reverse_order/train_metrics_robust_normalized.json"
        val_file = seed_path / "reverse_order/val_metrics_robust_normalized.json"
        test_file = seed_path / "reverse_order/test_metrics_robust_normalized.json"

        if train_file.exists() and val_file.exists() and test_file.exists():
            with open(train_file, 'r') as f:
                data = json.load(f)
                reverse_order_metrics["train_acc"].append(data["accuracy"])

            with open(val_file, 'r') as f:
                data = json.load(f)
                reverse_order_metrics["val_acc"].append(data["accuracy"])

            with open(test_file, 'r') as f:
                data = json.load(f)
                reverse_order_metrics["test_acc"].append(data["accuracy"])

    # Calculate averages and std
    results = {
        "reverse_order": {
            "train_accuracy": {
                "mean": float(np.mean(reverse_order_metrics["train_acc"])) * 100,
                "std": float(np.std(reverse_order_metrics["train_acc"])) * 100
            },
            "val_accuracy": {
                "mean": float(np.mean(reverse_order_metrics["val_acc"])) * 100,
                "std": float(np.std(reverse_order_metrics["val_acc"])) * 100
            },
            "test_accuracy": {
                "mean": float(np.mean(reverse_order_metrics["test_acc"])) * 100,
                "std": float(np.std(reverse_order_metrics["test_acc"])) * 100
            }
        },
        "num_seeds": len(seeds),
        "seeds_used": seeds
    }

    return results


def main():
    results = calculate_averages()

    # Print results
    print(json.dumps(results, indent=2))

    # Save to file
    output_file = Path(__file__).parent / "averaged_metrics_reverse_order_10seeds.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
