"""
Chip Temperature Normalization Autoencoder Training Script

This script trains the first autoencoder for each chip to normalize temperature variations.
Purpose: Transform spectral data from various temperatures (T=25-29°C) to a standard 27°C reference.

Input: Normalized spectral data (32 peaks + 1 temperature) at various temperatures
Target: Same chip data at T=27°C (temperature normalization task)

This is the THIRD step in the pipeline after train_baseline_autoencoder.py.

New in this version:
- Track classifier-after-autoencoder metrics per chip
- Compute split-wise (train/val/test) averages across chips
- Save all metrics to JSON

Usage:
    python train_chip_temperature_autoencoder.py
"""
import os
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder

from utils.train_test_utils import (
    train_multitask_autoencoder,
    evaluate_encoder_decoder,
    train_classifier,
    evaluate_classifier
)
from utils.plot_utils import plot_train_and_val_losses, plot_conf_matrix, plot_denoised_data_combined
from utils.models import LinearDenoiser, Classifier
from utils.autoencoder_utils import save_autoencoder_output
from utils.config import (
    device, batch_size, learning_rate, num_epochs,
    autoencoder_patience, autoencoder_early_stopping,
    classifier_patience, classifier_early_stopping,
    norm_name, output_base_dir, norm_folder, num_chips, total_num_chips
)
from utils.data_utils import (
    load_and_preprocess_data_autoencoder_prenormalized,
    tensor_dataset_multitask_autoencoder
)
import pandas as pd


# ===== Output folder structure for this stage =====
# Root for this autoencoder's artifacts
RUN_ROOT = os.path.join(output_base_dir, "chip_to_temperature_autoencoder")
PLOTS_DIR = os.path.join(RUN_ROOT, "train_val_plots")
METRICS_DIR = os.path.join(RUN_ROOT, "metrics")
CONF_DIR = os.path.join(RUN_ROOT, "confusion_matrices")
MODEL_DIR = os.path.join(RUN_ROOT, "model")
AUTOENCODER_OUTPUT_DIR = os.path.join(RUN_ROOT, "autoencoder_output")
FINAL_PLOTS_DIR = os.path.join(RUN_ROOT, "final_plots")

# Make sure they exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(CONF_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(AUTOENCODER_OUTPUT_DIR, exist_ok=True)
os.makedirs(FINAL_PLOTS_DIR, exist_ok=True)

# -------------------------------
# Helpers for metrics & JSON I/O
# -------------------------------

def _tuple_to_metrics_dict(metric_tuple):
    """Convert (acc, prec, rec, f1) -> dict of floats safe for JSON."""
    acc, prec, rec, f1 = metric_tuple
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1)
    }


def _compute_average_metrics(per_chip_metrics):
    """
    Compute averages and standard deviations over chips for each split (train/val/test).
    per_chip_metrics: dict[chip_id] -> {split -> metrics dict}

    Returns:
        dict with "train"/"val"/"test" each mapping to averaged metrics and std.
    """
    import math

    splits = ("train", "val", "test")
    keys = ("accuracy", "precision", "recall", "f1")

    averages = {}
    # If no chips had metrics, return empty averages with num_chips = 0
    if not per_chip_metrics:
        for s in splits:
            averages[s] = {k: None for k in keys}
            averages[s]["num_chips"] = 0
        return averages

    for s in splits:
        # Collect metrics per chip for this split
        collected = []
        for chip_id, split_dict in per_chip_metrics.items():
            if s in split_dict and split_dict[s] is not None:
                collected.append(split_dict[s])

        if len(collected) == 0:
            averages[s] = {k: None for k in keys}
            averages[s]["num_chips"] = 0
            continue

        avg_entry = {}
        for k in keys:
            values = [m[k] for m in collected]
            mean = sum(values) / len(values)
            avg_entry[k] = float(mean)

            # Compute standard deviation
            if len(values) > 1:
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                std = math.sqrt(variance)
                avg_entry[f"{k}_std"] = float(std)
            else:
                avg_entry[f"{k}_std"] = 0.0

        avg_entry["num_chips"] = len(collected)
        averages[s] = avg_entry

    return averages


def _save_metrics_json(per_chip_metrics, averages, summary, out_dir, norm_name):
    """
    Save metrics to JSON file in out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(
        out_dir,
        f"classifier_after_1st_autoencoder_metrics_{norm_name}.json"
    )

    payload = {
        "summary": summary,
        "averages": averages,
        "per_chip": per_chip_metrics
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"✓ Classifier metrics (per-chip & averages) saved to: {output_path}")
    return output_path


# -------------------------------
# Core training functions (unchanged behavior, safe additions)
# -------------------------------

def train_denoising_autoencoder(chip_id, train_loader, val_loader, test_loader, num_classes,
                                 train_eval_loader=None):
    """
    Train denoising autoencoder for a specific chip with multi-task learning.

    Args:
        chip_id: Chip identifier
        train_loader: Training data loader (with class labels, shuffle=True)
        val_loader: Validation data loader (with class labels)
        test_loader: Test data loader (with class labels)
        num_classes: Number of classes for classification head
        train_eval_loader: Non-shuffled train loader for evaluation (maintains index alignment)

    Returns:
        Tuple of (model, training_losses, validation_losses, denoised_data_splits)
        where denoised_data_splits = (denoised_train, denoised_val, denoised_test)
    """
    print(f"\n{'='*80}")
    print(f"TRAINING DENOISING AUTOENCODER - CHIP {chip_id} (Multi-Task Learning)")
    print(f"{'='*80}\n")
    print(f"Number of classes: {num_classes}")

    # Initialize model with classification head
    model = LinearDenoiser(input_size=33, output_size=32, num_classes=num_classes).to(device)
    reconstruction_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.33, mode='min', patience=autoencoder_patience
    )

    # Train autoencoder with multi-task learning
    autoencoder_model_name = f'autoencoder_{norm_name}_chip_{chip_id}'
    model_denoiser, training_losses, validation_losses = train_multitask_autoencoder(
        epochs=num_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        reconstruction_criterion=reconstruction_criterion,
        classification_criterion=classification_criterion,
        scheduler=scheduler,
        model=model,
        device=device,
        model_name=autoencoder_model_name,
        early_stopping_patience=autoencoder_early_stopping,
        alpha=0.9,  # Weight for reconstruction loss
        beta=0.1,   # Weight for classification loss
        output_dir=MODEL_DIR  # Save to chip_to_temperature_autoencoder/model/
    )

    # Save loss history to JSON
    loss_history = {
        'train_loss': [float(x) for x in training_losses['total']],
        'val_loss': [float(x) for x in validation_losses['total']],
        'epochs': list(range(len(training_losses['total'])))
    }
    loss_history_path = os.path.join(PLOTS_DIR, f'{autoencoder_model_name}_loss_history.json')
    with open(loss_history_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    print(f"  Saved autoencoder loss history to: {loss_history_path}")

    # Plot losses (using 'total' loss from the dict)
    plot_train_and_val_losses(
        training_losses['total'],
        validation_losses['total'],
        autoencoder_model_name,
        output_dir=PLOTS_DIR
    )

    # Evaluate the first autoencoder on train/val/test sets to get denoised outputs
    # Use non-shuffled train_eval_loader to maintain alignment with original indices
    eval_loader_to_use = train_eval_loader if train_eval_loader is not None else train_loader
    _, denoised_train_data, _ = evaluate_encoder_decoder(
        model_encoder_decoder=model_denoiser,
        test_loader=eval_loader_to_use,  # Use non-shuffled loader for alignment
        criterion=reconstruction_criterion,
        device=device
    )

    _, denoised_val_data, _ = evaluate_encoder_decoder(
        model_encoder_decoder=model_denoiser,
        test_loader=val_loader,
        criterion=reconstruction_criterion,
        device=device
    )

    _, denoised_test_data, _ = evaluate_encoder_decoder(
        model_encoder_decoder=model_denoiser,
        test_loader=test_loader,
        criterion=reconstruction_criterion,
        device=device
    )

    print(f"✓ Denoising autoencoder for chip {chip_id} trained successfully")

    return model_denoiser, training_losses, validation_losses, (denoised_train_data, denoised_val_data, denoised_test_data)


def train_denoised_classifier(chip_id, denoised_data_splits, indices_splits, original_df, class_col):
    """
    Train classifier on denoised data to verify class preservation.

    Args:
        chip_id: Chip identifier
        denoised_data_splits: Tuple of (train, val, test) denoised data
        indices_splits: Tuple of (train, val, test) indices
        original_df: Original dataframe with class labels
        class_col: Name of class column

    Returns:
        Dictionary with train/val/test metrics
    """
    if class_col is None:
        print("No class column found, skipping classifier training")
        return None

    print(f"\n{'='*60}")
    print(f"Training classifier on DENOISED data (chip {chip_id}) to check class preservation...")
    print(f"{'='*60}\n")

    denoised_train_data, denoised_val_data, denoised_test_data = denoised_data_splits
    indices_train, indices_val, indices_test = indices_splits

    # Create label encoder for this chip
    denoised_label_encoder = LabelEncoder()
    encoded_classes = denoised_label_encoder.fit_transform(original_df[class_col].values)

    # Get class labels for train/val/test splits (using encoded values)
    train_class_labels = torch.tensor(encoded_classes[indices_train], dtype=torch.long)
    val_class_labels = torch.tensor(encoded_classes[indices_val], dtype=torch.long)
    test_class_labels = torch.tensor(encoded_classes[indices_test], dtype=torch.long)

    # Prepare data loaders for classifier
    denoised_train_dataset = TensorDataset(denoised_train_data, train_class_labels)
    denoised_val_dataset = TensorDataset(denoised_val_data, val_class_labels)
    denoised_test_dataset = TensorDataset(denoised_test_data, test_class_labels)

    denoised_train_loader = DataLoader(denoised_train_dataset, batch_size=batch_size, shuffle=True)
    denoised_val_loader = DataLoader(denoised_val_dataset, batch_size=batch_size, shuffle=False)
    denoised_test_loader = DataLoader(denoised_test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize classifier
    num_classes = len(denoised_label_encoder.classes_)
    denoised_classifier = Classifier(input_length=32, num_classes=num_classes).to(device)
    classifier_criterion = nn.CrossEntropyLoss()
    classifier_optimizer = optim.AdamW(denoised_classifier.parameters(), lr=learning_rate, weight_decay=1e-4)
    classifier_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        classifier_optimizer, mode='min', patience=classifier_patience
    )

    # Train classifier
    classifier_model_name = f'classifier_after_1st_autoencoder_{norm_name}_chip_{chip_id}'
    denoised_classifier, clf_train_losses, clf_val_losses = train_classifier(
        epochs=num_epochs,
        train_loader=denoised_train_loader,
        val_loader=denoised_val_loader,
        optimizer=classifier_optimizer,
        criterion=classifier_criterion,
        scheduler=classifier_scheduler,
        model_classifier=denoised_classifier,
        device=device,
        model_classifier_name=classifier_model_name,
        early_stopping_patience=classifier_early_stopping,
        output_dir=MODEL_DIR
    )

    # Save classifier loss history to JSON
    clf_loss_history = {
        'train_loss': [float(x) for x in clf_train_losses],
        'val_loss': [float(x) for x in clf_val_losses],
        'epochs': list(range(len(clf_train_losses)))
    }
    clf_loss_history_path = os.path.join(PLOTS_DIR, f'{classifier_model_name}_loss_history.json')
    with open(clf_loss_history_path, 'w') as f:
        json.dump(clf_loss_history, f, indent=2)
    print(f"  Saved classifier loss history to: {clf_loss_history_path}")

    # Plot training losses
    plot_train_and_val_losses(
        clf_train_losses, clf_val_losses, classifier_model_name, output_dir=PLOTS_DIR
    )

    # Evaluate on all splits
    train_acc, train_prec, train_rec, train_f1, train_conf_mat = evaluate_classifier(
        model_classifier=denoised_classifier,
        test_loader=denoised_train_loader,
        device=device,
        label_encoder=denoised_label_encoder
    )
    plot_conf_matrix(
        train_conf_mat, denoised_label_encoder,
        model_name=f'{classifier_model_name}_training_eval', output_dir=CONF_DIR
    )

    val_acc, val_prec, val_rec, val_f1, val_conf_mat = evaluate_classifier(
        model_classifier=denoised_classifier,
        test_loader=denoised_val_loader,
        device=device,
        label_encoder=denoised_label_encoder
    )
    plot_conf_matrix(
        val_conf_mat, denoised_label_encoder,
        model_name=f'{classifier_model_name}_validation_eval', output_dir=CONF_DIR
    )

    test_acc, test_prec, test_rec, test_f1, test_conf_mat = evaluate_classifier(
        model_classifier=denoised_classifier,
        test_loader=denoised_test_loader,
        device=device,
        label_encoder=denoised_label_encoder
    )
    plot_conf_matrix(
        test_conf_mat, denoised_label_encoder,
        model_name=f'{classifier_model_name}_test_eval', output_dir=CONF_DIR
    )

    print(f"✓ Confusion matrices saved in: {CONF_DIR}")

    print(f"\n{'='*60}")
    print(f"Classifier Results After 1st Autoencoder (Chip {chip_id}):")
    print(f"{'='*60}")
    print(f"Train - Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}")
    print(f"Val   - Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")
    print(f"Test  - Acc: {test_acc:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}, F1: {test_f1:.4f}")
    print(f"{'='*60}\n")

    return {
        'train': (train_acc, train_prec, train_rec, train_f1),
        'val': (val_acc, val_prec, val_rec, val_f1),
        'test': (test_acc, test_prec, test_rec, test_f1)
    }


def main():
    """Main chip temperature autoencoder training pipeline."""

    print("\n" + "="*80)
    print("CHIP TEMPERATURE NORMALIZATION AUTOENCODER TRAINING")
    print("="*80)
    print(f"Configuration: {total_num_chips} chips, {norm_name}")
    print(f"Output directory: {output_base_dir}")
    print("="*80 + "\n")

    # Ensure output directory exists (for plots & JSON)
    os.makedirs(output_base_dir, exist_ok=True)

    # -------------------------------
    # NEW: Track per-chip metrics
    # -------------------------------
    per_chip_metrics = {}
    num_processed = 0  # chips we actually attempted (dataset exists)

    # Storage for visualization data across all chips
    all_denoised = {'train': [], 'val': [], 'test': []}
    all_labels = {'train': [], 'val': [], 'test': []}

    # Process each chip
    for chip_idx, chip_id in enumerate(num_chips):
        print(f"\n{'='*80}")
        print(f"PROCESSING CHIP {chip_id}/{total_num_chips} ({chip_idx + 1} of {len(num_chips)})")
        print(f"{'='*80}\n")

        # Load chip dataset (created by data_preparation.py)
        temp_file_path = f"data/out/shuffled_dataset/{chip_id}_self_match_27C.csv"

        if not os.path.exists(temp_file_path):
            print(f"❌ Dataset not found: {temp_file_path}")
            print(f"   Please run data_preparation.py first!")
            continue

        num_processed += 1
        print(f"Loading chip dataset from: {temp_file_path}")

        # Load and split data into train/val/test sets
        (
            X_train, y_train, X_val, y_val,
            X_test, y_test,
            class_train, class_val, class_test,
            label_encoder,
            indices_train, indices_val, indices_test
        ) = load_and_preprocess_data_autoencoder_prenormalized(
            file_path=temp_file_path, finetune=False
        )

        # Load original dataframe for class labels (needed for saving outputs and getting raw labels)
        original_df = pd.read_csv(temp_file_path)
        class_col = "train_Class" if "train_Class" in original_df.columns else None

        # Get RAW (unencoded) class labels for each split from original dataframe
        # Note: class_train/val/test from function are already encoded, but we need raw labels for some operations
        class_train_raw = original_df.iloc[indices_train][class_col].values if class_col else None
        class_val_raw = original_df.iloc[indices_val][class_col].values if class_col else None
        class_test_raw = original_df.iloc[indices_test][class_col].values if class_col else None

        # Create multi-task PyTorch dataloaders (includes class labels for multi-task learning)
        # Use raw labels here as tensor_dataset_multitask_autoencoder will encode them
        train_loader, val_loader, test_loader, indices_train, indices_val, indices_test = tensor_dataset_multitask_autoencoder(
            batch_size=batch_size,
            X_train=X_train, y_train=y_train, class_train=class_train_raw,
            X_val=X_val, y_val=y_val, class_val=class_val_raw,
            X_test=X_test, y_test=y_test, class_test=class_test_raw,
            indices_train=indices_train, indices_val=indices_val, indices_test=indices_test
        )

        # Create NON-SHUFFLED train loader for evaluation to maintain index alignment
        # The train_loader has shuffle=True which breaks alignment with indices
        # Use the already-encoded class labels from the function return
        train_eval_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train.values, dtype=torch.float32),
                torch.tensor(y_train.values, dtype=torch.float32),
                torch.tensor(class_train, dtype=torch.long)
            ),
            batch_size=batch_size, shuffle=False, drop_last=False
        )

        # Train temperature normalization autoencoder (1st autoencoder)
        num_classes = len(label_encoder.classes_)
        denoising_model, denoising_train_losses, denoising_val_losses, denoised_data_splits = train_denoising_autoencoder(
            chip_id, train_loader, val_loader, test_loader, num_classes, train_eval_loader
        )

        # Save denoised outputs
        denoised_output_file = save_autoencoder_output(
            denoised_data_splits,
            (indices_train, indices_val, indices_test),
            chip_id, original_df, class_col, AUTOENCODER_OUTPUT_DIR, norm_name
        )
        print(f"✓ Temperature-normalized data saved to: {denoised_output_file}")

        # Train classifier on denoised data
        denoised_classifier_metrics = train_denoised_classifier(
            chip_id, denoised_data_splits,
            (indices_train, indices_val, indices_test),
            original_df, class_col
        )

        # -------------------------------
        # NEW: Store per-chip metrics
        # -------------------------------
        if denoised_classifier_metrics is not None:
            per_chip_metrics[str(chip_id)] = {
                "train": _tuple_to_metrics_dict(denoised_classifier_metrics["train"]),
                "val":   _tuple_to_metrics_dict(denoised_classifier_metrics["val"]),
                "test":  _tuple_to_metrics_dict(denoised_classifier_metrics["test"]),
            }
        else:
            print(f"ℹ️  Skipping metrics storage for chip {chip_id} (no class column / no classifier).")

        # Collect denoised data for visualization
        if class_col is not None:
            # Create label encoder for visualization
            plot_label_encoder = LabelEncoder()
            all_encoded_classes = plot_label_encoder.fit_transform(original_df[class_col].values)

            train_class_labels = torch.tensor(all_encoded_classes[indices_train], dtype=torch.long)
            val_class_labels = torch.tensor(all_encoded_classes[indices_val], dtype=torch.long)
            test_class_labels = torch.tensor(all_encoded_classes[indices_test], dtype=torch.long)

            # Collect denoised data from temperature autoencoder
            denoised_train_data, denoised_val_data, denoised_test_data = denoised_data_splits
            all_denoised['train'].append(denoised_train_data)
            all_denoised['val'].append(denoised_val_data)
            all_denoised['test'].append(denoised_test_data)

            all_labels['train'].append(train_class_labels)
            all_labels['val'].append(val_class_labels)
            all_labels['test'].append(test_class_labels)

        print(f"✓ Chip {chip_id} processing complete\n")

    # -------------------------------
    # NEW: Compute averages & save JSON
    # -------------------------------
    averages = _compute_average_metrics(per_chip_metrics)
    summary = {
        "norm_name": norm_name,
        "num_chips_requested": len(num_chips),
        "num_chips_processed": num_processed,
        "num_chips_with_metrics": len(per_chip_metrics),
        "generated_at": datetime.now().isoformat()
    }
    _save_metrics_json(per_chip_metrics, averages, summary, METRICS_DIR, norm_name)

    # Create combined visualizations (mean plots across all chips)
    print(f"\n{'='*80}")
    print(f"CREATING COMBINED VISUALIZATIONS")
    print(f"{'='*80}\n")

    if all_denoised['train']:
        print(f"Creating combined temperature-normalized plots across all {total_num_chips} chips...")
        plot_denoised_data_combined(all_denoised['train'], all_labels['train'], 'train', output_dir=FINAL_PLOTS_DIR)
        plot_denoised_data_combined(all_denoised['val'], all_labels['val'], 'val', output_dir=FINAL_PLOTS_DIR)
        plot_denoised_data_combined(all_denoised['test'], all_labels['test'], 'test', output_dir=FINAL_PLOTS_DIR)

    print(f"\n{'='*80}")
    print(f"CHIP TEMPERATURE NORMALIZATION COMPLETE!")
    print(f"{'='*80}")
    print(f"✓ {len(num_chips)} chips requested")
    print(f"✓ {num_processed} chips processed")
    print(f"✓ Temperature-normalized outputs saved to: {output_base_dir}")
    print(f"✓ Classifier metrics JSON saved in: {output_base_dir}")
    print(f"✓ Final combined plots saved to: {FINAL_PLOTS_DIR}")
    print(f"\nNext step: Run train_chip_to_baseline_autoencoder.py")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
