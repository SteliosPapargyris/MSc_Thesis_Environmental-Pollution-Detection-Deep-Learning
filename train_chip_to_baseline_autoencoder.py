"""
Chip-to-Baseline Transfer Autoencoder Training Script (PARALLEL TRAINING)

This script trains the transfer autoencoder to map chip spectral data to baseline chip space.
Purpose: Standardize chip-specific variations by mapping to a universal baseline reference.

IMPORTANT: This is PARALLEL training - it runs independently of the temperature autoencoder.

Input: Normalized chip spectral data (32 peaks, from data_preparation.py)
Target: Baseline chip spectral data, matched by class AND temperature

This can be run in parallel with train_chip_temperature_autoencoder.py.

Usage:
    python train_chip_to_baseline_autoencoder.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder

from utils.train_test_utils import train_multitask_autoencoder, evaluate_encoder_decoder, train_classifier, evaluate_classifier
from utils.plot_utils import plot_train_and_val_losses, plot_conf_matrix, plot_transferred_data_combined, plot_denoised_data_combined
from utils.models import LinearDenoiser, Classifier
from utils.autoencoder_utils import create_raw_baseline_targets, save_transfer_output, merge_chip_outputs, collect_data_for_visualization
from utils.config import (device, batch_size, learning_rate, num_epochs,
                          autoencoder_patience, autoencoder_early_stopping,
                          classifier_patience, classifier_early_stopping,
                          norm_name, output_base_dir, norm_folder, num_chips, total_num_chips, extended)
from utils.data_utils import load_and_preprocess_data_autoencoder_prenormalized
import pandas as pd
import os
import json
from datetime import datetime

# ===== Output folder structure for this stage =====
RUN_ROOT = os.path.join(output_base_dir, "chip_to_baseline_autoencoder")
PLOTS_DIR = os.path.join(RUN_ROOT, "train_val_plots")
CONF_DIR = os.path.join(RUN_ROOT, "confusion_matrices")
MODEL_DIR = os.path.join(RUN_ROOT, "model")
TRANSFER_OUTPUT_DIR = os.path.join(RUN_ROOT, "transfer_output")
DENOISED_OUTPUT_DIR = os.path.join(output_base_dir, "chip_to_temperature_autoencoder", "autoencoder_output")
METRICS_DIR = os.path.join(RUN_ROOT, "metrics")
FINAL_PLOTS_DIR = os.path.join(RUN_ROOT, "final_plots")

# Make sure they exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CONF_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TRANSFER_OUTPUT_DIR, exist_ok=True)
os.makedirs(DENOISED_OUTPUT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(FINAL_PLOTS_DIR, exist_ok=True)

def _tuple_to_metrics_dict(metric_tuple):
    acc, prec, rec, f1 = metric_tuple
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }

def _compute_average_metrics(per_chip_metrics):
    splits = ("train", "val", "test")
    keys = ("accuracy", "precision", "recall", "f1")
    averages = {}
    if not per_chip_metrics:
        for s in splits:
            averages[s] = {k: None for k in keys}
            averages[s]["num_chips"] = 0
        return averages

    for s in splits:
        collected = [split_dict[s] for split_dict in per_chip_metrics.values() if s in split_dict and split_dict[s] is not None]
        if not collected:
            averages[s] = {k: None for k in keys}
            averages[s]["num_chips"] = 0
            continue
        avg_entry = {k: float(sum(m[k] for m in collected) / len(collected)) for k in keys}
        avg_entry["num_chips"] = len(collected)
        averages[s] = avg_entry
    return averages

def _save_metrics_json(per_chip_metrics, averages, summary, out_dir, norm_name):
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"classifier_after_2nd_autoencoder_metrics_{norm_name}.json")
    payload = {"summary": summary, "averages": averages, "per_chip": per_chip_metrics}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"✓ Classifier metrics (per-chip & averages) saved to: {output_path}")
    return output_path

def train_transfer_autoencoder(chip_id, original_data_splits, indices_splits,
                               original_df, class_col, baseline_full_df):
    """
    Train transfer autoencoder to map chip data to baseline space (PARALLEL TRAINING).

    Args:
        chip_id: Chip identifier
        original_data_splits: Tuple of (train, val, test) normalized chip data (NOT denoised)
        indices_splits: Tuple of (train, val, test) indices
        original_df: Original dataframe with class labels
        class_col: Name of class column
        baseline_full_df: Full normalized baseline dataframe (all ~400 rows)

    Returns:
        Tuple of (model, training_losses, validation_losses, transferred_data_splits)
        where transferred_data_splits = (transferred_train, transferred_val, transferred_test)
    """
    print(f"\n{'='*80}")
    print(f"TRAINING TRANSFER AUTOENCODER - CHIP {chip_id} (PARALLEL TRAINING)")
    print(f"Input: Normalized chip data (32 peaks)")
    print(f"Target: Normalized baseline chip data (~20k rows, matched by class & temperature)")
    print(f"Note: This runs in parallel with temperature autoencoder, not sequentially")
    print(f"{'='*80}\n")

    original_train_data, original_val_data, original_test_data = original_data_splits
    indices_train, indices_val, indices_test = indices_splits

    # Create baseline targets aligned with the data splits using full normalized baseline
    if class_col is None:
        raise ValueError("Class column not found - cannot create baseline targets")

    # Determine temperature column name (could be "Temperature" or "train_Temperature")
    temp_col = "train_Temperature" if "train_Temperature" in original_df.columns else "Temperature"

    y_baseline_train = create_raw_baseline_targets(indices_train, original_df, class_col, baseline_full_df, temp_col_name=temp_col)
    y_baseline_val = create_raw_baseline_targets(indices_val, original_df, class_col, baseline_full_df, temp_col_name=temp_col)
    y_baseline_test = create_raw_baseline_targets(indices_test, original_df, class_col, baseline_full_df, temp_col_name=temp_col)

    # Get class labels for each split (already encoded)
    transfer_label_encoder = LabelEncoder()
    encoded_classes = transfer_label_encoder.fit_transform(original_df[class_col].values)
    class_train = torch.tensor(encoded_classes[indices_train], dtype=torch.long)
    class_val = torch.tensor(encoded_classes[indices_val], dtype=torch.long)
    class_test = torch.tensor(encoded_classes[indices_test], dtype=torch.long)
    num_classes = len(transfer_label_encoder.classes_)

    # Convert to tensors
    y_baseline_train_tensor = torch.tensor(y_baseline_train, dtype=torch.float32)
    y_baseline_val_tensor = torch.tensor(y_baseline_val, dtype=torch.float32)
    y_baseline_test_tensor = torch.tensor(y_baseline_test, dtype=torch.float32)

    # Create multi-task data loaders (include class labels)
    transfer_train_dataset = TensorDataset(original_train_data.squeeze(), y_baseline_train_tensor, class_train)
    transfer_val_dataset = TensorDataset(original_val_data.squeeze(), y_baseline_val_tensor, class_val)
    transfer_test_dataset = TensorDataset(original_test_data.squeeze(), y_baseline_test_tensor, class_test)

    # Create shuffled loader for training
    transfer_train_loader = DataLoader(transfer_train_dataset, batch_size=batch_size, shuffle=True)
    transfer_val_loader = DataLoader(transfer_val_dataset, batch_size=batch_size, shuffle=False)
    transfer_test_loader = DataLoader(transfer_test_dataset, batch_size=batch_size, shuffle=False)

    # Create NON-SHUFFLED train loader for evaluation to maintain index alignment
    transfer_train_eval_loader = DataLoader(transfer_train_dataset, batch_size=batch_size, shuffle=False)

    print(f"Number of classes: {num_classes}")

    # Initialize model with classification head
    transfer_model = LinearDenoiser(input_size=32, output_size=32, num_classes=num_classes).to(device)
    reconstruction_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()
    transfer_optimizer = optim.AdamW(transfer_model.parameters(), lr=learning_rate)
    transfer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(transfer_optimizer, factor=0.33, mode='min', patience=autoencoder_patience)

    transfer_model_name = f'transfer_autoencoder_{norm_name}_chip_{chip_id}_to_baseline'

    # Train transfer autoencoder with multi-task learning
    trained_transfer_model, transfer_training_losses, transfer_validation_losses = train_multitask_autoencoder(
        epochs=num_epochs,
        train_loader=transfer_train_loader,
        val_loader=transfer_val_loader,
        optimizer=transfer_optimizer,
        reconstruction_criterion=reconstruction_criterion,
        classification_criterion=classification_criterion,
        scheduler=transfer_scheduler,
        model=transfer_model,
        device=device,
        model_name=transfer_model_name,
        early_stopping_patience=autoencoder_early_stopping,
        alpha=0.3,  # Weight for reconstruction loss
        beta=0.7,   # Weight for classification loss
        output_dir=MODEL_DIR  # Save to chip_to_baseline_autoencoder/model/
    )

    # Save transfer autoencoder loss history to JSON
    transfer_loss_history = {
        'train_loss': [float(x) for x in transfer_training_losses['total']],
        'val_loss': [float(x) for x in transfer_validation_losses['total']],
        'epochs': list(range(len(transfer_training_losses['total'])))
    }
    transfer_loss_history_path = os.path.join(PLOTS_DIR, f'{transfer_model_name}_loss_history.json')
    with open(transfer_loss_history_path, 'w') as f:
        json.dump(transfer_loss_history, f, indent=2)
    print(f"  Saved transfer autoencoder loss history to: {transfer_loss_history_path}")

    # Plot losses (using 'total' loss from the dict)
    plot_train_and_val_losses(transfer_training_losses['total'], transfer_validation_losses['total'],
                              transfer_model_name, output_dir=PLOTS_DIR)

    # Evaluate the second autoencoder to get transferred (standardized) outputs
    # Use non-shuffled loader to maintain alignment with original indices
    _, transferred_train_data, _ = evaluate_encoder_decoder(
        model_encoder_decoder=trained_transfer_model,
        test_loader=transfer_train_eval_loader,  # Use non-shuffled loader for alignment!
        criterion=reconstruction_criterion,
        device=device
    )

    _, transferred_val_data, _ = evaluate_encoder_decoder(
        model_encoder_decoder=trained_transfer_model,
        test_loader=transfer_val_loader,
        criterion=reconstruction_criterion,
        device=device
    )

    _, transferred_test_data, _ = evaluate_encoder_decoder(
        model_encoder_decoder=trained_transfer_model,
        test_loader=transfer_test_loader,
        criterion=reconstruction_criterion,
        device=device
    )

    print(f"✓ Transfer autoencoder for chip {chip_id} trained successfully")

    return trained_transfer_model, transfer_training_losses, transfer_validation_losses, (transferred_train_data, transferred_val_data, transferred_test_data)


def train_transfer_classifier(chip_id, transferred_data_splits, indices_splits, original_df, class_col):
    """
    Train classifier on transferred data to verify class preservation after transfer.

    Args:
        chip_id: Chip identifier
        transferred_data_splits: Tuple of (train, val, test) transferred data
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
    print(f"Training classifier on TRANSFERRED data (chip {chip_id}) to check class preservation...")
    print(f"{'='*60}\n")

    transferred_train_data, transferred_val_data, transferred_test_data = transferred_data_splits
    indices_train, indices_val, indices_test = indices_splits

    # Create label encoder for this chip
    transfer_label_encoder = LabelEncoder()
    encoded_classes = transfer_label_encoder.fit_transform(original_df[class_col].values)

    # Get class labels for train/val/test splits (using encoded values)
    transfer_train_class_labels = torch.tensor(encoded_classes[indices_train], dtype=torch.long)
    transfer_val_class_labels = torch.tensor(encoded_classes[indices_val], dtype=torch.long)
    transfer_test_class_labels = torch.tensor(encoded_classes[indices_test], dtype=torch.long)

    # Prepare data loaders for classifier using transferred data
    transferred_train_dataset = TensorDataset(transferred_train_data, transfer_train_class_labels)
    transferred_val_dataset = TensorDataset(transferred_val_data, transfer_val_class_labels)
    transferred_test_dataset = TensorDataset(transferred_test_data, transfer_test_class_labels)

    transferred_train_loader = DataLoader(transferred_train_dataset, batch_size=batch_size, shuffle=True)
    transferred_val_loader = DataLoader(transferred_val_dataset, batch_size=batch_size, shuffle=False)
    transferred_test_loader = DataLoader(transferred_test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize classifier
    num_transfer_classes = len(transfer_label_encoder.classes_)
    transfer_classifier = Classifier(input_length=32, num_classes=num_transfer_classes).to(device)
    transfer_clf_criterion = nn.CrossEntropyLoss()
    transfer_clf_optimizer = optim.AdamW(transfer_classifier.parameters(), lr=learning_rate, weight_decay=1e-4)
    transfer_clf_scheduler = optim.lr_scheduler.ReduceLROnPlateau(transfer_clf_optimizer, mode='min', patience=classifier_patience)

    # Train classifier
    transfer_classifier_model_name = f'classifier_after_2nd_autoencoder_{norm_name}_chip_{chip_id}'
    transfer_classifier, transfer_clf_train_losses, transfer_clf_val_losses = train_classifier(
        epochs=num_epochs,
        train_loader=transferred_train_loader,
        val_loader=transferred_val_loader,
        optimizer=transfer_clf_optimizer,
        criterion=transfer_clf_criterion,
        scheduler=transfer_clf_scheduler,
        model_classifier=transfer_classifier,
        device=device,
        model_classifier_name=transfer_classifier_model_name,
        early_stopping_patience=classifier_early_stopping,
        output_dir=MODEL_DIR
    )

    # Save transfer classifier loss history to JSON
    transfer_clf_loss_history = {
        'train_loss': [float(x) for x in transfer_clf_train_losses],
        'val_loss': [float(x) for x in transfer_clf_val_losses],
        'epochs': list(range(len(transfer_clf_train_losses)))
    }
    transfer_clf_loss_history_path = os.path.join(PLOTS_DIR, f'{transfer_classifier_model_name}_loss_history.json')
    with open(transfer_clf_loss_history_path, 'w') as f:
        json.dump(transfer_clf_loss_history, f, indent=2)
    print(f"  Saved transfer classifier loss history to: {transfer_clf_loss_history_path}")

    # Plot training losses
    plot_train_and_val_losses(transfer_clf_train_losses, transfer_clf_val_losses, transfer_classifier_model_name, output_dir=PLOTS_DIR)

    # Evaluate on all splits
    transfer_train_acc, transfer_train_prec, transfer_train_rec, transfer_train_f1, transfer_train_conf_mat = evaluate_classifier(
        model_classifier=transfer_classifier,
        test_loader=transferred_train_loader,
        device=device,
        label_encoder=transfer_label_encoder
    )
    plot_conf_matrix(transfer_train_conf_mat, transfer_label_encoder,
                    model_name=f'{transfer_classifier_model_name}_training_eval', output_dir=CONF_DIR)

    transfer_val_acc, transfer_val_prec, transfer_val_rec, transfer_val_f1, transfer_val_conf_mat = evaluate_classifier(
        model_classifier=transfer_classifier,
        test_loader=transferred_val_loader,
        device=device,
        label_encoder=transfer_label_encoder,
    )
    plot_conf_matrix(transfer_val_conf_mat, transfer_label_encoder,
                    model_name=f'{transfer_classifier_model_name}_validation_eval', output_dir=CONF_DIR)

    transfer_test_acc, transfer_test_prec, transfer_test_rec, transfer_test_f1, transfer_test_conf_mat = evaluate_classifier(
        model_classifier=transfer_classifier,
        test_loader=transferred_test_loader,
        device=device,
        label_encoder=transfer_label_encoder,
    )
    plot_conf_matrix(transfer_test_conf_mat, transfer_label_encoder,
                    model_name=f'{transfer_classifier_model_name}_test_eval', output_dir=CONF_DIR)

    print(f"\n{'='*60}")
    print(f"Classifier Results After 2nd Autoencoder (Chip {chip_id}):")
    print(f"{'='*60}")
    print(f"Train - Acc: {transfer_train_acc:.4f}, Prec: {transfer_train_prec:.4f}, Rec: {transfer_train_rec:.4f}, F1: {transfer_train_f1:.4f}")
    print(f"Val   - Acc: {transfer_val_acc:.4f}, Prec: {transfer_val_prec:.4f}, Rec: {transfer_val_rec:.4f}, F1: {transfer_val_f1:.4f}")
    print(f"Test  - Acc: {transfer_test_acc:.4f}, Prec: {transfer_test_prec:.4f}, Rec: {transfer_test_rec:.4f}, F1: {transfer_test_f1:.4f}")
    print(f"{'='*60}\n")

    return {
        'train': (transfer_train_acc, transfer_train_prec, transfer_train_rec, transfer_train_f1),
        'val': (transfer_val_acc, transfer_val_prec, transfer_val_rec, transfer_val_f1),
        'test': (transfer_test_acc, transfer_test_prec, transfer_test_rec, transfer_test_f1)
    }


def main():
    """Main chip-to-baseline transfer autoencoder training pipeline."""

    print("\n" + "="*80)
    print("CHIP-TO-BASELINE TRANSFER AUTOENCODER TRAINING")
    print("="*80)
    print(f"Configuration: {total_num_chips} chips, {norm_name}")
    print(f"Output directory: {RUN_ROOT}")
    print("="*80 + "\n")

    per_chip_metrics = {}
    num_processed = 0

    # Load baseline full data (for targets)
    # Use extended dataset if configured
    baseline_suffix = "extended_" if extended else ""
    baseline_full_path = f"data/out/{norm_folder}/baseline/with_standard_scaler/baseline_chip_{baseline_suffix}{norm_folder}.csv"
    if not os.path.exists(baseline_full_path):
        print(f"❌ Baseline data not found: {baseline_full_path}")
        print(f"   Please run data_preparation.py first!")
        return

    baseline_full_df = pd.read_csv(baseline_full_path)
    print(f"Loaded baseline reference: {len(baseline_full_df)} rows (extended={extended})")

    # Storage for visualization data across all chips
    all_transferred = {'train': [], 'val': [], 'test': []}
    all_denoised = {'train': [], 'val': [], 'test': []}
    all_labels = {'train': [], 'val': [], 'test': []}

    # Process each chip
    for chip_idx, chip_id in enumerate(num_chips):
        print(f"\n{'='*80}")
        print(f"PROCESSING CHIP {chip_id}/{total_num_chips} ({chip_idx + 1} of {len(num_chips)})")
        print(f"{'='*80}\n")

        # Load chip dataset from normalized data (PARALLEL TRAINING)
        # Input: Normalized chip data directly (not from temperature autoencoder)
        # Use extended dataset if configured
        chips_folder = f"{total_num_chips}chips_extended"
        chip_data_path = f"data/out/{norm_folder}/{chips_folder}/with_standard_scaler/chip_{chip_id}_{norm_folder}.csv"

        if not os.path.exists(chip_data_path):
            print(f"❌ Chip data not found: {chip_data_path}")
            print(f"   Please run data_preparation.py first!")
            continue

        print(f"Loading normalized chip data from: {chip_data_path} (extended={extended})")

        # Load and split the normalized chip data
        (
            X_train, y_train, X_val, y_val, X_test, y_test,
            class_train, class_val, class_test,
            label_encoder,
            indices_train, indices_val, indices_test
        ) = load_and_preprocess_data_autoencoder_prenormalized(
            file_path=chip_data_path, finetune=False
        )

        # Load original dataframe for class labels
        original_df = pd.read_csv(chip_data_path)
        # Check for different class column names (train_Class or Class)
        if "train_Class" in original_df.columns:
            class_col = "train_Class"
        elif "Class" in original_df.columns:
            class_col = "Class"
        else:
            class_col = None

        if class_col is None:
            print(f"❌ No class column found in {chip_data_path}")
            print(f"   Available columns: {list(original_df.columns)}")
            continue

        # Use the peak columns from X_train/val/test (32 peaks only, no temperature)
        # For parallel training, we use only the 32 peak features as input
        original_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)  # Use y_train which has 32 peaks
        original_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
        original_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
        original_data_splits = (original_train_tensor, original_val_tensor, original_test_tensor)

        # Train transfer autoencoder (chip to baseline)
        transfer_model, transfer_train_losses, transfer_val_losses, transferred_data_splits = train_transfer_autoencoder(
            chip_id, original_data_splits,
            (indices_train, indices_val, indices_test),
            original_df, class_col, baseline_full_df
        )

        # Save transferred outputs
        transfer_output_file = save_transfer_output(
            transferred_data_splits,
            (indices_train, indices_val, indices_test),
            chip_id, original_df, class_col, TRANSFER_OUTPUT_DIR, norm_name
        )
        print(f"✓ Transferred data saved to: {transfer_output_file}")

        num_processed += 1

        # Train classifier on transferred data
        transfer_classifier_metrics = train_transfer_classifier(
            chip_id, transferred_data_splits,
            (indices_train, indices_val, indices_test),
            original_df, class_col
        )

        # Store metrics for this chip
        per_chip_metrics[chip_id] = {
            'train': _tuple_to_metrics_dict(transfer_classifier_metrics['train']),
            'val': _tuple_to_metrics_dict(transfer_classifier_metrics['val']),
            'test': _tuple_to_metrics_dict(transfer_classifier_metrics['test'])
        }

        # Collect data for visualization
        if class_col is not None:
            # Create label encoder for visualization
            plot_label_encoder = LabelEncoder()
            all_encoded_classes = plot_label_encoder.fit_transform(original_df[class_col].values)

            train_class_labels = torch.tensor(all_encoded_classes[indices_train], dtype=torch.long)
            val_class_labels = torch.tensor(all_encoded_classes[indices_val], dtype=torch.long)
            test_class_labels = torch.tensor(all_encoded_classes[indices_test], dtype=torch.long)

            # Collect transferred data
            collect_data_for_visualization(
                transferred_data_splits,
                (original_train_tensor.squeeze(), original_val_tensor.squeeze(), original_test_tensor.squeeze()),
                (train_class_labels, val_class_labels, test_class_labels),
                all_transferred, all_denoised, all_labels
            )

        print(f"✓ Chip {chip_id} processing complete\n")

    # Merge outputs from all chips
    print(f"\n{'='*80}")
    print(f"MERGING OUTPUTS FROM ALL CHIPS")
    print(f"{'='*80}\n")

    # ---- Save metrics JSON (per-chip + averages), same style as temperature script ----
    averages = _compute_average_metrics(per_chip_metrics)
    summary = {
        "norm_name": norm_name,
        "num_chips_requested": len(num_chips),
        "num_chips_processed": num_processed,
        "num_chips_with_metrics": len(per_chip_metrics),
        "generated_at": datetime.now().isoformat(),
    }
    _save_metrics_json(per_chip_metrics, averages, summary, METRICS_DIR, norm_name)


    # Merge transfer autoencoder outputs
    merged_transfer_path = merge_chip_outputs(num_chips, TRANSFER_OUTPUT_DIR, norm_name, output_type='transfer')
    if merged_transfer_path:
        print(f"✓ Transfer outputs merged: {merged_transfer_path}")

    # Merge denoising autoencoder outputs (from previous step)
    merged_denoised_path = merge_chip_outputs(num_chips, DENOISED_OUTPUT_DIR, norm_name, output_type='denoised')
    if merged_denoised_path:
        print(f"✓ Denoised outputs merged: {merged_denoised_path}")

    # Create combined visualizations
    print(f"\n{'='*80}")
    print(f"CREATING COMBINED VISUALIZATIONS")
    print(f"{'='*80}\n")

    # Plot transferred data from all chips combined
    if all_transferred['train']:
        print(f"Creating combined transferred plots across all {total_num_chips} chips...")
        plot_transferred_data_combined(all_transferred['train'], all_labels['train'], 'train', output_dir=FINAL_PLOTS_DIR)
        plot_transferred_data_combined(all_transferred['val'], all_labels['val'], 'val', output_dir=FINAL_PLOTS_DIR)
        plot_transferred_data_combined(all_transferred['test'], all_labels['test'], 'test', output_dir=FINAL_PLOTS_DIR)

    # Plot denoised data from all chips combined
    if all_denoised['train']:
        print(f"Creating combined denoised plots across all {total_num_chips} chips...")
        plot_denoised_data_combined(all_denoised['train'], all_labels['train'], 'train', output_dir=FINAL_PLOTS_DIR)
        plot_denoised_data_combined(all_denoised['val'], all_labels['val'], 'val', output_dir=FINAL_PLOTS_DIR)
        plot_denoised_data_combined(all_denoised['test'], all_labels['test'], 'test', output_dir=FINAL_PLOTS_DIR)

    print(f"\n{'='*80}")
    print(f"CHIP-TO-BASELINE TRANSFER COMPLETE!")
    print(f"{'='*80}")
    print(f"✓ {len(num_chips)} chips processed")
    print(f"✓ Transferred outputs saved to: {TRANSFER_OUTPUT_DIR}")
    print(f"✓ Classifier metrics JSON saved in: {METRICS_DIR}")
    print(f"✓ Merged datasets created")
    print(f"✓ Final combined plots saved to: {FINAL_PLOTS_DIR}")
    print(f"\n🎉 FULL PIPELINE COMPLETE!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
