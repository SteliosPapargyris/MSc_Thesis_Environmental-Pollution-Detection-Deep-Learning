"""
Inference Script with Joint Training of Autoencoders and Classifier

This script performs the complete pipeline:
1. Load train/val/test data from all chips (70-20-10 split)
2. Load pre-trained chip-specific autoencoders (trainable)
3. Train global classifier AND autoencoders jointly on train data
4. Evaluate on train/val/test datasets
5. Generate comprehensive metrics (JSON) and confusion matrices (PNG) for all datasets
6. Save trained models, predictions, and visualizations

Pipeline:
Data (33: 32 peaks + temp) -> Autoencoder_1st (trainable) -> Autoencoder_2nd (trainable)
  -> 32 features -> Global Classifier (trainable) -> Class prediction

Key Points:
- Training is ENABLED by default (trains joint model)
- To skip training and use saved models, comment out train_joint_model() call
  and uncomment the model loading section (lines 754-794)
- Generates JSON metrics for train, val, and test datasets
- Creates confusion matrix plots for train, val, and test datasets
- Saves predictions with extracted features for all datasets

Outputs:
- Trained models: classifier + autoencoders
- JSON metrics: train_metrics, val_metrics, test_metrics
- PNG plots: confusion matrices for train, val, test
- CSV files: predictions with features for train, val, test

Usage:
    python inference_with_training.py
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from utils.config import (
    device, seed, num_chips, norm_name, norm_folder,
    output_base_dir, total_num_chips, extended
)
from utils.models import LinearDenoiser, Classifier


def load_chip_data_and_split(chip_id, norm_folder, total_num_chips, extended, seed=42):
    """
    Load chip data and perform train/val/test split.

    Args:
        chip_id: Chip identifier
        norm_folder: Normalization folder (e.g., 'robust', 'mean_std')
        total_num_chips: Total number of chips
        extended: Whether to use extended dataset
        seed: Random seed for splitting

    Returns:
        Dictionary with keys 'train', 'val', 'test', each containing:
        (X, y, temp, chip_id_array)
    """
    # Construct chip data path
    chips_folder = f"{total_num_chips}chips_extended" if extended else f"{total_num_chips}chips"
    chip_data_path = f"data/out/{norm_folder}/{chips_folder}/with_standard_scaler/chip_{chip_id}_{norm_folder}.csv"

    if not os.path.exists(chip_data_path):
        raise FileNotFoundError(f"Chip data not found: {chip_data_path}")

    # Load chip data
    df = pd.read_csv(chip_data_path)

    # Extract features (32 peaks) and labels
    peak_cols = [f'Peak {i}' for i in range(1, 33)]
    X = df[peak_cols].values

    # Get class labels
    if "train_Class" in df.columns:
        class_col = "train_Class"
    elif "Class" in df.columns:
        class_col = "Class"
    else:
        raise ValueError(f"No class column found in {chip_data_path}")

    y = df[class_col].values

    # Get temperature column
    if "train_Temperature" in df.columns:
        temp_col = "train_Temperature"
    elif "Temperature" in df.columns:
        temp_col = "Temperature"
    else:
        raise ValueError(f"No temperature column found in {chip_data_path}")

    temp = df[temp_col].values

    # Scale temperature to [-1, 1] range (matching training)
    temp_scaler = MinMaxScaler(feature_range=(-1, 1))
    temp_scaled = temp_scaler.fit_transform(temp.reshape(-1, 1)).flatten()

    # Perform 70-20-10 split
    X_train, X_temp_split, y_train, y_temp_split, temp_train, temp_temp_split = train_test_split(
        X, y, temp_scaled, test_size=0.3, random_state=seed, stratify=y
    )

    X_val, X_test, y_val, y_test, temp_val, temp_test = train_test_split(
        X_temp_split, y_temp_split, temp_temp_split, test_size=0.33333, random_state=seed, stratify=y_temp_split
    )

    # Create chip ID arrays for each split
    chip_id_train = np.full(len(X_train), chip_id)
    chip_id_val = np.full(len(X_val), chip_id)
    chip_id_test = np.full(len(X_test), chip_id)

    return {
        'train': (X_train, y_train, temp_train, chip_id_train),
        'val': (X_val, y_val, temp_val, chip_id_val),
        'test': (X_test, y_test, temp_test, chip_id_test)
    }


def load_autoencoder_models(chip_id, norm_name, output_base_dir, num_classes, device, total_num_chips):
    """
    Load the two autoencoder models for a specific chip.

    Args:
        chip_id: Chip identifier
        norm_name: Normalization name (e.g., 'robust_normalized')
        output_base_dir: Base output directory
        num_classes: Number of classes for the models
        device: Device to load models on
        total_num_chips: Total number of chips (for path construction)

    Returns:
        Tuple of (autoencoder_1st, autoencoder_2nd)
    """
    # Paths to saved autoencoder models - use dynamic path based on config
    # Use output_base_dir from config (supports environment variable override for hyperparameter search)
    model_1st_path = f"{output_base_dir}/chip_to_temperature_autoencoder/model/autoencoder_{norm_name}_chip_{chip_id}.pth"
    model_2nd_path = f"{output_base_dir}/chip_to_baseline_autoencoder/model/transfer_autoencoder_{norm_name}_chip_{chip_id}_to_baseline.pth"

    # Check if models exist
    if not os.path.exists(model_1st_path):
        raise FileNotFoundError(f"1st autoencoder model not found: {model_1st_path}")
    if not os.path.exists(model_2nd_path):
        raise FileNotFoundError(f"2nd autoencoder model not found: {model_2nd_path}")

    # Initialize models
    autoencoder_1st = LinearDenoiser(input_size=33, output_size=32, num_classes=num_classes).to(device)
    autoencoder_2nd = LinearDenoiser(input_size=32, output_size=32, num_classes=num_classes).to(device)

    # Load weights
    autoencoder_1st.load_state_dict(torch.load(model_1st_path, map_location=device))
    autoencoder_2nd.load_state_dict(torch.load(model_2nd_path, map_location=device))

    # Set to training mode (NOT eval mode!)
    autoencoder_1st.train()
    autoencoder_2nd.train()

    print(f" Loaded models for chip {chip_id} (train mode)")

    return autoencoder_1st, autoencoder_2nd


def sequential_forward_with_gradients(data_with_temp, autoencoder_1st, autoencoder_2nd):
    """
    Perform sequential forward pass through both autoencoders with gradient tracking.

    Args:
        data_with_temp: Input data with temperature (33 features)
        autoencoder_1st: First autoencoder (temperature normalization)
        autoencoder_2nd: Second autoencoder (baseline transfer)

    Returns:
        Tuple of (final_output, reconstruction_1st, reconstruction_2nd, class_logits_1st, class_logits_2nd)
    """
    # Pass through 1st autoencoder (33 -> 32)
    # Autoencoder returns (reconstruction, latent_features, class_logits)
    reconstruction_1st, latent_1st, class_logits_1st = autoencoder_1st(data_with_temp)

    # Pass through 2nd autoencoder (32 -> 32)
    # Autoencoder returns (reconstruction, latent_features, class_logits)
    reconstruction_2nd, latent_2nd, class_logits_2nd = autoencoder_2nd(reconstruction_1st)

    return reconstruction_2nd, reconstruction_1st, class_logits_1st, class_logits_2nd


def train_joint_model(train_loader, val_loader, autoencoder_models, num_classes, device,
                      learning_rate=0.001, num_epochs=100, patience=15,
                      alpha_recon=0.5, alpha_class_ae=0.1, alpha_class_global=1.0):
    """
    Train autoencoders and global classifier jointly.

    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        autoencoder_models: Dictionary of {chip_id: (autoencoder_1st, autoencoder_2nd)}
        num_classes: Number of classes
        device: Device to train on
        learning_rate: Learning rate
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        alpha_recon: Weight for reconstruction loss
        alpha_class_ae: Weight for autoencoder classification loss
        alpha_class_global: Weight for global classifier loss

    Returns:
        Trained classifier model and updated autoencoder models
    """
    print(f"\n{'='*80}")
    print("TRAINING JOINT MODEL (AUTOENCODERS + GLOBAL CLASSIFIER)")
    print(f"{'='*80}")
    print(f"  Input features: 32 peaks + 1 temperature = 33")
    print(f"  Num classes: {num_classes}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max epochs: {num_epochs}")
    print(f"  Loss weights:")
    print(f"    - Reconstruction: {alpha_recon}")
    print(f"    - Autoencoder classification: {alpha_class_ae}")
    print(f"    - Global classification: {alpha_class_global}")
    print(f"{'='*80}\n")

    # Initialize global classifier
    classifier = Classifier(input_length=32, num_classes=num_classes).to(device)

    # Collect all parameters from all models
    all_params = list(classifier.parameters())
    for chip_id, (ae1, ae2) in autoencoder_models.items():
        all_params.extend(list(ae1.parameters()))
        all_params.extend(list(ae2.parameters()))

    # Create optimizer for all models
    optimizer = optim.Adam(all_params, lr=learning_rate)

    # Loss functions
    criterion_classification = nn.CrossEntropyLoss()
    criterion_reconstruction = nn.MSELoss()

    # Training loop with early stopping
    best_val_acc = 0.0
    best_model_states = None
    epochs_without_improvement = 0

    # Track loss history
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        # Training phase
        classifier.train()
        for chip_id, (ae1, ae2) in autoencoder_models.items():
            ae1.train()
            ae2.train()

        train_loss_total = 0.0
        train_loss_recon = 0.0
        train_loss_class_ae = 0.0
        train_loss_class_global = 0.0
        train_correct = 0
        train_total = 0

        for batch_data in train_loader:
            batch_X, batch_temp, batch_y, batch_chip_ids = batch_data
            batch_X = batch_X.to(device)
            batch_temp = batch_temp.to(device)
            batch_y = batch_y.to(device)
            batch_chip_ids = batch_chip_ids.cpu().numpy()

            optimizer.zero_grad()

            # Process each chip separately (different autoencoders per chip)
            batch_outputs = []
            batch_recon_losses = []
            batch_class_ae_losses = []

            for chip_id in np.unique(batch_chip_ids):
                if chip_id not in autoencoder_models:
                    continue

                # TODO -> 1. reduce batch size to 2, then just load 10 autoencoders instead of having 2 models in autoencoder_models, i should have 10. not empty autoencoder models
                
                # Get indices for this chip in the batch
                chip_mask = batch_chip_ids == chip_id
                chip_indices = np.where(chip_mask)[0]

                # Get data for this chip
                X_chip = batch_X[chip_indices]
                temp_chip = batch_temp[chip_indices]
                y_chip = batch_y[chip_indices]

                # Concatenate 32 peaks + 1 temperature = 33 features
                data_with_temp = torch.cat([X_chip, temp_chip.unsqueeze(1)], dim=1)

                # Get autoencoders for this chip
                autoencoder_1st, autoencoder_2nd = autoencoder_models[chip_id]

                # If batch size is 1, temporarily set to eval mode to avoid BatchNorm issues
                # BatchNorm requires at least 2 samples in train mode
                temp_eval_mode = len(X_chip) == 1
                if temp_eval_mode:
                    autoencoder_1st.eval()
                    autoencoder_2nd.eval()

                # Forward pass through autoencoders
                final_output, recon_1st, class_logits_1st, class_logits_2nd = sequential_forward_with_gradients(
                    data_with_temp, autoencoder_1st, autoencoder_2nd
                )

                # Restore to train mode if needed
                if temp_eval_mode:
                    autoencoder_1st.train()
                    autoencoder_2nd.train()

                # Store outputs for classifier
                batch_outputs.append((chip_indices, final_output))

                # Reconstruction loss: compare final output to original 32 peaks
                recon_loss = criterion_reconstruction(final_output.squeeze(1), X_chip)
                batch_recon_losses.append(recon_loss)

                # Classification loss from autoencoders
                class_ae_loss = (
                    criterion_classification(class_logits_1st.squeeze(1), y_chip) +
                    criterion_classification(class_logits_2nd.squeeze(1), y_chip)
                ) / 2.0
                batch_class_ae_losses.append(class_ae_loss)

            # Combine outputs in original batch order
            combined_outputs = torch.zeros(len(batch_X), 1, 32).to(device)
            for indices, outputs in batch_outputs:
                combined_outputs[indices] = outputs

            # Forward pass through global classifier
            global_class_outputs = classifier(combined_outputs)
            class_global_loss = criterion_classification(global_class_outputs, batch_y)

            # Combined loss
            # TODO -> only class_global_loss
            total_loss = (
                # alpha_recon * torch.mean(torch.stack(batch_recon_losses)) +
                # alpha_class_ae * torch.mean(torch.stack(batch_class_ae_losses)) +
                alpha_class_global * class_global_loss
            )

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Track metrics
            train_loss_total += total_loss.item() * len(batch_X)
            train_loss_recon += torch.mean(torch.stack(batch_recon_losses)).item() * len(batch_X)
            train_loss_class_ae += torch.mean(torch.stack(batch_class_ae_losses)).item() * len(batch_X)
            train_loss_class_global += class_global_loss.item() * len(batch_X)

            _, predicted = torch.max(global_class_outputs, 1)
            train_correct += (predicted == batch_y).sum().item()
            train_total += len(batch_y)

        train_loss_total /= train_total
        train_loss_recon /= train_total
        train_loss_class_ae /= train_total
        train_loss_class_global /= train_total
        train_acc = 100.0 * train_correct / train_total

        # Validation phase
        classifier.eval()
        for chip_id, (ae1, ae2) in autoencoder_models.items():
            ae1.eval()
            ae2.eval()

        val_loss_total = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_data in val_loader:
                batch_X, batch_temp, batch_y, batch_chip_ids = batch_data
                batch_X = batch_X.to(device)
                batch_temp = batch_temp.to(device)
                batch_y = batch_y.to(device)
                batch_chip_ids = batch_chip_ids.cpu().numpy()

                # Process each chip separately
                combined_outputs = torch.zeros(len(batch_X), 1, 32).to(device)

                for chip_id in np.unique(batch_chip_ids):
                    if chip_id not in autoencoder_models:
                        continue

                    chip_mask = batch_chip_ids == chip_id
                    chip_indices = np.where(chip_mask)[0]

                    X_chip = batch_X[chip_indices]
                    temp_chip = batch_temp[chip_indices]

                    data_with_temp = torch.cat([X_chip, temp_chip.unsqueeze(1)], dim=1)

                    autoencoder_1st, autoencoder_2nd = autoencoder_models[chip_id]

                    # Forward through autoencoders
                    output_1st = autoencoder_1st(data_with_temp)[0]
                    output_2nd = autoencoder_2nd(output_1st)[0]

                    combined_outputs[chip_indices] = output_2nd

                # Forward through classifier
                global_class_outputs = classifier(combined_outputs)
                val_loss = criterion_classification(global_class_outputs, batch_y)

                val_loss_total += val_loss.item() * len(batch_X)
                _, predicted = torch.max(global_class_outputs, 1)
                val_correct += (predicted == batch_y).sum().item()
                val_total += len(batch_y)

        val_loss_total /= val_total
        val_acc = 100.0 * val_correct / val_total

        # Track loss history
        train_loss_history.append(train_loss_total)
        val_loss_history.append(val_loss_total)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}]")
            print(f"  Train: Loss={train_loss_total:.4f} (Recon={train_loss_recon:.4f}, "
                  f"AE_Class={train_loss_class_ae:.4f}, Global_Class={train_loss_class_global:.4f}) "
                  f"Acc={train_acc:.2f}%")
            print(f"  Val:   Loss={val_loss_total:.4f}, Acc={val_acc:.2f}%")

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save states of all models
            best_model_states = {
                'classifier': classifier.state_dict().copy(),
                'autoencoders': {chip_id: (ae1.state_dict().copy(), ae2.state_dict().copy())
                                 for chip_id, (ae1, ae2) in autoencoder_models.items()}
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\n✓ Early stopping triggered at epoch {epoch+1}")
            print(f"  Best validation accuracy: {best_val_acc:.2f}%")
            break

    # Load best models
    if best_model_states is not None:
        classifier.load_state_dict(best_model_states['classifier'])
        for chip_id, (ae1, ae2) in autoencoder_models.items():
            ae1_state, ae2_state = best_model_states['autoencoders'][chip_id]
            ae1.load_state_dict(ae1_state)
            ae2.load_state_dict(ae2_state)
        print(f"\n✓ Loaded best models with validation accuracy: {best_val_acc:.2f}%")

    return classifier, autoencoder_models, train_loss_history, val_loss_history


def process_data(data_loader, autoencoder_models, classifier, device):
    """
    Process data through autoencoders and classifier.

    This function can be used for train, val, or test data.

    Args:
        data_loader: DataLoader for data
        autoencoder_models: Dictionary of autoencoder models
        classifier: Trained classifier
        device: Device to run on

    Returns:
        Tuple of (predictions, true_labels, features)
    """
    classifier.eval()
    for chip_id, (ae1, ae2) in autoencoder_models.items():
        ae1.eval()
        ae2.eval()

    all_predictions = []
    all_labels = []
    all_features = []

    with torch.no_grad():
        for batch_data in data_loader:
            batch_X, batch_temp, batch_y, batch_chip_ids = batch_data
            batch_X = batch_X.to(device)
            batch_temp = batch_temp.to(device)
            batch_chip_ids = batch_chip_ids.cpu().numpy()

            # Process each chip separately
            combined_outputs = torch.zeros(len(batch_X), 1, 32).to(device)

            for chip_id in np.unique(batch_chip_ids):
                if chip_id not in autoencoder_models:
                    continue

                chip_mask = batch_chip_ids == chip_id
                chip_indices = np.where(chip_mask)[0]

                X_chip = batch_X[chip_indices]
                temp_chip = batch_temp[chip_indices]

                data_with_temp = torch.cat([X_chip, temp_chip.unsqueeze(1)], dim=1)

                autoencoder_1st, autoencoder_2nd = autoencoder_models[chip_id]

                # Forward through autoencoders
                output_1st = autoencoder_1st(data_with_temp)[0]
                output_2nd = autoencoder_2nd(output_1st)[0]

                combined_outputs[chip_indices] = output_2nd

            # Forward through classifier
            global_class_outputs = classifier(combined_outputs)
            _, predicted = torch.max(global_class_outputs, 1)

            all_predictions.append(predicted.cpu().numpy())
            all_labels.append(batch_y.numpy())
            all_features.append(combined_outputs.squeeze(1).cpu().numpy())

    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)
    features = np.vstack(all_features)

    return predictions, labels, features


def compute_metrics_dict(y_true, y_pred, dataset_name="test"):
    """
    Compute comprehensive metrics and return as dictionary.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        dataset_name: Name of dataset (train/val/test)

    Returns:
        Dictionary containing all metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    # Weighted averages
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    conf_matrix = confusion_matrix(y_true, y_pred)

    # Build metrics dictionary
    metrics = {
        'dataset': dataset_name,
        'accuracy': float(accuracy),
        'macro_precision': float(precision_macro),
        'macro_recall': float(recall_macro),
        'macro_f1': float(f1_macro),
        'weighted_precision': float(precision_weighted),
        'weighted_recall': float(recall_weighted),
        'weighted_f1': float(f1_weighted),
        'per_class_metrics': {},
        'confusion_matrix': conf_matrix.tolist()
    }

    # Add per-class metrics
    for class_idx in range(len(precision)):
        metrics['per_class_metrics'][f'class_{class_idx}'] = {
            'precision': float(precision[class_idx]),
            'recall': float(recall[class_idx]),
            'f1_score': float(f1[class_idx]),
            'support': int(support[class_idx])
        }

    return metrics


def plot_confusion_matrix(y_true, y_pred, output_path, dataset_name="Test", class_names=None):
    """
    Plot and save confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save the plot
        dataset_name: Name of dataset (Train/Val/Test)
        class_names: Optional list of class names
    """
    conf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))

    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(conf_matrix))]

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})

    plt.title(f'Confusion Matrix - {dataset_name} Set', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Confusion matrix plot saved to: {output_path}")


def main():
    """Main pipeline with joint training of autoencoders and classifier."""

    print("\n" + "="*80)
    print("JOINT TRAINING: AUTOENCODERS + GLOBAL CLASSIFIER")
    print("="*80)
    print(f"Configuration: {total_num_chips} chips, {norm_name}")
    print(f"Extended dataset: {extended}")
    print(f"Device: {device}")
    print("="*80 + "\n")

    # Storage for train/val/test data from all chips
    all_train_data, all_train_labels, all_train_temp, all_train_chip_ids = [], [], [], []
    all_val_data, all_val_labels, all_val_temp, all_val_chip_ids = [], [], [], []
    all_test_data, all_test_labels, all_test_temp, all_test_chip_ids = [], [], [], []

    # Step 1: Load train/val/test data from all chips
    print("Step 1: Loading train/val/test data from all chips...")
    print("-" * 80)

    for chip_id in num_chips:
        try:
            splits = load_chip_data_and_split(
                chip_id, norm_folder, total_num_chips, extended, seed
            )

            # Unpack train/val/test splits
            X_train, y_train, temp_train, chip_id_train = splits['train']
            X_val, y_val, temp_val, chip_id_val = splits['val']
            X_test, y_test, temp_test, chip_id_test = splits['test']

            # Append to storage
            all_train_data.append(X_train)
            all_train_labels.append(y_train)
            all_train_temp.append(temp_train)
            all_train_chip_ids.append(chip_id_train)

            all_val_data.append(X_val)
            all_val_labels.append(y_val)
            all_val_temp.append(temp_val)
            all_val_chip_ids.append(chip_id_val)

            all_test_data.append(X_test)
            all_test_labels.append(y_test)
            all_test_temp.append(temp_test)
            all_test_chip_ids.append(chip_id_test)

            print(f"  ✓ Chip {chip_id}: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples")

        except Exception as e:
            print(f"  ❌ Error loading chip {chip_id}: {e}")
            continue

    # Concatenate all data
    all_train_data = np.vstack(all_train_data)
    all_train_labels = np.concatenate(all_train_labels)
    all_train_temp = np.concatenate(all_train_temp)
    all_train_chip_ids = np.concatenate(all_train_chip_ids)

    all_val_data = np.vstack(all_val_data)
    all_val_labels = np.concatenate(all_val_labels)
    all_val_temp = np.concatenate(all_val_temp)
    all_val_chip_ids = np.concatenate(all_val_chip_ids)

    all_test_data = np.vstack(all_test_data)
    all_test_labels = np.concatenate(all_test_labels)
    all_test_temp = np.concatenate(all_test_temp)
    all_test_chip_ids = np.concatenate(all_test_chip_ids)

    print(f"\n✓ Data loading complete:")
    print(f"  Train: {len(all_train_data)} samples")
    print(f"  Val: {len(all_val_data)} samples")
    print(f"  Test: {len(all_test_data)} samples")
    print(f"  Unique classes (before conversion): {np.unique(all_train_labels)}")

    # Convert labels to 0-indexed
    min_label = np.min(all_train_labels)
    if min_label > 0:
        print(f"\n⚠️  Converting labels from {min_label}-indexed to 0-indexed...")
        all_train_labels = all_train_labels - min_label
        all_val_labels = all_val_labels - min_label
        all_test_labels = all_test_labels - min_label
        print(f"  Unique classes (after conversion): {np.unique(all_train_labels)}")

    # Determine number of classes
    num_classes = len(np.unique(all_train_labels))
    print(f"  Number of classes: {num_classes}")

    # Step 2: Load all autoencoder models
    print("\n" + "="*80)
    print("Step 2: Loading autoencoder models for all chips (trainable mode)...")
    print("-" * 80)

    autoencoder_models = {}
    for chip_id in num_chips:
        try:
            model_1st, model_2nd = load_autoencoder_models(
                chip_id, norm_name, output_base_dir, num_classes, device, total_num_chips
            )
            autoencoder_models[chip_id] = (model_1st, model_2nd)
        except Exception as e:
            print(f"  ❌ Error loading models for chip {chip_id}: {e}")
            continue

    print(f"\n✓ Loaded autoencoder models for {len(autoencoder_models)} chips")

    # Step 3: Create DataLoaders
    print("\n" + "="*80)
    print("Step 3: Creating DataLoaders...")
    print("-" * 80)

    train_dataset = TensorDataset(
        torch.tensor(all_train_data, dtype=torch.float32),
        torch.tensor(all_train_temp, dtype=torch.float32),
        torch.tensor(all_train_labels, dtype=torch.long),
        torch.tensor(all_train_chip_ids, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(all_val_data, dtype=torch.float32),
        torch.tensor(all_val_temp, dtype=torch.float32),
        torch.tensor(all_val_labels, dtype=torch.long),
        torch.tensor(all_val_chip_ids, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(all_test_data, dtype=torch.float32),
        torch.tensor(all_test_temp, dtype=torch.float32),
        torch.tensor(all_test_labels, dtype=torch.long),
        torch.tensor(all_test_chip_ids, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    print(f"✓ DataLoaders created")

    # Step 4: Train joint model
    print("\n" + "="*80)
    print("Step 4: Training joint model (autoencoders + classifier)...")
    print("-" * 80)

    global_classifier, trained_autoencoders, train_loss_history, val_loss_history = train_joint_model(
        train_loader, val_loader, autoencoder_models, num_classes, device
    )

    # # INFERENCE-ONLY MODE - Uncomment below to skip training and load saved models
    # inference_output_dir = os.path.join(output_base_dir, "joint_training_results")
    # classifier_path = os.path.join(inference_output_dir, f"global_classifier_joint_{norm_name}.pth")

    # if not os.path.exists(classifier_path):
    #     raise FileNotFoundError(f"Saved classifier not found: {classifier_path}\n"
    #                             f"Please run the script with training enabled first.")

    # global_classifier = Classifier(input_length=32, num_classes=num_classes).to(device)
    # global_classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    # global_classifier.eval()
    # print(f"✓ Loaded global classifier from: {classifier_path}")

    # # Load the saved trained autoencoders
    # ae_dir = os.path.join(inference_output_dir, "trained_autoencoders")
    # trained_autoencoders = {}

    # for chip_id in num_chips:
    #     ae1_path = os.path.join(ae_dir, f"autoencoder_1st_chip_{chip_id}.pth")
    #     ae2_path = os.path.join(ae_dir, f"autoencoder_2nd_chip_{chip_id}.pth")

    #     if not os.path.exists(ae1_path) or not os.path.exists(ae2_path):
    #         print(f"⚠️  Skipping chip {chip_id}: Saved autoencoders not found")
    #         continue

    #     # Initialize autoencoder models
    #     ae1 = LinearDenoiser(input_size=33, output_size=32, num_classes=num_classes).to(device)
    #     ae2 = LinearDenoiser(input_size=32, output_size=32, num_classes=num_classes).to(device)

    #     # Load saved weights
    #     ae1.load_state_dict(torch.load(ae1_path, map_location=device))
    #     ae2.load_state_dict(torch.load(ae2_path, map_location=device))

    #     # Set to eval mode
    #     ae1.eval()
    #     ae2.eval()

    #     trained_autoencoders[chip_id] = (ae1, ae2)
    #     print(f"✓ Loaded trained autoencoders for chip {chip_id}")

    # print(f"\n✓ Loaded {len(trained_autoencoders)} trained autoencoder pairs")

    # Step 5: Evaluate on train, val, and test data
    print("\n" + "="*80)
    print("Step 5: Evaluating on train, val, and test data...")
    print("-" * 80)

    # Evaluate on all three datasets
    print("\nProcessing train set...")
    train_predictions, train_labels, train_features = process_data(
        train_loader, trained_autoencoders, global_classifier, device
    )

    print("Processing validation set...")
    val_predictions, val_labels, val_features = process_data(
        val_loader, trained_autoencoders, global_classifier, device
    )

    print("Processing test set...")
    test_predictions, test_labels, test_features = process_data(
        test_loader, trained_autoencoders, global_classifier, device
    )

    # Calculate metrics for all datasets
    train_metrics = compute_metrics_dict(train_labels, train_predictions, "train")
    val_metrics = compute_metrics_dict(val_labels, val_predictions, "val")
    test_metrics = compute_metrics_dict(test_labels, test_predictions, "test")

    print(f"\n✓ Train Accuracy: {train_metrics['accuracy'] * 100:.2f}%")
    print(f"✓ Val Accuracy: {val_metrics['accuracy'] * 100:.2f}%")
    print(f"✓ Test Accuracy: {test_metrics['accuracy'] * 100:.2f}%")

    # Step 6: Save results
    print("\n" + "="*80)
    print("Step 6: Saving results...")
    print("-" * 80)

    # Create output directory
    inference_output_dir = os.path.join(output_base_dir, "joint_training_results")
    os.makedirs(inference_output_dir, exist_ok=True)

    # Save global classifier
    classifier_path = os.path.join(inference_output_dir, f"global_classifier_joint_{norm_name}.pth")
    torch.save(global_classifier.state_dict(), classifier_path)
    print(f"✓ Global classifier saved to: {classifier_path}")

    # Save trained autoencoders
    ae_dir = os.path.join(inference_output_dir, "trained_autoencoders")
    os.makedirs(ae_dir, exist_ok=True)
    for chip_id, (ae1, ae2) in trained_autoencoders.items():
        ae1_path = os.path.join(ae_dir, f"autoencoder_1st_chip_{chip_id}.pth")
        ae2_path = os.path.join(ae_dir, f"autoencoder_2nd_chip_{chip_id}.pth")
        torch.save(ae1.state_dict(), ae1_path)
        torch.save(ae2.state_dict(), ae2_path)
    print(f"✓ Trained autoencoders saved to: {ae_dir}")

    # Save metrics as JSON files
    print("\nSaving metrics JSON files...")
    train_metrics_path = os.path.join(inference_output_dir, f"train_metrics_{norm_name}.json")
    val_metrics_path = os.path.join(inference_output_dir, f"val_metrics_{norm_name}.json")
    test_metrics_path = os.path.join(inference_output_dir, f"test_metrics_{norm_name}.json")

    with open(train_metrics_path, 'w') as f:
        json.dump(train_metrics, f, indent=4)
    print(f"✓ Train metrics saved to: {train_metrics_path}")

    with open(val_metrics_path, 'w') as f:
        json.dump(val_metrics, f, indent=4)
    print(f"✓ Val metrics saved to: {val_metrics_path}")

    with open(test_metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=4)
    print(f"✓ Test metrics saved to: {test_metrics_path}")

    # Save training loss history to JSON
    loss_history = {
        'train_loss': [float(x) for x in train_loss_history],
        'val_loss': [float(x) for x in val_loss_history],
        'epochs': list(range(len(train_loss_history)))
    }
    loss_history_path = os.path.join(inference_output_dir, f"joint_training_loss_history_{norm_name}.json")
    with open(loss_history_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    print(f"✓ Training loss history saved to: {loss_history_path}")

    # Create training loss plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history['epochs'], loss_history['train_loss'], 'b-', linewidth=2, label='Training Loss')
    plt.plot(loss_history['epochs'], loss_history['val_loss'], 'orange', linewidth=2, label='Validation Loss')
    plt.title('Joint Training: Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    loss_plot_path = os.path.join(inference_output_dir, f"joint_training_loss_plot_{norm_name}.png")
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training loss plot saved to: {loss_plot_path}")

    # Plot confusion matrices for all datasets
    print("\nGenerating confusion matrix plots...")
    plots_dir = os.path.join(inference_output_dir, "confusion_matrices")
    os.makedirs(plots_dir, exist_ok=True)

    train_cm_path = os.path.join(plots_dir, f"confusion_matrix_train_{norm_name}.png")
    val_cm_path = os.path.join(plots_dir, f"confusion_matrix_val_{norm_name}.png")
    test_cm_path = os.path.join(plots_dir, f"confusion_matrix_test_{norm_name}.png")

    plot_confusion_matrix(train_labels, train_predictions, train_cm_path, dataset_name="Train")
    plot_confusion_matrix(val_labels, val_predictions, val_cm_path, dataset_name="Validation")
    plot_confusion_matrix(test_labels, test_predictions, test_cm_path, dataset_name="Test")

    # Convert labels back to original indexing for CSV files
    true_labels_train_original = train_labels + min_label if min_label > 0 else train_labels
    pred_labels_train_original = train_predictions + min_label if min_label > 0 else train_predictions

    true_labels_val_original = val_labels + min_label if min_label > 0 else val_labels
    pred_labels_val_original = val_predictions + min_label if min_label > 0 else val_predictions

    true_labels_test_original = test_labels + min_label if min_label > 0 else test_labels
    pred_labels_test_original = test_predictions + min_label if min_label > 0 else test_predictions

    # Save predictions for all datasets
    print("\nSaving prediction CSV files...")

    # Train predictions
    train_results_df = pd.DataFrame({
        'Chip_ID': all_train_chip_ids,
        'True_Class': true_labels_train_original,
        'Predicted_Class': pred_labels_train_original,
        'Correct': (train_labels == train_predictions).astype(int)
    })
    for i in range(train_features.shape[1]):
        train_results_df[f'Feature_{i}'] = train_features[:, i]
    train_predictions_path = os.path.join(inference_output_dir, f"train_predictions_joint_{norm_name}.csv")
    train_results_df.to_csv(train_predictions_path, index=False)
    print(f"✓ Train predictions saved to: {train_predictions_path}")

    # Val predictions
    val_results_df = pd.DataFrame({
        'Chip_ID': all_val_chip_ids,
        'True_Class': true_labels_val_original,
        'Predicted_Class': pred_labels_val_original,
        'Correct': (val_labels == val_predictions).astype(int)
    })
    for i in range(val_features.shape[1]):
        val_results_df[f'Feature_{i}'] = val_features[:, i]
    val_predictions_path = os.path.join(inference_output_dir, f"val_predictions_joint_{norm_name}.csv")
    val_results_df.to_csv(val_predictions_path, index=False)
    print(f"✓ Val predictions saved to: {val_predictions_path}")

    # Test predictions
    test_results_df = pd.DataFrame({
        'Chip_ID': all_test_chip_ids,
        'True_Class': true_labels_test_original,
        'Predicted_Class': pred_labels_test_original,
        'Correct': (test_labels == test_predictions).astype(int)
    })
    for i in range(test_features.shape[1]):
        test_results_df[f'Feature_{i}'] = test_features[:, i]
    test_predictions_path = os.path.join(inference_output_dir, f"test_predictions_joint_{norm_name}.csv")
    test_results_df.to_csv(test_predictions_path, index=False)
    print(f"✓ Test predictions saved to: {test_predictions_path}")

    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"✓ Trained joint model on {len(all_train_data)} train samples")
    print(f"✓ Validated on {len(all_val_data)} validation samples")
    print(f"✓ Tested on {len(all_test_data)} test samples")
    print(f"\nAccuracies:")
    print(f"  - Train: {train_metrics['accuracy'] * 100:.2f}%")
    print(f"  - Val:   {val_metrics['accuracy'] * 100:.2f}%")
    print(f"  - Test:  {test_metrics['accuracy'] * 100:.2f}%")
    print(f"\n✓ Trained models saved (autoencoders + classifier)")
    print(f"✓ Metrics JSON files saved for train, val, and test")
    print(f"✓ Confusion matrix plots saved for train, val, and test")
    print(f"✓ Prediction CSV files saved for train, val, and test")
    print(f"✓ All results saved to: {inference_output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
