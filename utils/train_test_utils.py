import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
from utils.config import *
import os
import json

def train_encoder_decoder(epochs, train_loader, val_loader, optimizer, criterion, scheduler, model_encoder_decoder, device, model_encoder_decoder_name, early_stopping_patience):
    early_stopping_counter = 0
    model_encoder_decoder.to(device)
    best_val_loss = float('inf')
    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        # Training phase
        model_encoder_decoder.train()
        total_train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)
            labels = labels.unsqueeze(1)
            optimizer.zero_grad()
            denoised_output, latent_space = model_encoder_decoder(inputs)
            loss = criterion(denoised_output, labels)
            # loss = criterion(latent_space, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        print(f'Epoch {epoch} - Training Loss: {avg_train_loss:.6f}')

        # Validation phase
        model_encoder_decoder.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.unsqueeze(1)
                labels = labels.unsqueeze(1)
                denoised_output, latent_space = model_encoder_decoder(inputs)
                loss = criterion(denoised_output, labels)
                # loss = criterion(latent_space, labels)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)
        print(f'Epoch {epoch} - Validation Loss: {avg_val_loss:.6f}')

        # Learning rate adjustment and checkpointing
        if scheduler:
            scheduler.step(avg_val_loss)
            current_lr = scheduler.get_last_lr()[0]
            print(f'Current learning rate: {current_lr}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print('Validation loss decreased, saving model.')
            # Create normalization-specific directory structure
            # Extract norm type from model name (e.g., "autoencoder_minmax_normalized_train" -> "minmax_normalized")
            name_parts = model_encoder_decoder_name.split('_')
            if len(name_parts) >= 3 and name_parts[0] == 'autoencoder':
                norm_type = '_'.join(name_parts[1:-1])  # Everything between 'autoencoder' and 'train'
            else:
                norm_type = 'default'
            model_dir = f'pths/{norm_type}'
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model_encoder_decoder.state_dict(), f'{model_dir}/{model_encoder_decoder_name}.pth')
            early_stopping_counter = 0  # Reset the counter on improvement
        else:
            early_stopping_counter += 1  # Increment counter if no improvement
            print(f'No improvement in validation loss for {early_stopping_counter} consecutive epochs.')

        # Early stopping condition
        if early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping triggered after {early_stopping_patience} epochs with no improvement in validation loss.')
            # Extract norm type from model name (e.g., "autoencoder_minmax_normalized_train" -> "minmax_normalized")
            name_parts = model_encoder_decoder_name.split('_')
            if len(name_parts) >= 3 and name_parts[0] == 'autoencoder':
                norm_type = '_'.join(name_parts[1:-1])  # Everything between 'autoencoder' and 'train'
            else:
                norm_type = 'default'
            model_dir = f'pths/{norm_type}'
            model_encoder_decoder.load_state_dict(torch.load(f'{model_dir}/{model_encoder_decoder_name}.pth'))
            print('Model restored to best state based on validation loss.')
            break
        print("\n")
    return model_encoder_decoder, training_losses, validation_losses


def train_multitask_autoencoder(epochs, train_loader, val_loader, optimizer, reconstruction_criterion,
                                classification_criterion, scheduler, model, device, model_name,
                                early_stopping_patience, alpha=1.0, beta=0.5, output_dir=None):
    """
    Train autoencoder with multi-task learning: reconstruction + classification.

    Args:
        epochs: Number of training epochs
        train_loader: Training data loader (inputs, targets, class_labels)
        val_loader: Validation data loader (inputs, targets, class_labels)
        optimizer: Optimizer
        reconstruction_criterion: Loss for reconstruction (e.g., MSELoss)
        classification_criterion: Loss for classification (e.g., CrossEntropyLoss)
        scheduler: Learning rate scheduler
        model: Multi-task autoencoder model (with classifier head)
        device: Device to train on
        model_name: Name for saving model
        early_stopping_patience: Patience for early stopping
        alpha: Weight for reconstruction loss (default: 1.0)
        beta: Weight for classification loss (default: 0.5)
        output_dir: Directory to save model checkpoints (default: None, uses 'pths/{norm_type}')

    Returns:
        Trained model, training losses dict, validation losses dict
    """
    early_stopping_counter = 0
    model.to(device)
    best_val_loss = float('inf')

    # Track losses separately
    train_recon_losses = []
    train_class_losses = []
    train_total_losses = []
    val_recon_losses = []
    val_class_losses = []
    val_total_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_recon_loss = 0
        total_train_class_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, targets, class_labels in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            class_labels = class_labels.to(device)

            inputs = inputs.unsqueeze(1)
            targets = targets.unsqueeze(1)

            optimizer.zero_grad()

            # Forward pass - handle both with/without classifier
            output = model(inputs)
            if len(output) == 3:  # With classifier
                reconstruction, latent, class_logits = output
                has_classifier = True
            else:  # Without classifier
                reconstruction, latent = output
                has_classifier = False

            # Compute losses
            recon_loss = reconstruction_criterion(reconstruction, targets)

            if has_classifier:
                class_loss = classification_criterion(class_logits, class_labels)
                total_loss = alpha * recon_loss + beta * class_loss
                total_train_class_loss += class_loss.item()

                # Compute accuracy
                _, predicted = torch.max(class_logits, 1)
                train_total += class_labels.size(0)
                train_correct += (predicted == class_labels).sum().item()
            else:
                total_loss = recon_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

            total_train_recon_loss += recon_loss.item()

        avg_train_recon_loss = total_train_recon_loss / len(train_loader)

        train_recon_losses.append(avg_train_recon_loss)

        print(f'Epoch {epoch}:')
        if has_classifier:
            avg_train_class_loss = total_train_class_loss / len(train_loader)
            avg_train_total_loss = alpha * avg_train_recon_loss + beta * avg_train_class_loss
            train_accuracy = 100 * train_correct / train_total

            train_class_losses.append(avg_train_class_loss)
            train_total_losses.append(avg_train_total_loss)

            print(f'  Training - Recon Loss: {avg_train_recon_loss:.6f}, Class Loss: {avg_train_class_loss:.6f}, Total: {avg_train_total_loss:.6f}, Acc: {train_accuracy:.2f}%')
        else:
            train_class_losses.append(0.0)
            train_total_losses.append(avg_train_recon_loss)
            print(f'  Training - Recon Loss: {avg_train_recon_loss:.6f}')

        # Validation phase
        model.eval()
        total_val_recon_loss = 0
        total_val_class_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets, class_labels in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                class_labels = class_labels.to(device)

                inputs = inputs.unsqueeze(1)
                targets = targets.unsqueeze(1)

                # Forward pass - handle both with/without classifier
                output = model(inputs)
                if len(output) == 3:  # With classifier
                    reconstruction, latent, class_logits = output
                    has_classifier = True
                else:  # Without classifier
                    reconstruction, latent = output
                    has_classifier = False

                # Compute losses
                recon_loss = reconstruction_criterion(reconstruction, targets)
                total_val_recon_loss += recon_loss.item()

                if has_classifier:
                    class_loss = classification_criterion(class_logits, class_labels)
                    total_val_class_loss += class_loss.item()

                    # Compute accuracy
                    _, predicted = torch.max(class_logits, 1)
                    val_total += class_labels.size(0)
                    val_correct += (predicted == class_labels).sum().item()

        avg_val_recon_loss = total_val_recon_loss / len(val_loader)
        val_recon_losses.append(avg_val_recon_loss)

        if has_classifier:
            avg_val_class_loss = total_val_class_loss / len(val_loader)
            avg_val_total_loss = alpha * avg_val_recon_loss + beta * avg_val_class_loss
            val_accuracy = 100 * val_correct / val_total

            val_class_losses.append(avg_val_class_loss)
            val_total_losses.append(avg_val_total_loss)

            print(f'  Validation - Recon Loss: {avg_val_recon_loss:.6f}, Class Loss: {avg_val_class_loss:.6f}, Total: {avg_val_total_loss:.6f}, Acc: {val_accuracy:.2f}%')
        else:
            val_class_losses.append(0.0)
            val_total_losses.append(avg_val_recon_loss)
            avg_val_total_loss = avg_val_recon_loss
            print(f'  Validation - Recon Loss: {avg_val_recon_loss:.6f}')

        # Learning rate adjustment
        if scheduler:
            scheduler.step(avg_val_total_loss)
            current_lr = scheduler.get_last_lr()[0]
            print(f'  Current learning rate: {current_lr}')

        # Save best model
        if avg_val_total_loss < best_val_loss:
            best_val_loss = avg_val_total_loss
            print('  Validation loss decreased, saving model.')

            # Use provided output_dir or fallback to pths/{norm_type}
            if output_dir is not None:
                model_dir = output_dir
            else:
                # Extract norm type from model name
                name_parts = model_name.split('_')
                if len(name_parts) >= 3:
                    norm_type = '_'.join(name_parts[1:-1]) if name_parts[0] in ['autoencoder', 'baseline'] else 'default'
                else:
                    norm_type = 'default'
                model_dir = f'pths/{norm_type}'

            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{model_dir}/{model_name}.pth')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f'  No improvement in validation loss for {early_stopping_counter} consecutive epochs.')

        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping triggered after {early_stopping_patience} epochs with no improvement.')

            # Use provided output_dir or fallback to pths/{norm_type}
            if output_dir is not None:
                model_dir = output_dir
            else:
                # Extract norm type
                name_parts = model_name.split('_')
                if len(name_parts) >= 3:
                    norm_type = '_'.join(name_parts[1:-1]) if name_parts[0] in ['autoencoder', 'baseline'] else 'default'
                else:
                    norm_type = 'default'
                model_dir = f'pths/{norm_type}'

            model.load_state_dict(torch.load(f'{model_dir}/{model_name}.pth'))
            print('Model restored to best state.')
            break

        print()

    # Return losses as dictionary
    training_losses = {
        'reconstruction': train_recon_losses,
        'classification': train_class_losses,
        'total': train_total_losses
    }
    validation_losses = {
        'reconstruction': val_recon_losses,
        'classification': val_class_losses,
        'total': val_total_losses
    }

    return model, training_losses, validation_losses


def evaluate_encoder_decoder(model_encoder_decoder, test_loader, device, criterion):
    model_encoder_decoder.eval()
    model_encoder_decoder.to(device)
    total_test_loss = 0
    denoised_data = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            # Handle both 2-tuple (inputs, labels) and 3-tuple (inputs, labels, class_labels)
            if len(batch) == 3:
                inputs, labels, _ = batch  # Ignore class labels for reconstruction-only evaluation
            else:
                inputs, labels = batch

            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.unsqueeze(1)
            labels = labels.unsqueeze(1)

            # Forward pass - handle both old and new model outputs
            outputs = model_encoder_decoder(inputs)
            if len(outputs) == 3:
                denoised_output, latent_space, _ = outputs  # Model with classifier
            else:
                denoised_output, latent_space = outputs  # Model without classifier

            loss = criterion(denoised_output, labels)
            total_test_loss += loss.item()
            # Store reconstruction output for downstream classification
            # denoised_output is already [batch, 1, 32], so just store it
            denoised_data.append(denoised_output.cpu())
            all_labels.append(labels.cpu())

    avg_test_loss = total_test_loss / len(test_loader)
    print(f'Final Test Loss: {avg_test_loss:.6f}')

    denoised_data = torch.cat(denoised_data, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    print(f'Denoised data shape: {denoised_data.shape}')  # Debug: check shape
    print(f'All labels shape: {all_labels.shape}')  # Debug: check shape

    return avg_test_loss, denoised_data, all_labels

def train_classifier(epochs, train_loader, val_loader, optimizer, criterion, scheduler,
                     model_classifier, device, model_classifier_name, early_stopping_patience,
                     output_dir='pths'):
    early_stopping_counter = 0
    model_classifier.to(device)
    best_val_loss = float('inf')
    training_losses, validation_losses = [], []

    # Always keep only the filename; save directly in output_dir/model
    safe_name = os.path.splitext(os.path.basename(model_classifier_name))[0]
    base_dir = output_dir or 'pths'
    model_dir = base_dir  # no subfolder
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, f'{safe_name}.pth')

    for epoch in range(epochs):
        # ---- Train ----
        model_classifier.train()
        total_train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        print(f'Epoch {epoch} - Training Loss: {avg_train_loss:.6f}')

        # ---- Validate ----
        model_classifier.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model_classifier(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)
        print(f'Epoch {epoch} - Validation Loss: {avg_val_loss:.6f}')

        # ---- Scheduler ----
        if scheduler:
            try:
                scheduler.step(avg_val_loss)  # e.g., ReduceLROnPlateau
            except TypeError:
                scheduler.step()
            try:
                print(f'Current learning rate: {scheduler.get_last_lr()[0]}')
            except Exception:
                pass

        # ---- Checkpoint best ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f'Validation loss decreased; saving model to: {checkpoint_path}')
            torch.save(model_classifier.state_dict(), checkpoint_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f'No improvement for {early_stopping_counter} epoch(s).')

        # ---- Early stopping ----
        if early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping after {early_stopping_patience} epochs without improvement.')
            if os.path.isfile(checkpoint_path):
                model_classifier.load_state_dict(torch.load(checkpoint_path, map_location=device))
                print(f'Model restored from best checkpoint: {checkpoint_path}')
            else:
                print('Warning: best checkpoint file not found; keeping current weights.')
            break

        print()

    return model_classifier, training_losses, validation_losses


def evaluate_classifier(model_classifier, test_loader, device, label_encoder):
    model_classifier.eval()
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for (X, y) in test_loader:
            X = X.to(device)
            y = y.to(device)

            y_hat_test = model_classifier(X)
            _, predicted = torch.max(y_hat_test.data, 1)
            y_scores.extend(y_hat_test.cpu().numpy())

            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculating additional metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    conf_mat = confusion_matrix(y_true, y_pred)

    # Generate classification report
    class_report = classification_report(
        y_true, y_pred,
        target_names=[str(class_name) for class_name in label_encoder.classes_],
        output_dict=True
    )

    # Return metrics and ROC curve data
    return acc, prec, rec, f1, conf_mat


def evaluate_autoencoder_classifier(model_autoencoder, test_loader, device, label_encoder, model_name, output_dir=output_base_dir, use_train_mode=False):
    """
    Evaluate the classifier head inside a multitask autoencoder.

    Args:
        model_autoencoder: Autoencoder model with classifier head that outputs (reconstruction, latent, class_logits)
        test_loader: DataLoader with (features, labels)
        device: Device to run evaluation on
        label_encoder: Label encoder for class names
        model_name: Name for saving evaluation report
        output_dir: Directory to save outputs
        use_train_mode: If True, use train mode (for evaluating on training set with dropout/BN in training mode)

    Returns:
        Tuple of (accuracy, precision, recall, f1, confusion_matrix)
    """
    if use_train_mode:
        model_autoencoder.train()
    else:
        model_autoencoder.eval()
    model_autoencoder.to(device)
    y_true = []
    y_pred = []
    y_scores = []

    batch_idx = 0
    with torch.no_grad():
        for (X, y) in test_loader:
            X = X.to(device)
            y = y.to(device)

            # Autoencoder returns (reconstruction, latent, class_logits)
            # We need the class_logits (3rd element)
            # Note: Model's encode() handles both 2D [batch, features] and 3D [batch, 1, features] input
            output = model_autoencoder(X)

            # Extract class logits from tuple output
            if isinstance(output, tuple) and len(output) == 3:
                class_logits = output[2]
            else:
                raise ValueError(f"Expected autoencoder to return 3-tuple (reconstruction, latent, class_logits), got {type(output)}")

            _, predicted = torch.max(class_logits.data, 1)
            y_scores.extend(class_logits.cpu().numpy())

            # DEBUG: Print first batch details
            if batch_idx == 0:
                print(f"\n[DEBUG] First batch in {model_name}:")
                print(f"  Input shape: {X.shape}")
                print(f"  True labels: {y[:10].cpu().numpy()}")
                print(f"  Predictions: {predicted[:10].cpu().numpy()}")
                print(f"  Logits (first sample): {class_logits[0].cpu().numpy()}")
                print(f"  Accuracy (this batch): {(predicted == y).float().mean().item():.4f}")

            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            batch_idx += 1

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculating additional metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    conf_mat = confusion_matrix(y_true, y_pred)

    # Generate classification report
    class_report = classification_report(
        y_true, y_pred,
        target_names=[str(class_name) for class_name in label_encoder.classes_],
        output_dict=True
    )
    # Save as JSON
    json_filename = f"{output_dir}/{model_name}.json"
    with open(json_filename, 'w') as f:
        json.dump(class_report, f, indent=2)

    print(f"Classification report saved to {json_filename}")

    # Return metrics
    return acc, prec, rec, f1, conf_mat