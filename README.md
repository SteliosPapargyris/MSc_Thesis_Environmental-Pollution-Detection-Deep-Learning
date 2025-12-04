# Environmental Pollution Detection Using Deep Learning

A deep learning pipeline for environmental pollution detection using cascaded autoencoders and classifiers to normalize spectral sensor data across temperature variations.

## Overview

This project implements a novel approach to environmental pollution detection using spectral sensor chips. The system addresses the challenge of temperature-dependent sensor readings by employing a two-stage autoencoder architecture that normalizes spectral data across different temperatures before classification.

### Key Features

- **Temperature Normalization**: First-stage autoencoders normalize spectral data from various temperatures (T=25-29°C) to a standard 27°C reference
- **Baseline Transfer Learning**: Second-stage autoencoders transfer chip-specific patterns to a baseline chip representation
- **Multi-Seed Training**: Robust evaluation across multiple random seeds for reproducible results
- **Hyperparameter Optimization**: Grid search capabilities for finding optimal training configurations
- **Comprehensive Metrics**: Detailed tracking of autoencoder reconstruction loss, classification accuracy, and per-chip performance

## Architecture

The system consists of three main components:

1. **Chip Temperature Autoencoder** (`train_chip_temperature_autoencoder.py`)
   - Normalizes spectral data across temperature variations
   - Input: Spectral data (32 peaks + 1 temperature) at T=25-29°C
   - Target: Same chip data at T=27°C

2. **Chip-to-Baseline Autoencoder** (`train_chip_to_baseline_autoencoder.py`)
   - Transfers normalized chip data to baseline chip representation
   - Enables cross-chip standardization

3. **Joint Training with Inference** (`inference_with_training.py`)
   - Combines both autoencoders with classification
   - Evaluates end-to-end pipeline performance
   - Generates comprehensive metrics and visualizations

## Project Structure

```
.
├── run_training_multiple_seeds.py     # Master script for multi-seed training
├── train_chip_temperature_autoencoder.py
├── train_chip_to_baseline_autoencoder.py
├── inference_with_training.py
├── hyperparameter_search.py           # Grid search for optimal hyperparameters
├── create_averaged_plots_from_json.py # Visualization utilities
├── utils/
│   ├── models.py                      # Neural network architectures
│   ├── data_utils.py                  # Data loading and preprocessing
│   ├── train_test_utils.py            # Training and evaluation utilities
│   ├── plot_utils.py                  # Plotting functions
│   ├── config.py                      # Configuration parameters
│   └── autoencoder_utils.py           # Autoencoder-specific utilities
├── pyproject.toml                     # Project dependencies
└── README.md                          # This file
```

## Requirements

- Python >= 3.10
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- pandas
- scipy

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Enviromental-Pollution-Detection-using-Deep-Learning_new_implementation

# Install dependencies (using poetry)
poetry install
```

## Usage

### Basic Training Pipeline

Run the complete training pipeline for multiple seeds:

```bash
python run_training_multiple_seeds.py
```

This script will:
1. Train chip temperature autoencoders for all 10 chips
2. Train chip-to-baseline autoencoders
3. Run joint training with inference
4. Generate averaged loss plots and metrics

Results are saved to:
```
hyperparameter_search_results/ae_p10_es30_clf_p5_es15_bs256/seed_<SEED>/
```

### Hyperparameter Search

To find optimal hyperparameters:

```bash
python hyperparameter_search.py
```

### Configuration

Key hyperparameters can be configured via environment variables or by editing the configuration in the training scripts:

- `SEED`: Random seed for reproducibility (default: [0, 42, 123, 777, 1234, 3407, 5555, 8888, 12345])
- `AUTOENCODER_PATIENCE`: Early stopping patience for autoencoders (default: 10)
- `AUTOENCODER_EARLY_STOPPING`: Maximum epochs for autoencoders (default: 30)
- `CLASSIFIER_PATIENCE`: Early stopping patience for classifiers (default: 5)
- `CLASSIFIER_EARLY_STOPPING`: Maximum epochs for classifiers (default: 15)
- `BATCH_SIZE`: Training batch size (default: 256)

### Custom Training

For individual component training:

```bash
# Train temperature autoencoders only
python train_chip_temperature_autoencoder.py

# Train baseline transfer autoencoders only
python train_chip_to_baseline_autoencoder.py

# Run inference and joint training
python inference_with_training.py
```

## Results

The pipeline generates:
- **Loss plots**: Training and validation loss curves with standard deviation bands
- **Confusion matrices**: Classification performance per chip
- **Metrics JSON**: Detailed performance metrics for each seed
- **Averaged plots**: Cross-chip averaged visualizations

## Pipeline Details

### Data Flow

1. Raw spectral data (32 peaks) from 10 sensor chips at various temperatures
2. First autoencoder: Temperature normalization to T=27°C
3. First classifier: Pollution classification on temperature-normalized data
4. Second autoencoder: Transfer to baseline chip representation
5. Second classifier: Final pollution classification
6. Evaluation: Comprehensive metrics and visualizations

### Training Strategy

- **Multi-task learning**: Autoencoders trained with reconstruction loss + classification loss
- **Early stopping**: Prevents overfitting with configurable patience
- **Curriculum learning**: Progressive training through the autoencoder pipeline
- **Robust evaluation**: Multiple random seeds ensure statistical reliability

## Technical Notes

- The system uses 10 sensor chips with spectral data collected at different temperatures
- Robust normalization techniques handle sensor variability
- The baseline chip serves as a reference for cross-chip standardization
- All models use PyTorch for efficient GPU training

## License

This project is part of a research initiative on environmental pollution detection.

## Authors

Stelios Papargyris - papargirisstelios3@gmail.com

## Citation

If you use this code in your research, please cite the corresponding thesis/paper:
- MSc Thesis: `msc_thesis.tex`
- Journal Paper: `journal_paper.tex`

## Acknowledgments

This work builds on advances in deep learning for environmental monitoring and transfer learning techniques for sensor data normalization.
