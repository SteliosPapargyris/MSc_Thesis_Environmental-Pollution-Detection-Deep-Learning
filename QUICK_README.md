# Quick Start Guide

## Prerequisites

Install dependencies using Poetry:
```bash
poetry install
```

## Training Pipeline

### Option 1: Single Seed Training

Run these scripts in order:

```bash
# Step 1: Normalize chip data
poetry run python normalize_chips.py

# Step 2: Prepare matched datasets for training
poetry run python data_preparation.py

# Step 3: Train first-stage temperature autoencoders (T=25-29°C → T=27°C)
poetry run python train_chip_temperature_autoencoder.py

# Step 4: Train second-stage chip-to-baseline autoencoders
poetry run python train_chip_to_baseline_autoencoder.py

# Step 5: Joint training and inference (trains autoencoders + classifier together)
poetry run python inference_with_training.py
```

### Option 2: Multiple Seeds Training

For robust evaluation across multiple random seeds:

```bash
# Step 1: Normalize chip data
poetry run python normalize_chips.py

# Step 2: Prepare matched datasets for training
poetry run python data_preparation.py

# Step 3: Run complete pipeline for all seeds [0, 42, 123, 777, 1234, 3407, 5555, 8888, 12345]
poetry run python run_training_multiple_seeds.py
```

This will automatically run steps 3-5 for each seed and generate averaged metrics.

## Configuration

Key settings in [utils/config.py](utils/config.py):

- `total_num_chips`: Number of chips to use (default: 10)
- `CURRENT_NORMALIZATION`: Normalization method (default: 'class_based_robust')
- `extended`: Use extended datasets (default: True)
- `seed`: Random seed (default: 42, overridable via environment variable)
- `batch_size`: Training batch size (default: 256)

## Output Locations

### Single Seed:
- Autoencoder outputs: `out/{total_num_chips}chips/{norm_name}/`
- Trained models and metrics: Same directory

### Multiple Seeds:
- Results: `hyperparameter_search_results/ae_p10_es30_clf_p5_es15_bs256/seed_<SEED>/`
- Averaged metrics: `hyperparameter_search_results/ae_p10_es30_clf_p5_es15_bs256/averaged_metrics_across_seeds.json`

## Pipeline Overview

1. **normalize_chips.py**: Applies class-based normalization to raw chip data
2. **data_preparation.py**: Creates matched train/target pairs (each sample paired with its 27°C reference)
3. **train_chip_temperature_autoencoder.py**: Trains autoencoders to normalize temperature variations
4. **train_chip_to_baseline_autoencoder.py**: Trains autoencoders to transfer to baseline chip representation
5. **inference_with_training.py**: Jointly trains both autoencoders + classifier, evaluates full pipeline

## Notes

- Data files are saved to `data/out/shuffled_dataset/` by data_preparation.py
- Training uses GPU (MPS on Mac, CUDA if available) automatically
- All scripts use configuration from [utils/config.py](utils/config.py)
