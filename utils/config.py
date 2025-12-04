import matplotlib
import random
import torch
import numpy as np
import warnings
import os
from utils.path_utils import get_paths_for_normalization

# Allow environment variable overrides for parallel execution
total_num_chips = int(os.environ.get('TOTAL_NUM_CHIPS', 10))  # Change this to use different number of chips
CURRENT_NORMALIZATION = os.environ.get('CURRENT_NORMALIZATION', 'class_based_robust')  # Change this to use different methods
extended = True  # Set to True to use extended datasets (e.g., 10chips_extended, baseline_chip_extended.csv)
base_path, stats_path = get_paths_for_normalization(CURRENT_NORMALIZATION)

# Recommended settings for autoencoder (with environment variable override support)
autoencoder_patience = int(os.environ.get('AUTOENCODER_PATIENCE', 10))         # Learning rate patience
autoencoder_early_stopping = int(os.environ.get('AUTOENCODER_EARLY_STOPPING', 30))   # Early stopping patience

# For classifier (can keep more aggressive)
classifier_patience = int(os.environ.get('CLASSIFIER_PATIENCE', 5))
classifier_early_stopping = int(os.environ.get('CLASSIFIER_EARLY_STOPPING', 15))

# Hyperparameters
seed = int(os.environ.get('SEED', 42))
batch_size = int(os.environ.get('BATCH_SIZE', 256))
# batch_size = int(os.environ.get('BATCH_SIZE', 1))
learning_rate = 1e-3
num_epochs = 500

num_chips = list(range(1, total_num_chips+1))
chip_column = "Chip"
class_column = "Class"
target_class = 4
current_path = f"{base_path}"
matplotlib.use('Agg')  # Use a non-interactive backend
torch.manual_seed(seed), torch.cuda.manual_seed_all(seed), np.random.seed(seed), random.seed(seed)
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
warnings.filterwarnings("ignore")

# Normalization configuration
NORMALIZATION_CONFIG = {
    'class_based_mean_std': {
        'name': 'class_based_mean_std_normalized',
        'folder': 'mean_std',
        'description': 'Class-4 Mean/Std Normalization'
    },
    'class_based_minmax': {
        'name': 'minmax_normalized',
        'folder': 'minmax',
        'description': 'Class-4 Min-Max Normalization'
    },
    'class_based_robust': {
        'name': 'robust_normalized',
        'folder': 'robust',
        'description': 'Class-4 Robust Scaling Normalization'
    },
    'none': {
        'name': 'raw',
        'folder': 'raw',
        'description': 'No Normalization'
    }
}

# Current normalization configuration - available everywhere
norm_config = NORMALIZATION_CONFIG[CURRENT_NORMALIZATION]
norm_name = norm_config['name']
norm_folder = norm_config['folder']
norm_description = norm_config['description']

# Dynamic output directory structure
# Path adjusted to work from subfolder (1stAutoencoder_toBaselineChipT25_29_SecondAutoencoder_toBaselineChipT27/)
# output_base_dir = f'1stAutoencoder_toBaselineChipT25_29_SecondAutoencoder_toBaselineChipT27/out/{total_num_chips}chips/{norm_name}'
# Support environment variable override for hyperparameter search
output_base_dir = os.environ.get('OUTPUT_BASE_DIR', f'out/{total_num_chips}chips/{norm_name}')

# Data paths - single source of truth for file locations
def get_merged_transfer_data_path():
    """Get the path to the merged transfer autoencoder outputs (for classifier training)"""
    return f"{output_base_dir}/merged_transfer_autoencoder_outputs_{norm_name}_to_baseline.csv"

def get_baseline_normalized_path():
    """Get the path to the baseline normalized chip data (with StandardScaler applied)"""
    return f"data/out/{norm_folder}/baseline/with_standard_scaler/baseline_chip_{norm_folder}.csv"

def print_current_config():
    """Print current configuration info"""
    print(f"Current config: {total_num_chips} chips, normalization: {norm_description}")
    print(f"Output directory: {output_base_dir}")

# # Print configuration on import
# print_current_config()