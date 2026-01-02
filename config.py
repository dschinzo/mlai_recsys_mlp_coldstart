"""
Configuration file for the Neural Network Recommendation System
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'experiments', 'results')

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Dataset Configuration
DATASET_CONFIG = {
    'name': 'movielens-100k',
    'url': 'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'min_ratings_per_user': 5,
    'min_ratings_per_item': 5,
    'test_size': 0.15,
    'val_size': 0.15,
    'random_state': 42
}

# Model Configuration
MODEL_CONFIG = {
    'embedding_dim': 32,
    'hidden_layers': [64, 32],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 256,
    'epochs': 100,
    'early_stopping_patience': 10,
    'weight_decay': 1e-5
}

# Evaluation Configuration
EVAL_CONFIG = {
    'k_values': [5, 10, 20],
    'primary_k': 10,
    'metrics': ['precision', 'recall', 'ndcg', 'hit_rate', 'rmse']
}

# Cold Start Configuration
COLD_START_CONFIG = {
    'cold_user_threshold': 5,  # Users with fewer ratings are "cold"
    'cold_item_threshold': 5
}

# Device Configuration
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {DEVICE}")