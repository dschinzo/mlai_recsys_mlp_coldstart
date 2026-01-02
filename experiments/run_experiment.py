"""
Main experiment runner - Compare all recommendation models
"""

import sys
sys.path.append('..')

import argparse
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch

from config import CONFIG
from src.data.data_loader import load_movielens, create_cold_start_split
from src.data.feature_engineer import FeatureEngineer
from src.models.baseline import RandomRecommender, PopularityRecommender
from src.models.matrix_factorization import MatrixFactorization
from src.models.ncf import NCF
from src.models.hybrid_nn import HybridNN
from src.evaluation.metrics import evaluate_model, statistical_test


def run_single_experiment(model, train, test, user_features, item_features, model_name):
    """Train and evaluate a single model"""
    print(f"\n{'='*50}")
    print(f"Training: {model_name}")
    print('='*50)
    
    start_time = time.time()
    
    # Train
    if model_name in ['Random', 'Popularity']:
        model.fit(train)
    elif model_name == 'MatrixFactorization':
        model.fit(train)
    else:
        model.fit(train, user_features, item_features)
    
    train_time = time.time() - start_time
    
    # Evaluate
    results = evaluate_model(model, test, k=CONFIG['top_k'])
    results['train_time'] = train_time
    results['model'] = model_name
    
    print(f"Precision@{CONFIG['top_k']}: {results['precision']:.4f}")
    print(f"Recall@{CONFIG['top_k']}: {results['recall']:.4f}")
    print(f"NDCG@{CONFIG['top_k']}: {results['ndcg']:.4f}")
    print(f"Hit Rate: {results['hit_rate']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"Training Time: {train_time:.2f}s")
    
    return results


def run_all_experiments(cold_start=True):
    """Run experiments on all models"""
    
    # Load data
    print("Loading MovieLens data...")
    train, test, user_features, item_features = load_movielens()
    
    if cold_start:
        print("Creating cold-start split...")
        train, test = create_cold_start_split(train, test, threshold=5)
    
    # Feature engineering
    print("Engineering features...")
    fe = FeatureEngineer()
    user_features = fe.process_user_features(user_features)
    item_features = fe.process_item_features(item_features)
    
    # Initialize models
    models = {
        'Random': RandomRecommender(),
        'Popularity': PopularityRecommender(),
        'MatrixFactorization': MatrixFactorization(
            n_factors=CONFIG['mf_factors'],
            lr=CONFIG['learning_rate'],
            epochs=CONFIG['epochs']
        ),
        'NCF': NCF(
            n_users=user_features.shape[0],
            n_items=item_features.shape[0],
            embed_dim=CONFIG['embedding_dim'],
            hidden_layers=CONFIG['hidden_layers']
        ),
        'HybridNN': HybridNN(
            user_dim=user_features.shape[1],
            item_dim=item_features.shape[1],
            embed_dim=CONFIG['embedding_dim'],
            hidden_layers=CONFIG['hidden_layers'],
            dropout=CONFIG['dropout_rate']
        )
    }
    
    # Run experiments
    all_results = []
    for name, model in models.items():
        results = run_single_experiment(
            model, train, test, 
            user_features, item_features, name
        )
        all_results.append(results)
    
    return pd.DataFrame(all_results)


def run_hypothesis_test(results_df):
    """Statistical test: HybridNN vs NCF"""
    print("\n" + "="*50)
    print("HYPOTHESIS TESTING")
    print("="*50)
    
    hybrid_precision = results_df[results_df['model'] == 'HybridNN']['precision'].values[0]
    ncf_precision = results_df[results_df['model'] == 'NCF']['precision'].values[0]
    
    improvement = hybrid_precision - ncf_precision
    
    print(f"\nHybridNN Precision@10: {hybrid_precision:.4f}")
    print(f"NCF Precision@10: {ncf_precision:.4f}")
    print(f"Improvement: {improvement:.4f} ({improvement/ncf_precision*100:.1f}%)")
    
    # Check hypothesis
    if improvement >= 0.03:
        print("\n✓ H1 SUPPORTED: Improvement >= 3%")
    else:
        print("\n✗ H1 NOT SUPPORTED: Improvement < 3%")
    
    return improvement


def save_results(results_df, experiment_name):
    """Save experiment results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiments/results/{experiment_name}_{timestamp}.json"
    
    results_df.to_json(filename, orient='records', indent=2)
    print(f"\nResults saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Run recommendation experiments')
    parser.add_argument('--model', type=str, default='all', 
                        choices=['all', 'random', 'popularity', 'mf', 'ncf', 'hybrid_nn'])
    parser.add_argument('--cold-start', action='store_true', default=True)
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    parser.add_argument('--save', action='store_true', default=True)
    
    args = parser.parse_args()
    
    print("="*50)
    print("NEURAL NETWORK RECOMMENDATION EXPERIMENT")
    print("="*50)
    
    if args.tune:
        from experiments.hyperparameter_tuning import run_tuning
        best_params = run_tuning()
        print(f"Best parameters: {best_params}")
    
    # Run experiments
    results_df = run_all_experiments(cold_start=args.cold_start)
    
    # Hypothesis test
    run_hypothesis_test(results_df)
    
    # Print summary
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    print(results_df.to_string(index=False))
    
    # Save
    if args.save:
        save_results(results_df, "full_experiment")


if __name__ == "__main__":
    main()