"""
Helper utility functions
"""

import os
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_results(results, filepath):
    """Save experiment results to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    
    results_converted = convert(results)
    results_converted['timestamp'] = datetime.now().isoformat()
    
    with open(filepath, 'w') as f:
        json.dump(results_converted, f, indent=2)
    
    print(f"Results saved to {filepath}")


def load_results(filepath):
    """Load experiment results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_results(results, save_path=None):
    """Plot comparison of model results"""
    
    models = list(results.keys())
    metrics = ['precision@10', 'recall@10', 'ndcg@10', 'hit_rate@10', 'rmse']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        values = []
        for model in models:
            if metric in results[model]:
                values.append(results[model][metric])
            else:
                values.append(0)
        
        ax = axes[idx]
        bars = ax.bar(models, values, color=sns.color_palette("husl", len(models)))
        ax.set_title(metric.upper(), fontsize=12, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xticklabels(models, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Remove empty subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_cold_start_comparison(results, save_path=None):
    """Plot cold-start vs warm user performance comparison"""
    
    models = list(results.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Precision comparison
    warm_precision = [results[m].get('precision@10', 0) for m in models]
    cold_precision = [results[m].get('cold_precision@10', 0) for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[0].bar(x - width/2, warm_precision, width, label='Warm Users', color='steelblue')
    axes[0].bar(x + width/2, cold_precision, width, label='Cold Users', color='coral')
    axes[0].set_ylabel('Precision@10')
    axes[0].set_title('Precision@10: Warm vs Cold Users')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].legend()
    
    # Hit Rate comparison
    warm_hr = [results[m].get('hit_rate@10', 0) for m in models]
    cold_hr = [results[m].get('cold_hit_rate@10', 0) for m in models]
    
    axes[1].bar(x - width/2, warm_hr, width, label='Warm Users', color='steelblue')
    axes[1].bar(x + width/2, cold_hr, width, label='Cold Users', color='coral')
    axes[1].set_ylabel('Hit Rate@10')
    axes[1].set_title('Hit Rate@10: Warm vs Cold Users')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def print_results_table(results):
    """Print results in a formatted table"""
    
    models = list(results.keys())
    
    # Header
    print("\n" + "="*80)
    print(f"{'Model':<20} {'Prec@10':>10} {'Recall@10':>10} {'NDCG@10':>10} {'HR@10':>10} {'RMSE':>10}")
    print("="*80)
    
    for model in models:
        r = results[model]
        print(f"{model:<20} {r.get('precision@10', 0):>10.4f} {r.get('recall@10', 0):>10.4f} "
              f"{r.get('ndcg@10', 0):>10.4f} {r.get('hit_rate@10', 0):>10.4f} "
              f"{r.get('rmse', 0):>10.4f}")
    
    print("="*80)
    
    # Cold start results
    print("\nCold-Start Performance:")
    print("-"*80)
    print(f"{'Model':<20} {'Cold Prec@10':>12} {'Cold HR@10':>12} {'Cold RMSE':>12}")
    print("-"*80)
    
    for model in models:
        r = results[model]
        print(f"{model:<20} {r.get('cold_precision@10', 0):>12.4f} "
              f"{r.get('cold_hit_rate@10', 0):>12.4f} "
              f"{r.get('cold_rmse', 0):>12.4f}")
    
    print("-"*80 + "\n")


if __name__ == "__main__":
    # Test utilities
    set_seed(42)
    print("Random test:", np.random.rand(3))
    
    # Test results
    test_results = {
        'Random': {'precision@10': 0.05, 'recall@10': 0.03, 'ndcg@10': 0.04, 'hit_rate@10': 0.20, 'rmse': 1.50},
        'Popularity': {'precision@10': 0.15, 'recall@10': 0.10, 'ndcg@10': 0.12, 'hit_rate@10': 0.50, 'rmse': 1.20}
    }
    
    print_results_table(test_results)