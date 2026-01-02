"""
Baseline recommendation models
"""

import numpy as np
from collections import defaultdict


class RandomRecommender:
    """Random recommendation baseline"""
    
    def __init__(self, n_items):
        self.n_items = n_items
        self.name = "Random"
    
    def fit(self, train_df, **kwargs):
        """No training needed for random"""
        pass
    
    def predict(self, user_idx, item_idx):
        """Random prediction between 0 and 1"""
        return np.random.random()
    
    def recommend(self, user_idx, n_items=10, exclude_items=None):
        """Random recommendations"""
        if exclude_items is None:
            exclude_items = set()
        
        available_items = [i for i in range(self.n_items) if i not in exclude_items]
        
        if len(available_items) <= n_items:
            return available_items
        
        return list(np.random.choice(available_items, n_items, replace=False))


class PopularityRecommender:
    """Popularity-based recommendation baseline"""
    
    def __init__(self, n_items):
        self.n_items = n_items
        self.item_popularity = None
        self.popular_items = None
        self.name = "Popularity"
    
    def fit(self, train_df, **kwargs):
        """Calculate item popularity from training data"""
        # Count interactions per item
        item_counts = train_df['item_idx'].value_counts()
        
        # Normalize to 0-1
        max_count = item_counts.max()
        self.item_popularity = (item_counts / max_count).to_dict()
        
        # Sort items by popularity
        self.popular_items = item_counts.index.tolist()
        
        print(f"Popularity model fitted. Most popular item: {self.popular_items[0]}")
    
    def predict(self, user_idx, item_idx):
        """Predict based on item popularity"""
        return self.item_popularity.get(item_idx, 0.0)
    
    def recommend(self, user_idx, n_items=10, exclude_items=None):
        """Recommend most popular items"""
        if exclude_items is None:
            exclude_items = set()
        
        recommendations = []
        for item in self.popular_items:
            if item not in exclude_items:
                recommendations.append(item)
                if len(recommendations) >= n_items:
                    break
        
        return recommendations
    
    def predict_batch(self, user_indices, item_indices):
        """Batch prediction"""
        predictions = []
        for item_idx in item_indices:
            predictions.append(self.item_popularity.get(item_idx, 0.0))
        return np.array(predictions)


if __name__ == "__main__":
    # Test baselines
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.data_loader import load_movielens
    
    data = load_movielens()
    
    # Test Random
    random_rec = RandomRecommender(data['n_items'])
    recs = random_rec.recommend(0, n_items=5)
    print(f"Random recommendations: {recs}")
    
    # Test Popularity
    pop_rec = PopularityRecommender(data['n_items'])
    pop_rec.fit(data['train'])
    recs = pop_rec.recommend(0, n_items=5)
    print(f"Popularity recommendations: {recs}")