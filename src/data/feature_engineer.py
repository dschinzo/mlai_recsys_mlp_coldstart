"""
Feature engineering for recommendation system
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.sparse import csr_matrix
import torch


class FeatureEngineer:
    """Feature engineering for users and items"""
    
    def __init__(self, n_users, n_items, embedding_dim=32):
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.user_scaler = StandardScaler()
        self.item_scaler = StandardScaler()
        
    def create_user_features(self, users_df):
        """Create user feature matrix"""
        
        # Demographic features
        demographic_features = []
        
        for _, user in users_df.iterrows():
            features = [
                user['age_normalized'],
                user['gender'],
                user['age_group'] / 4.0,  # Normalize age group
            ]
            demographic_features.append(features)
        
        demographic_matrix = np.array(demographic_features)
        
        # One-hot encode occupation
        occupation_onehot = pd.get_dummies(
            users_df['occupation_encoded'], 
            prefix='occ'
        ).values
        
        # Combine features
        user_features = np.hstack([demographic_matrix, occupation_onehot])
        
        return user_features.astype(np.float32)
    
    def create_item_features(self, items_df, genre_names):
        """Create item feature matrix"""
        
        # Genre features (already binary)
        genre_features = items_df[genre_names].values
        
        # Normalize number of genres
        num_genres = items_df['num_genres'].values.reshape(-1, 1) / len(genre_names)
        
        # Combine features
        item_features = np.hstack([genre_features, num_genres])
        
        return item_features.astype(np.float32)
    
    def create_interaction_features(self, train_df, n_users, n_items):
        """Create sparse interaction matrix"""
        
        row = train_df['user_idx'].values
        col = train_df['item_idx'].values
        data = train_df['rating'].values
        
        interaction_matrix = csr_matrix(
            (data, (row, col)), 
            shape=(n_users, n_items)
        )
        
        return interaction_matrix
    
    def get_user_item_pairs(self, df):
        """Get user-item pairs with labels for training"""
        
        users = df['user_idx'].values
        items = df['item_idx'].values
        ratings = df['rating'].values
        
        # Normalize ratings to 0-1
        labels = (ratings - 1) / 4.0
        
        return users, items, labels
    
    def create_negative_samples(self, train_df, n_users, n_items, neg_ratio=4):
        """Create negative samples for training"""
        
        # Get positive interactions
        positive_set = set(zip(train_df['user_idx'], train_df['item_idx']))
        
        negative_samples = []
        
        for user_idx in train_df['user_idx'].unique():
            user_positives = train_df[train_df['user_idx'] == user_idx]['item_idx'].values
            n_neg = len(user_positives) * neg_ratio
            
            neg_items = []
            while len(neg_items) < n_neg:
                neg_item = np.random.randint(0, n_items)
                if (user_idx, neg_item) not in positive_set:
                    neg_items.append(neg_item)
            
            for neg_item in neg_items:
                negative_samples.append({
                    'user_idx': user_idx,
                    'item_idx': neg_item,
                    'rating': 0
                })
        
        neg_df = pd.DataFrame(negative_samples)
        
        return neg_df
    
    def prepare_batch(self, users, items, user_features, item_features, labels=None):
        """Prepare a batch for model input"""
        
        user_feat = user_features[users]
        item_feat = item_features[items]
        
        batch = {
            'user_idx': torch.LongTensor(users),
            'item_idx': torch.LongTensor(items),
            'user_features': torch.FloatTensor(user_feat),
            'item_features': torch.FloatTensor(item_feat)
        }
        
        if labels is not None:
            batch['labels'] = torch.FloatTensor(labels)
        
        return batch


if __name__ == "__main__":
    # Test feature engineering
    from data_loader import load_movielens
    
    data = load_movielens()
    
    fe = FeatureEngineer(data['n_users'], data['n_items'])
    
    user_features = fe.create_user_features(data['users'])
    item_features = fe.create_item_features(data['items'], data['genre_names'])
    
    print(f"User features shape: {user_features.shape}")
    print(f"Item features shape: {item_features.shape}")