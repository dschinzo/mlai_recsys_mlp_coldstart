"""
Data loading and preprocessing for MovieLens dataset
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASET_CONFIG


class DataLoader:
    """Load and preprocess MovieLens 100K dataset"""
    
    def __init__(self):
        self.data_path = os.path.join(RAW_DATA_DIR, 'ml-100k')
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.occupation_encoder = LabelEncoder()
        self.age_scaler = MinMaxScaler()
        
    def load_ratings(self):
        """Load ratings data"""
        ratings_file = os.path.join(self.data_path, 'u.data')
        ratings = pd.read_csv(
            ratings_file,
            sep='\t',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            engine='python'
        )
        return ratings
    
    def load_users(self):
        """Load user demographic data"""
        users_file = os.path.join(self.data_path, 'u.user')
        users = pd.read_csv(
            users_file,
            sep='|',
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
            engine='python'
        )
        return users
    
    def load_items(self):
        """Load item (movie) data"""
        items_file = os.path.join(self.data_path, 'u.item')
        genre_names = [
            'unknown', 'Action', 'Adventure', 'Animation', 'Children',
            'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
            'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
            'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
        
        columns = ['item_id', 'title', 'release_date', 'video_release_date',
                   'imdb_url'] + genre_names
        
        items = pd.read_csv(
            items_file,
            sep='|',
            names=columns,
            encoding='latin-1',
            engine='python'
        )
        return items, genre_names
    
    def preprocess_users(self, users):
        """Preprocess user features"""
        users_processed = users.copy()
        
        # Encode gender (M=1, F=0)
        users_processed['gender'] = (users_processed['gender'] == 'M').astype(int)
        
        # Normalize age
        users_processed['age_normalized'] = self.age_scaler.fit_transform(
            users_processed[['age']]
        )
        
        # Encode occupation
        users_processed['occupation_encoded'] = self.occupation_encoder.fit_transform(
            users_processed['occupation']
        )
        
        # Create age groups
        users_processed['age_group'] = pd.cut(
            users_processed['age'],
            bins=[0, 18, 25, 35, 50, 100],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)
        
        return users_processed
    
    def preprocess_items(self, items, genre_names):
        """Preprocess item features"""
        items_processed = items.copy()
        
        # Genre columns are already binary (0/1)
        # Create genre vector
        items_processed['genre_vector'] = items_processed[genre_names].values.tolist()
        
        # Count number of genres per movie
        items_processed['num_genres'] = items_processed[genre_names].sum(axis=1)
        
        return items_processed, genre_names
    
    def create_interaction_matrix(self, ratings, n_users, n_items):
        """Create user-item interaction matrix"""
        matrix = np.zeros((n_users, n_items))
        for row in ratings.itertuples():
            matrix[row.user_idx, row.item_idx] = row.rating
        return matrix
    
    def split_data(self, ratings):
        """Split data into train, validation, and test sets"""
        # Sort by timestamp for temporal split
        ratings_sorted = ratings.sort_values('timestamp')
        
        # Split: 70% train, 15% validation, 15% test
        train_val, test = train_test_split(
            ratings_sorted,
            test_size=DATASET_CONFIG['test_size'],
            random_state=DATASET_CONFIG['random_state']
        )
        
        train, val = train_test_split(
            train_val,
            test_size=DATASET_CONFIG['val_size'] / (1 - DATASET_CONFIG['test_size']),
            random_state=DATASET_CONFIG['random_state']
        )
        
        return train, val, test
    
    def identify_cold_users(self, train, test, threshold=5):
        """Identify cold start users in test set"""
        train_user_counts = train['user_idx'].value_counts()
        
        cold_users = []
        for user_idx in test['user_idx'].unique():
            if user_idx not in train_user_counts or train_user_counts[user_idx] < threshold:
                cold_users.append(user_idx)
        
        return cold_users
    
    def load_and_preprocess(self):
        """Main function to load and preprocess all data"""
        print("Loading MovieLens 100K dataset...")
        
        # Load raw data
        ratings = self.load_ratings()
        users = self.load_users()
        items, genre_names = self.load_items()
        
        print(f"Loaded {len(ratings)} ratings, {len(users)} users, {len(items)} items")
        
        # Encode user and item IDs
        ratings['user_idx'] = self.user_encoder.fit_transform(ratings['user_id'])
        ratings['item_idx'] = self.item_encoder.fit_transform(ratings['item_id'])
        
        n_users = ratings['user_idx'].nunique()
        n_items = ratings['item_idx'].nunique()
        
        print(f"Unique users: {n_users}, Unique items: {n_items}")
        
        # Preprocess features
        users_processed = self.preprocess_users(users)
        items_processed, genre_names = self.preprocess_items(items, genre_names)
        
        # Merge user features with ratings
        ratings = ratings.merge(
            users_processed[['user_id', 'age_normalized', 'gender', 
                            'occupation_encoded', 'age_group']],
            on='user_id'
        )
        
        # Split data
        train, val, test = self.split_data(ratings)
        
        print(f"Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")
        
        # Identify cold users
        cold_users = self.identify_cold_users(train, test)
        print(f"Cold start users in test set: {len(cold_users)}")
        
        return {
            'train': train,
            'val': val,
            'test': test,
            'users': users_processed,
            'items': items_processed,
            'genre_names': genre_names,
            'n_users': n_users,
            'n_items': n_items,
            'cold_users': cold_users,
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder
        }


def load_movielens():
    """Convenience function to load MovieLens data"""
    loader = DataLoader()
    return loader.load_and_preprocess()


if __name__ == "__main__":
    # Test data loading
    data = load_movielens()
    print("\nData loading complete!")
    print(f"Keys: {data.keys()}")