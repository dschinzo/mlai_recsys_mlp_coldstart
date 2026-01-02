"""
Hybrid Neural Network for Cold-Start Recommendation
This is our proposed model that combines demographic and content features
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class HybridNNModel(nn.Module):
    """
    Hybrid Neural Network combining:
    - User demographic features
    - Item content features  
    - Collaborative embeddings
    """
    
    def __init__(self, n_users, n_items, user_feature_dim, item_feature_dim,
                 embedding_dim=32, hidden_layers=[64, 32], dropout_rate=0.3):
        super(HybridNNModel, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        
        # Collaborative embeddings (for warm users)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Feature transformation layers
        self.user_feature_layer = nn.Sequential(
            nn.Linear(user_feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.item_feature_layer = nn.Sequential(
            nn.Linear(item_feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Combined input dimension: 
        # user_emb + user_feat + item_emb + item_feat = 4 * embedding_dim
        mlp_input_dim = embedding_dim * 4
        
        # MLP layers
        self.mlp_layers = nn.ModuleList()
        for hidden_dim in hidden_layers:
            self.mlp_layers.append(nn.Linear(mlp_input_dim, hidden_dim))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            self.mlp_layers.append(nn.Dropout(dropout_rate))
            mlp_input_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(hidden_layers[-1], 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, user_idx, item_idx, user_features, item_features):
        # Collaborative embeddings
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        
        # Feature embeddings
        user_feat_emb = self.user_feature_layer(user_features)
        item_feat_emb = self.item_feature_layer(item_features)
        
        # Concatenate all representations
        combined = torch.cat([
            user_emb, user_feat_emb, 
            item_emb, item_feat_emb
        ], dim=1)
        
        # MLP forward pass
        mlp_output = combined
        for layer in self.mlp_layers:
            mlp_output = layer(mlp_output)
        
        # Output
        output = self.output_layer(mlp_output)
        
        return torch.sigmoid(output).squeeze()
    
    def forward_cold_start(self, user_features, item_idx, item_features):
        """
        Forward pass for cold-start users (no collaborative embedding)
        Uses only demographic and content features
        """
        batch_size = user_features.size(0)
        
        # Use mean embedding for cold users
        user_emb = self.user_embedding.weight.mean(dim=0).unsqueeze(0).expand(batch_size, -1)
        item_emb = self.item_embedding(item_idx)
        
        # Feature embeddings
        user_feat_emb = self.user_feature_layer(user_features)
        item_feat_emb = self.item_feature_layer(item_features)
        
        # Concatenate
        combined = torch.cat([
            user_emb, user_feat_emb,
            item_emb, item_feat_emb
        ], dim=1)
        
        # MLP forward
        mlp_output = combined
        for layer in self.mlp_layers:
            mlp_output = layer(mlp_output)
        
        output = self.output_layer(mlp_output)
        
        return torch.sigmoid(output).squeeze()


class HybridNN:
    """Hybrid Neural Network recommender for cold-start problem"""
    
    def __init__(self, n_users, n_items, user_feature_dim, item_feature_dim,
                 embedding_dim=32, hidden_layers=[64, 32], dropout_rate=0.3,
                 lr=0.001, weight_decay=1e-5, device='cpu'):
        
        self.n_users = n_users
        self.n_items = n_items
        self.user_feature_dim = user_feature_dim
        self.item_feature_dim = item_feature_dim
        self.device = device
        self.name = "HybridNN"
        
        self.model = HybridNNModel(
            n_users, n_items, user_feature_dim, item_feature_dim,
            embedding_dim, hidden_layers, dropout_rate
        ).to(device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = nn.BCELoss()
        
        # Store features for recommendation
        self.user_features = None
        self.item_features = None
    
    def fit(self, train_df, user_features, item_features, val_df=None,
            epochs=50, batch_size=256, early_stopping_patience=10, **kwargs):
        """Train the model"""
        
        # Store features
        self.user_features = torch.FloatTensor(user_features).to(self.device)
        self.item_features = torch.FloatTensor(item_features).to(self.device)
        
        # Prepare training data
        users = train_df['user_idx'].values
        items = train_df['item_idx'].values
        ratings = (train_df['rating'].values - 1) / 4.0
        
        n_samples = len(users)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        best_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        print(f"Training {self.name} model...")
        print(f"User features dim: {user_features.shape[1]}, Item features dim: {item_features.shape[1]}")
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            # Shuffle
            indices = np.random.permutation(n_samples)
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_users = torch.LongTensor(users[batch_indices]).to(self.device)
                batch_items = torch.LongTensor(items[batch_indices]).to(self.device)
                batch_ratings = torch.FloatTensor(ratings[batch_indices]).to(self.device)
                
                batch_user_features = self.user_features[batch_users]
                batch_item_features = self.item_features[batch_items]
                
                self.optimizer.zero_grad()
                predictions = self.model(
                    batch_users, batch_items,
                    batch_user_features, batch_item_features
                )
                loss = self.criterion(predictions, batch_ratings)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / n_batches
            training_history.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model
                self.best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    self.model.load_state_dict(self.best_state)
                    break
        
        return training_history
    
    def predict(self, user_idx, item_idx, is_cold_user=False):
        """Predict for a single user-item pair"""
        self.model.eval()
        with torch.no_grad():
            user = torch.LongTensor([user_idx]).to(self.device)
            item = torch.LongTensor([item_idx]).to(self.device)
            user_feat = self.user_features[user_idx:user_idx+1]
            item_feat = self.item_features[item_idx:item_idx+1]
            
            if is_cold_user:
                prediction = self.model.forward_cold_start(user_feat, item, item_feat)
            else:
                prediction = self.model(user, item, user_feat, item_feat)
        
        return prediction.item()
    
    def predict_batch(self, user_indices, item_indices, is_cold_user=False):
        """Batch prediction"""
        self.model.eval()
        with torch.no_grad():
            users = torch.LongTensor(user_indices).to(self.device)
            items = torch.LongTensor(item_indices).to(self.device)
            user_feat = self.user_features[users]
            item_feat = self.item_features[items]
            
            if is_cold_user:
                predictions = self.model.forward_cold_start(user_feat, items, item_feat)
            else:
                predictions = self.model(users, items, user_feat, item_feat)
        
        return predictions.cpu().numpy()
    
    def recommend(self, user_idx, n_items=10, exclude_items=None, is_cold_user=False):
        """Get top-N recommendations for a user"""
        if exclude_items is None:
            exclude_items = set()
        
        self.model.eval()
        with torch.no_grad():
            if is_cold_user:
                # For cold users, use demographic features only
                user_feat = self.user_features[user_idx:user_idx+1].expand(self.n_items, -1)
                items = torch.LongTensor(range(self.n_items)).to(self.device)
                item_feat = self.item_features
                scores = self.model.forward_cold_start(user_feat, items, item_feat)
            else:
                users = torch.LongTensor([user_idx] * self.n_items).to(self.device)
                items = torch.LongTensor(range(self.n_items)).to(self.device)
                user_feat = self.user_features[user_idx:user_idx+1].expand(self.n_items, -1)
                item_feat = self.item_features
                scores = self.model(users, items, user_feat, item_feat)
            
            scores = scores.cpu().numpy()
        
        # Exclude seen items
        for item in exclude_items:
            if item < len(scores):
                scores[item] = -np.inf
        
        top_items = np.argsort(scores)[::-1][:n_items]
        
        return top_items.tolist()
    
    def recommend_for_cold_user(self, user_features_vector, n_items=10):
        """
        Recommend items for a completely new user given only demographic features
        """
        self.model.eval()
        with torch.no_grad():
            user_feat = torch.FloatTensor(user_features_vector).unsqueeze(0).to(self.device)
            user_feat = user_feat.expand(self.n_items, -1)
            items = torch.LongTensor(range(self.n_items)).to(self.device)
            item_feat = self.item_features
            
            scores = self.model.forward_cold_start(user_feat, items, item_feat)
            scores = scores.cpu().numpy()
        
        top_items = np.argsort(scores)[::-1][:n_items]
        
        return top_items.tolist()


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.data_loader import load_movielens
    from data.feature_engineer import FeatureEngineer
    
    # Load data
    data = load_movielens()
    
    # Create features
    fe = FeatureEngineer(data['n_users'], data['n_items'])
    user_features = fe.create_user_features(data['users'])
    item_features = fe.create_item_features(data['items'], data['genre_names'])
    
    print(f"User features shape: {user_features.shape}")
    print(f"Item features shape: {item_features.shape}")
    
    # Train model
    hybrid = HybridNN(
        data['n_users'], 
        data['n_items'],
        user_features.shape[1],
        item_features.shape[1]
    )
    
    hybrid.fit(data['train'], user_features, item_features, epochs=20)
    
    # Test recommendations
    recs = hybrid.recommend(0, n_items=5)
    print(f"HybridNN recommendations for user 0: {recs}")
    
    # Test cold start recommendations
    cold_recs = hybrid.recommend(0, n_items=5, is_cold_user=True)
    print(f"HybridNN cold-start recommendations: {cold_recs}")