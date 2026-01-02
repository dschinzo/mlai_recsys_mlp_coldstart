"""
Matrix Factorization model using PyTorch
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class MFModel(nn.Module):
    """Matrix Factorization with bias"""
    
    def __init__(self, n_users, n_items, embedding_dim=32):
        super(MFModel, self).__init__()
        
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
    
    def forward(self, user_idx, item_idx):
        user_emb = self.user_embeddings(user_idx)
        item_emb = self.item_embeddings(item_idx)
        
        # Dot product
        dot = (user_emb * item_emb).sum(dim=1)
        
        # Add biases
        user_b = self.user_bias(user_idx).squeeze()
        item_b = self.item_bias(item_idx).squeeze()
        
        prediction = dot + user_b + item_b + self.global_bias
        
        return torch.sigmoid(prediction)


class MatrixFactorization:
    """Matrix Factorization recommender"""
    
    def __init__(self, n_users, n_items, embedding_dim=32, lr=0.001, 
                 weight_decay=1e-5, device='cpu'):
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.device = device
        self.name = "MatrixFactorization"
        
        self.model = MFModel(n_users, n_items, embedding_dim).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.criterion = nn.BCELoss()
    
    def fit(self, train_df, val_df=None, epochs=50, batch_size=256, **kwargs):
        """Train the model"""
        
        # Prepare data
        users = torch.LongTensor(train_df['user_idx'].values).to(self.device)
        items = torch.LongTensor(train_df['item_idx'].values).to(self.device)
        ratings = torch.FloatTensor((train_df['rating'].values - 1) / 4.0).to(self.device)
        
        n_samples = len(users)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            # Shuffle data
            indices = torch.randperm(n_samples)
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_users = users[batch_indices]
                batch_items = items[batch_indices]
                batch_ratings = ratings[batch_indices]
                
                self.optimizer.zero_grad()
                predictions = self.model(batch_users, batch_items)
                loss = self.criterion(predictions, batch_ratings)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / n_batches
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    def predict(self, user_idx, item_idx):
        """Predict rating for a single user-item pair"""
        self.model.eval()
        with torch.no_grad():
            user = torch.LongTensor([user_idx]).to(self.device)
            item = torch.LongTensor([item_idx]).to(self.device)
            prediction = self.model(user, item)
        return prediction.item()
    
    def predict_batch(self, user_indices, item_indices):
        """Batch prediction"""
        self.model.eval()
        with torch.no_grad():
            users = torch.LongTensor(user_indices).to(self.device)
            items = torch.LongTensor(item_indices).to(self.device)
            predictions = self.model(users, items)
        return predictions.cpu().numpy()
    
    def recommend(self, user_idx, n_items=10, exclude_items=None):
        """Get top-N recommendations for a user"""
        if exclude_items is None:
            exclude_items = set()
        
        self.model.eval()
        with torch.no_grad():
            user = torch.LongTensor([user_idx] * self.n_items).to(self.device)
            items = torch.LongTensor(range(self.n_items)).to(self.device)
            scores = self.model(user, items).cpu().numpy()
        
        # Exclude already seen items
        for item in exclude_items:
            scores[item] = -np.inf
        
        # Get top-N
        top_items = np.argsort(scores)[::-1][:n_items]
        
        return top_items.tolist()


if __name__ == "__main__":
    # Test Matrix Factorization
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.data_loader import load_movielens
    
    data = load_movielens()
    
    mf = MatrixFactorization(data['n_users'], data['n_items'])
    mf.fit(data['train'], epochs=20)
    
    recs = mf.recommend(0, n_items=5)
    print(f"MF recommendations for user 0: {recs}")