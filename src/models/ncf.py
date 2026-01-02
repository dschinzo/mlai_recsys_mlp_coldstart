"""
Neural Collaborative Filtering (NCF) model
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class NCFModel(nn.Module):
    """Neural Collaborative Filtering"""
    
    def __init__(self, n_users, n_items, embedding_dim=32, hidden_layers=[64, 32], 
                 dropout_rate=0.3):
        super(NCFModel, self).__init__()
        
        # GMF embeddings
        self.user_embedding_gmf = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(n_items, embedding_dim)
        
        # MLP embeddings
        self.user_embedding_mlp = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(n_items, embedding_dim)
        
        # MLP layers
        mlp_input_dim = embedding_dim * 2
        self.mlp_layers = nn.ModuleList()
        
        for hidden_dim in hidden_layers:
            self.mlp_layers.append(nn.Linear(mlp_input_dim, hidden_dim))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout_rate))
            mlp_input_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(embedding_dim + hidden_layers[-1], 1)
        
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
    
    def forward(self, user_idx, item_idx):
        # GMF part
        user_emb_gmf = self.user_embedding_gmf(user_idx)
        item_emb_gmf = self.item_embedding_gmf(item_idx)
        gmf_output = user_emb_gmf * item_emb_gmf
        
        # MLP part
        user_emb_mlp = self.user_embedding_mlp(user_idx)
        item_emb_mlp = self.item_embedding_mlp(item_idx)
        mlp_input = torch.cat([user_emb_mlp, item_emb_mlp], dim=1)
        
        mlp_output = mlp_input
        for layer in self.mlp_layers:
            mlp_output = layer(mlp_output)
        
        # Combine GMF and MLP
        combined = torch.cat([gmf_output, mlp_output], dim=1)
        output = self.output_layer(combined)
        
        return torch.sigmoid(output).squeeze()


class NCF:
    """Neural Collaborative Filtering recommender"""
    
    def __init__(self, n_users, n_items, embedding_dim=32, hidden_layers=[64, 32],
                 dropout_rate=0.3, lr=0.001, weight_decay=1e-5, device='cpu'):
        self.n_users = n_users
        self.n_items = n_items
        self.device = device
        self.name = "NCF"
        
        self.model = NCFModel(
            n_users, n_items, embedding_dim, hidden_layers, dropout_rate
        ).to(device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = nn.BCELoss()
    
    def fit(self, train_df, val_df=None, epochs=50, batch_size=256, 
            early_stopping_patience=10, **kwargs):
        """Train the model"""
        
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
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    def predict(self, user_idx, item_idx):
        """Predict for a single pair"""
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
        """Get top-N recommendations"""
        if exclude_items is None:
            exclude_items = set()
        
        self.model.eval()
        with torch.no_grad():
            user = torch.LongTensor([user_idx] * self.n_items).to(self.device)
            items = torch.LongTensor(range(self.n_items)).to(self.device)
            scores = self.model(user, items).cpu().numpy()
        
        for item in exclude_items:
            scores[item] = -np.inf
        
        top_items = np.argsort(scores)[::-1][:n_items]
        
        return top_items.tolist()


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.data_loader import load_movielens
    
    data = load_movielens()
    
    ncf = NCF(data['n_users'], data['n_items'])
    ncf.fit(data['train'], epochs=20)
    
    recs = ncf.recommend(0, n_items=5)
    print(f"NCF recommendations for user 0: {recs}")