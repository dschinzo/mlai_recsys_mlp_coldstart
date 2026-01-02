"""
Evaluation metrics for recommendation systems
"""

import numpy as np
from collections import defaultdict
from tqdm import tqdm


def precision_at_k(recommended, relevant, k):
    """
    Calculate Precision@K
    
    Args:
        recommended: list of recommended item ids
        relevant: set of relevant (ground truth) item ids
        k: number of recommendations to consider
    
    Returns:
        Precision@K score
    """
    if k == 0:
        return 0.0
    
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    
    hits = len(set(recommended_k) & relevant_set)
    
    return hits / k


def recall_at_k(recommended, relevant, k):
    """
    Calculate Recall@K
    
    Args:
        recommended: list of recommended item ids
        relevant: set of relevant (ground truth) item ids
        k: number of recommendations to consider
    
    Returns:
        Recall@K score
    """
    if len(relevant) == 0:
        return 0.0
    
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    
    hits = len(set(recommended_k) & relevant_set)
    
    return hits / len(relevant_set)


def dcg_at_k(scores, k):
    """Calculate DCG@K"""
    scores = np.array(scores)[:k]
    if len(scores) == 0:
        return 0.0
    
    gains = 2 ** scores - 1
    discounts = np.log2(np.arange(len(scores)) + 2)
    
    return np.sum(gains / discounts)


def ndcg_at_k(recommended, relevant, k):
    """
    Calculate NDCG@K (Normalized Discounted Cumulative Gain)
    
    Args:
        recommended: list of recommended item ids
        relevant: set of relevant (ground truth) item ids
        k: number of recommendations to consider
    
    Returns:
        NDCG@K score
    """
    if len(relevant) == 0:
        return 0.0
    
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    
    # Create relevance scores (1 if relevant, 0 otherwise)
    relevance_scores = [1 if item in relevant_set else 0 for item in recommended_k]
    
    # Calculate DCG
    dcg = dcg_at_k(relevance_scores, k)
    
    # Calculate ideal DCG (all relevant items at the top)
    ideal_scores = [1] * min(len(relevant_set), k)
    idcg = dcg_at_k(ideal_scores, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def hit_rate_at_k(recommended, relevant, k):
    """
    Calculate Hit Rate@K (whether at least one relevant item is in top-K)
    
    Args:
        recommended: list of recommended item ids
        relevant: set of relevant (ground truth) item ids
        k: number of recommendations to consider
    
    Returns:
        1 if hit, 0 otherwise
    """
    recommended_k = set(recommended[:k])
    relevant_set = set(relevant)
    
    return 1 if len(recommended_k & relevant_set) > 0 else 0


def rmse(predictions, actuals):
    """
    Calculate Root Mean Square Error
    
    Args:
        predictions: array of predicted values
        actuals: array of actual values
    
    Returns:
        RMSE score
    """
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    return np.sqrt(np.mean((predictions - actuals) ** 2))


def mean_absolute_error(predictions, actuals):
    """Calculate Mean Absolute Error"""
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    return np.mean(np.abs(predictions - actuals))


def evaluate_model(model, test_df, train_df, k_values=[5, 10, 20], 
                   user_features=None, item_features=None):
    """
    Comprehensive model evaluation
    
    Args:
        model: trained recommendation model
        test_df: test DataFrame
        train_df: training DataFrame (to exclude seen items)
        k_values: list of K values for evaluation
        user_features: user feature matrix (for hybrid model)
        item_features: item feature matrix (for hybrid model)
    
    Returns:
        Dictionary of evaluation results
    """
    
    results = {k: {'precision': [], 'recall': [], 'ndcg': [], 'hit_rate': []} 
               for k in k_values}
    
    all_predictions = []
    all_actuals = []
    
    # Group test data by user
    test_users = test_df.groupby('user_idx')
    train_users = train_df.groupby('user_idx')
    
    # Get training items for each user
    train_items_per_user = {}
    for user_idx, group in train_users:
        train_items_per_user[user_idx] = set(group['item_idx'].values)
    
    print(f"Evaluating {model.name}...")
    
    for user_idx, group in tqdm(test_users, desc="Evaluating users"):
        # Get relevant items (items user rated highly in test set)
        relevant_items = set(group[group['rating'] >= 4]['item_idx'].values)
        
        if len(relevant_items) == 0:
            continue
        
        # Get items to exclude (already seen in training)
        exclude_items = train_items_per_user.get(user_idx, set())
        
        # Get recommendations
        max_k = max(k_values)
        try:
            recommended = model.recommend(user_idx, n_items=max_k, exclude_items=exclude_items)
        except Exception as e:
            continue
        
        # Calculate metrics for each K
        for k in k_values:
            results[k]['precision'].append(precision_at_k(recommended, relevant_items, k))
            results[k]['recall'].append(recall_at_k(recommended, relevant_items, k))
            results[k]['ndcg'].append(ndcg_at_k(recommended, relevant_items, k))
            results[k]['hit_rate'].append(hit_rate_at_k(recommended, relevant_items, k))
        
        # Calculate RMSE (on test ratings)
        for _, row in group.iterrows():
            try:
                pred = model.predict(user_idx, row['item_idx'])
                actual = (row['rating'] - 1) / 4.0  # Normalize to 0-1
                all_predictions.append(pred)
                all_actuals.append(actual)
            except:
                continue
    
    # Aggregate results
    final_results = {}
    for k in k_values:
        final_results[f'precision@{k}'] = np.mean(results[k]['precision'])
        final_results[f'recall@{k}'] = np.mean(results[k]['recall'])
        final_results[f'ndcg@{k}'] = np.mean(results[k]['ndcg'])
        final_results[f'hit_rate@{k}'] = np.mean(results[k]['hit_rate'])
    
    if len(all_predictions) > 0:
        final_results['rmse'] = rmse(all_predictions, all_actuals)
        final_results['mae'] = mean_absolute_error(all_predictions, all_actuals)
    
    return final_results


def evaluate_cold_start(model, test_df, train_df, cold_users, k_values=[5, 10, 20],
                        user_features=None, item_features=None):
    """
    Evaluate model performance on cold-start users
    
    Args:
        model: trained recommendation model
        test_df: test DataFrame
        train_df: training DataFrame
        cold_users: list of cold-start user indices
        k_values: list of K values
        user_features: user feature matrix
        item_features: item feature matrix
    
    Returns:
        Dictionary of cold-start evaluation results
    """
    
    # Filter test data for cold users only
    cold_test_df = test_df[test_df['user_idx'].isin(cold_users)]
    
    if len(cold_test_df) == 0:
        print("No cold users in test set")
        return {}
    
    results = {k: {'precision': [], 'recall': [], 'ndcg': [], 'hit_rate': []} 
               for k in k_values}
    
    all_predictions = []
    all_actuals = []
    
    cold_test_users = cold_test_df.groupby('user_idx')
    
    print(f"Evaluating cold-start performance for {len(cold_users)} users...")
    
    for user_idx, group in tqdm(cold_test_users, desc="Evaluating cold users"):
        relevant_items = set(group[group['rating'] >= 4]['item_idx'].values)
        
        if len(relevant_items) == 0:
            continue
        
        max_k = max(k_values)
        
        # For cold start, we don't exclude any items
        try:
            if hasattr(model, 'recommend') and 'is_cold_user' in model.recommend.__code__.co_varnames:
                recommended = model.recommend(user_idx, n_items=max_k, is_cold_user=True)
            else:
                recommended = model.recommend(user_idx, n_items=max_k)
        except Exception as e:
            continue
        
        for k in k_values:
            results[k]['precision'].append(precision_at_k(recommended, relevant_items, k))
            results[k]['recall'].append(recall_at_k(recommended, relevant_items, k))
            results[k]['ndcg'].append(ndcg_at_k(recommended, relevant_items, k))
            results[k]['hit_rate'].append(hit_rate_at_k(recommended, relevant_items, k))
        
        # RMSE
        for _, row in group.iterrows():
            try:
                if hasattr(model, 'predict') and 'is_cold_user' in model.predict.__code__.co_varnames:
                    pred = model.predict(user_idx, row['item_idx'], is_cold_user=True)
                else:
                    pred = model.predict(user_idx, row['item_idx'])
                actual = (row['rating'] - 1) / 4.0
                all_predictions.append(pred)
                all_actuals.append(actual)
            except:
                continue
    
    # Aggregate
    final_results = {}
    for k in k_values:
        if len(results[k]['precision']) > 0:
            final_results[f'cold_precision@{k}'] = np.mean(results[k]['precision'])
            final_results[f'cold_recall@{k}'] = np.mean(results[k]['recall'])
            final_results[f'cold_ndcg@{k}'] = np.mean(results[k]['ndcg'])
            final_results[f'cold_hit_rate@{k}'] = np.mean(results[k]['hit_rate'])
    
    if len(all_predictions) > 0:
        final_results['cold_rmse'] = rmse(all_predictions, all_actuals)
    
    return final_results


if __name__ == "__main__":
    # Test metrics
    recommended = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    relevant = {2, 5, 8, 11, 15}
    
    print(f"Precision@5: {precision_at_k(recommended, relevant, 5):.4f}")
    print(f"Recall@5: {recall_at_k(recommended, relevant, 5):.4f}")
    print(f"NDCG@5: {ndcg_at_k(recommended, relevant, 5):.4f}")
    print(f"Hit Rate@5: {hit_rate_at_k(recommended, relevant, 5)}")
    
    predictions = [0.8, 0.6, 0.7, 0.9, 0.5]
    actuals = [0.9, 0.5, 0.8, 0.85, 0.6]
    print(f"RMSE: {rmse(predictions, actuals):.4f}")