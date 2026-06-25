import numpy as np
import pandas as pd
from collections import defaultdict
import config
from src.utils import get_rich_console

console = get_rich_console()

def calculate_rmse(y_true, y_pred):
    """Calculates Root Mean Squared Error."""
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def calculate_mae(y_true, y_pred):
    """Calculates Mean Absolute Error."""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def evaluate_predictions(model, test_ratings, method='hybrid'):
    """Evaluates the predictions of a model on the test ratings set using RMSE and MAE."""
    y_true = []
    y_pred = []
    
    for _, row in test_ratings.iterrows():
        uid = int(row['user_id'])
        mid = int(row['movie_id'])
        r_actual = float(row['rating'])
        
        # Depending on model class type, predict rating
        if hasattr(model, 'predict_rating'):
            r_pred = model.predict_rating(uid, mid, method=method)
        else:
            r_pred = model.predict(uid, mid)
            
        y_true.append(r_actual)
        y_pred.append(r_pred)
        
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    return rmse, mae

def evaluate_precision_recall_at_k(model, test_ratings, k=10, threshold=3.5, method='hybrid'):
    """Calculates Precision@K and Recall@K for the model."""
    # First, map user to a list of (predicted_rating, actual_rating)
    user_ratings = defaultdict(list)
    
    for _, row in test_ratings.iterrows():
        uid = int(row['user_id'])
        mid = int(row['movie_id'])
        r_actual = float(row['rating'])
        
        if hasattr(model, 'predict_rating'):
            r_pred = model.predict_rating(uid, mid, method=method)
        else:
            r_pred = model.predict(uid, mid)
            
        user_ratings[uid].append((r_pred, r_actual))
        
    precisions = {}
    recalls = {}
    
    for uid, ratings_list in user_ratings.items():
        # Sort ratings by predicted value
        ratings_list.sort(key=lambda x: x[0], reverse=True)
        
        # Number of actual relevant items (actual rating >= threshold)
        n_rel = sum((act >= threshold) for (_, act) in ratings_list)
        
        # Number of recommended items in top-K that are predicted relevant
        n_rec_k = sum((pred >= threshold) for (pred, _) in ratings_list[:k])
        
        # Number of recommended items in top-K that are actually relevant
        n_rel_and_rec_k = sum(((act >= threshold) and (pred >= threshold)) for (pred, act) in ratings_list[:k])
        
        # Precision@K: Proportion of recommended items that are actually relevant
        # If no items recommended in top K, precision is 1.0 (since no false positives)
        precisions[uid] = n_rel_and_rec_k / k
        
        # Recall@K: Proportion of actual relevant items that are recommended in top K
        # If no actual relevant items, recall is 1.0 (since no false negatives)
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1.0
        
    # Calculate average across all users
    avg_precision = np.mean(list(precisions.values()))
    avg_recall = np.mean(list(recalls.values()))
    
    # Calculate F1-score
    if avg_precision + avg_recall > 0:
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    else:
        f1_score = 0.0
        
    return avg_precision, avg_recall, f1_score
