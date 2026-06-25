import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import config
from src.utils import get_rich_console

console = get_rich_console()

class UserCollaborativeFiltering:
    def __init__(self, k_neighbors=None, min_similarity=None):
        self.k_neighbors = k_neighbors or config.CF_K_NEIGHBORS
        self.min_similarity = min_similarity or config.CF_MIN_SIMILARITY
        self.user_item_matrix = None
        self.user_similarity = None
        self.user_means = None
        self.global_mean = 3.0
        self.all_users = []
        self.all_movies = []

    def fit(self, train_ratings):
        """Fits the CF model on the training ratings data."""
        console.print("[info]Fitting User-Based Collaborative Filtering model...[/info]")
        
        self.global_mean = train_ratings['rating'].mean()
        self.all_users = sorted(train_ratings['user_id'].unique())
        self.all_movies = sorted(train_ratings['movie_id'].unique())
        
        # Create Pivot Table
        self.user_item_matrix = train_ratings.pivot(
            index='user_id', columns='movie_id', values='rating'
        )
        
        # Calculate mean rating for each user
        self.user_means = self.user_item_matrix.mean(axis=1)
        
        # Center ratings (mean-centering) to address differing rating scales
        centered_matrix = self.user_item_matrix.sub(self.user_means, axis=0).fillna(0)
        
        # Compute Cosine Similarity between users
        console.print("[info]  - Calculating User-User Similarity Matrix...[/info]")
        sim_matrix = cosine_similarity(centered_matrix.values)
        
        self.user_similarity = pd.DataFrame(
            sim_matrix, index=self.user_item_matrix.index, columns=self.user_item_matrix.index
        )
        
        # Set self-similarity to 0 to prevent self-referential prediction bias
        np.fill_diagonal(self.user_similarity.values, 0)
        
        console.print("[success]✔ Collaborative Filtering fitted successfully![/success]")
        return self

    def predict(self, user_id, movie_id):
        """Predicts the rating a user would give to a movie."""
        # Handle cases where user or movie is unseen (Cold Start)
        if user_id not in self.user_means.index:
            return self.global_mean
            
        user_mean = self.user_means.loc[user_id]
        
        if movie_id not in self.all_movies:
            return user_mean # fallback to user's average rating
            
        # Get users who have rated the movie
        other_ratings = self.user_item_matrix[movie_id].dropna()
        if len(other_ratings) == 0:
            return user_mean
            
        # Get similarities of other users with the target user
        similarities = self.user_similarity.loc[user_id, other_ratings.index]
        
        # Filter by minimum similarity threshold
        valid_indices = similarities[similarities >= self.min_similarity].index
        if len(valid_indices) == 0:
            return user_mean # fallback to average
            
        similarities = similarities.loc[valid_indices]
        other_ratings = other_ratings.loc[valid_indices]
        
        # Get top-K neighbors
        if len(similarities) > self.k_neighbors:
            top_k_indices = similarities.nlargest(self.k_neighbors).index
            similarities = similarities.loc[top_k_indices]
            other_ratings = other_ratings.loc[top_k_indices]
            
        # Compute mean-centered weighted average rating
        other_means = self.user_means.loc[other_ratings.index]
        centered_ratings = other_ratings - other_means
        
        sim_sum = similarities.abs().sum()
        if sim_sum == 0:
            return user_mean
            
        predicted_rating = user_mean + (np.dot(similarities, centered_ratings) / sim_sum)
        return float(np.clip(predicted_rating, 1.0, 5.0))

    def recommend(self, user_id, top_n=10, exclude_rated=True):
        """Generates recommendations for a specific user."""
        if user_id not in self.user_means.index:
            # Cold start: return top movies overall (fallback)
            return []
            
        rated_movies = []
        if exclude_rated:
            rated_movies = self.user_item_matrix.loc[user_id].dropna().index.tolist()
            
        unrated_movies = [m for m in self.all_movies if m not in rated_movies]
        
        predictions = []
        for movie_id in unrated_movies:
            pred_rating = self.predict(user_id, movie_id)
            predictions.append((movie_id, pred_rating))
            
        # Sort predictions by rating desc
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_n]
