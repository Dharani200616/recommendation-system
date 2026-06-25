import numpy as np
import pandas as pd
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import config
from src.utils import get_rich_console

console = get_rich_console()

class SVDMatrixFactorization:
    def __init__(self, n_factors=None, lr=None, reg=None, epochs=None):
        self.n_factors = n_factors or config.SVD_K
        self.lr = lr or config.SVD_LR
        self.reg = reg or config.SVD_REG
        self.epochs = epochs or config.SVD_EPOCHS
        
        # Latent parameters and biases
        self.mu = 0.0
        self.b_u = None # User biases
        self.b_i = None # Item biases
        self.P = None   # User latent factors
        self.Q = None   # Item latent factors
        
        # User/Item mappings
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        
    def fit(self, train_ratings, verbose=True):
        """Fits SVD on the training ratings data using Stochastic Gradient Descent (SGD)."""
        console.print(f"[info]Fitting SVD Matrix Factorization model (Factors: {self.n_factors}, Epochs: {self.epochs})...[/info]")
        
        # Extract unique users and items
        unique_users = train_ratings['user_id'].unique()
        unique_items = train_ratings['movie_id'].unique()
        
        # Generate mapping indices
        self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user = {idx: uid for idx, uid in enumerate(unique_users)}
        self.item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
        self.idx_to_item = {idx: iid for idx, iid in enumerate(unique_items)}
        
        num_users = len(unique_users)
        num_items = len(unique_items)
        
        # Initialize biases and latent vectors
        self.mu = train_ratings['rating'].mean()
        self.b_u = np.zeros(num_users)
        self.b_i = np.zeros(num_items)
        
        # Initialize user and item latent matrices with random normal values
        self.P = np.random.normal(0, 0.1, (num_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (num_items, self.n_factors))
        
        # Parse records for faster iteration in SGD
        u_indices = np.array([self.user_to_idx[uid] for uid in train_ratings['user_id']])
        i_indices = np.array([self.item_to_idx[iid] for iid in train_ratings['movie_id']])
        ratings = train_ratings['rating'].values
        
        num_samples = len(ratings)
        
        # Training loop
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            console=console,
            disable=not verbose
        ) as progress:
            
            task = progress.add_task("[yellow]SVD Optimization (SGD)...[/yellow]", total=self.epochs)
            
            for epoch in range(self.epochs):
                loss_sum = 0.0
                
                # Shuffle the training indices each epoch for SGD stability
                shuffled_indices = np.random.permutation(num_samples)
                
                for idx in shuffled_indices:
                    u = u_indices[idx]
                    i = i_indices[idx]
                    r = ratings[idx]
                    
                    # Compute prediction
                    prediction = self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
                    
                    # Compute prediction error
                    err = r - prediction
                    loss_sum += err ** 2
                    
                    # Update biases using SGD update equations
                    self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                    self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])
                    
                    # Update latent matrices
                    p_temp = self.P[u].copy()
                    self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                    self.Q[i] += self.lr * (err * p_temp - self.reg * self.Q[i])
                
                # Compute and print training RMSE for the epoch
                epoch_rmse = np.sqrt(loss_sum / num_samples)
                progress.update(task, advance=1, description=f"[yellow]SVD Epoch {epoch+1}/{self.epochs} - RMSE: {epoch_rmse:.4f}[/yellow]")
                
        console.print("[success]✔ SVD Matrix Factorization optimized successfully![/success]")
        return self
        
    def predict(self, user_id, movie_id):
        """Predicts the rating a user would give to a movie using the SVD model."""
        user_exists = user_id in self.user_to_idx
        item_exists = movie_id in self.item_to_idx
        
        if user_exists and item_exists:
            u_idx = self.user_to_idx[user_id]
            i_idx = self.item_to_idx[movie_id]
            prediction = self.mu + self.b_u[u_idx] + self.b_i[i_idx] + np.dot(self.P[u_idx], self.Q[i_idx])
            return float(np.clip(prediction, 1.0, 5.0))
        elif user_exists:
            # Fallback for unseen items: return global mean + user bias
            u_idx = self.user_to_idx[user_id]
            return float(np.clip(self.mu + self.b_u[u_idx], 1.0, 5.0))
        elif item_exists:
            # Fallback for unseen users: return global mean + item bias
            i_idx = self.item_to_idx[movie_id]
            return float(np.clip(self.mu + self.b_i[i_idx], 1.0, 5.0))
        else:
            # Fallback for complete cold start
            return self.mu

    def recommend(self, user_id, top_n=10, exclude_rated=True, train_ratings=None):
        """Generates recommendations for a specific user."""
        if user_id not in self.user_to_idx:
            return []
            
        rated_movies = []
        if exclude_rated and train_ratings is not None:
            rated_movies = train_ratings[train_ratings['user_id'] == user_id]['movie_id'].tolist()
            
        all_movies = list(self.item_to_idx.keys())
        unrated_movies = [m for m in all_movies if m not in rated_movies]
        
        predictions = []
        for movie_id in unrated_movies:
            pred_rating = self.predict(user_id, movie_id)
            predictions.append((movie_id, pred_rating))
            
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_n]
