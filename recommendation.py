import pandas as pd
import numpy as np
import config
from src.utils import get_rich_console, load_pkl

console = get_rich_console()

class HybridRecommender:
    def __init__(self, cf_model=None, svd_model=None, movies_df=None, train_ratings=None):
        self.cf_model = cf_model
        self.svd_model = svd_model
        self.movies_df = movies_df
        self.train_ratings = train_ratings
        
        # Load from models folder if not provided
        self._load_cached_models()
        
    def _load_cached_models(self):
        """Loads models and data frames from disk if they exist and are not already loaded."""
        if self.movies_df is None or self.train_ratings is None:
            encoded_data = load_pkl(config.ENCODED_DATA_PKL)
            if encoded_data:
                self.movies_df = encoded_data.get('movies')
                self.train_ratings = encoded_data.get('train_ratings')
                
        if self.cf_model is None:
            self.cf_model = load_pkl(config.USER_SIMILARITY_PKL)
            
        if self.svd_model is None:
            self.svd_model = load_pkl(config.SVD_MODEL_PKL)

    def predict_rating(self, user_id, movie_id, method='hybrid'):
        """Predicts the rating a user would give to a movie using a selected method."""
        # Ensure models are loaded
        self._load_cached_models()
        
        if method == 'cf':
            if self.cf_model:
                return self.cf_model.predict(user_id, movie_id)
            return 3.0 # Failsafe
        elif method == 'svd':
            if self.svd_model:
                return self.svd_model.predict(user_id, movie_id)
            return 3.0 # Failsafe
        else: # Hybrid
            cf_pred = self.cf_model.predict(user_id, movie_id) if self.cf_model else 3.0
            svd_pred = self.svd_model.predict(user_id, movie_id) if self.svd_model else 3.0
            
            # Combine SVD and CF predictions using weights
            hybrid_pred = (config.HYBRID_WEIGHT_CF * cf_pred) + (config.HYBRID_WEIGHT_SVD * svd_pred)
            return float(np.clip(hybrid_pred, 1.0, 5.0))

    def get_popular_movies(self, top_n=10, genre=None):
        """Standard cold-start recommendation helper. Retrieves highest-rated popular movies."""
        if self.train_ratings is None or self.movies_df is None:
            self._load_cached_models()
            
        if self.train_ratings is None or self.movies_df is None:
            return pd.DataFrame()
            
        # Group ratings to get mean rating and count
        stats = self.train_ratings.groupby('movie_id').agg(
            avg_rating=('rating', 'mean'),
            vote_count=('rating', 'count')
        ).reset_index()
        
        # Merge movie titles
        popular = stats.merge(self.movies_df, on='movie_id')
        
        # Filter by genre if requested
        if genre:
            popular = popular[popular['genres'].str.contains(genre, case=False, na=False)]
            
        # Bayesian average or weighted formula to handle low votes
        # Weighted Rating = (v/(v+m) * R) + (m/(v+m) * C)
        # v = vote_count, m = min_votes (e.g. 5), R = avg_rating, C = global_mean
        m = 3
        C = self.train_ratings['rating'].mean()
        
        popular['weighted_score'] = (popular['vote_count'] / (popular['vote_count'] + m) * popular['avg_rating']) + \
                                    (m / (popular['vote_count'] + m) * C)
                                    
        # Sort by weighted score
        popular = popular.sort_values(by='weighted_score', ascending=False)
        return popular.head(top_n)

    def get_recommendations(self, user_id, top_n=10, method='hybrid', genre=None, exclude_rated=True):
        """Generates sorted list of recommended movies for a user, handling unseen users gracefully."""
        self._load_cached_models()
        
        # Validate user existence
        user_exists = False
        if self.train_ratings is not None:
            user_exists = user_id in self.train_ratings['user_id'].unique()
            
        # Handle cold start for missing or unseen users
        if not user_exists:
            console.print(f"[warning]User {user_id} is a new user (Cold Start). Returning popular movies...[/warning]")
            popular_df = self.get_popular_movies(top_n=top_n, genre=genre)
            if popular_df.empty:
                return []
            return popular_df[['movie_id', 'title', 'genres', 'weighted_score']].rename(
                columns={'weighted_score': 'predicted_rating'}
            ).to_dict(orient='records')
            
        # Get list of candidate movies
        rated_movies = []
        if exclude_rated and self.train_ratings is not None:
            rated_movies = self.train_ratings[self.train_ratings['user_id'] == user_id]['movie_id'].tolist()
            
        all_movies = self.movies_df.to_dict(orient='records') if self.movies_df is not None else []
        candidates = [m for m in all_movies if m['movie_id'] not in rated_movies]
        
        # Filter candidates by genre if requested
        if genre:
            candidates = [m for m in candidates if genre.lower() in m['genres'].lower()]
            
        predictions = []
        for movie in candidates:
            movie_id = movie['movie_id']
            pred_score = self.predict_rating(user_id, movie_id, method=method)
            predictions.append({
                'movie_id': movie_id,
                'title': movie['title'],
                'genres': movie['genres'],
                'release_year': movie.get('release_year', ''),
                'predicted_rating': round(pred_score, 2)
            })
            
        # Sort by prediction score descending
        predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return predictions[:top_n]
        
    def search_movies(self, query):
        """Search movies in catalog by title or genre."""
        self._load_cached_models()
        if self.movies_df is None:
            return []
            
        results = self.movies_df[
            self.movies_df['title'].str.contains(query, case=False, na=False) |
            self.movies_df['genres'].str.contains(query, case=False, na=False)
        ]
        return results.to_dict(orient='records')
