import os
import pandas as pd
from sklearn.model_selection import train_test_split
import config
from src.utils import get_rich_console, save_pkl

console = get_rich_console()

def load_raw_data():
    """Loads raw dataframes from CSV files."""
    if not (os.path.exists(config.RATINGS_CSV) and 
            os.path.exists(config.MOVIES_CSV) and 
            os.path.exists(config.USERS_CSV)):
        raise FileNotFoundError("Raw files missing. Please run `create_sample_data.py` first.")
    
    ratings = pd.read_csv(config.RATINGS_CSV)
    movies = pd.read_csv(config.MOVIES_CSV)
    users = pd.read_csv(config.USERS_CSV)
    return ratings, movies, users

def preprocess_pipeline(test_size=0.2, random_state=42):
    """Executes full preprocessing: cleans datasets, creates mappings, splits train/test, and serializes."""
    console.print("[info]Starting preprocessing pipeline...[/info]")
    
    # 1. Load Data
    ratings, movies, users = load_raw_data()
    
    # 2. Check for missing values
    ratings.dropna(subset=['user_id', 'movie_id', 'rating'], inplace=True)
    movies.dropna(subset=['movie_id', 'title', 'genres'], inplace=True)
    users.dropna(subset=['user_id'], inplace=True)
    
    # 3. Handle Types
    ratings['user_id'] = ratings['user_id'].astype(int)
    ratings['movie_id'] = ratings['movie_id'].astype(int)
    ratings['rating'] = ratings['rating'].astype(float)
    
    movies['movie_id'] = movies['movie_id'].astype(int)
    users['user_id'] = users['user_id'].astype(int)
    
    # 4. Check boundaries and stats
    num_users = ratings['user_id'].nunique()
    num_movies = ratings['movie_id'].nunique()
    total_ratings = len(ratings)
    sparsity = 1.0 - (total_ratings / (num_users * num_movies))
    
    console.print(f"[info]  - Unique Users: {num_users}[/info]")
    console.print(f"[info]  - Unique Movies in Ratings: {num_movies} (Catalog: {len(movies)})[/info]")
    console.print(f"[info]  - Sparsity: {sparsity * 100:.2f}%[/info]")
    
    # 5. Split Train / Test
    train_ratings, test_ratings = train_test_split(
        ratings, test_size=test_size, random_state=random_state, stratify=ratings['user_id']
    )
    
    # 6. Save preprocessed data
    train_ratings.to_csv(config.PREPROCESSED_RATINGS_CSV, index=False)
    
    # Bundle data for modeling
    encoded_data = {
        'train_ratings': train_ratings,
        'test_ratings': test_ratings,
        'movies': movies,
        'users': users,
        'num_users': max(ratings['user_id'].max(), users['user_id'].max()) + 1,
        'num_movies': max(ratings['movie_id'].max(), movies['movie_id'].max()) + 1
    }
    
    save_pkl(encoded_data, config.ENCODED_DATA_PKL)
    
    console.print("[success]✔ Preprocessing pipeline completed and cached successfully![/success]")
    return encoded_data

if __name__ == "__main__":
    preprocess_pipeline()
