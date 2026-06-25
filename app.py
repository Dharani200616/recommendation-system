import os
import sys
from flask import Flask, render_template, request, jsonify

# Add parent directory to path so config and src can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.recommendation import HybridRecommender
from src.utils import get_rich_console, load_pkl

console = get_rich_console()

app = Flask(__name__)
recommender = None

def get_recommender():
    global recommender
    if recommender is None:
        recommender = HybridRecommender()
    return recommender

@app.route('/')
def index():
    rec = get_recommender()
    
    # Calculate some fast stats for the UI Dashboard
    num_users = 0
    num_movies = 0
    num_ratings = 0
    avg_rating = 0.0
    
    if rec.train_ratings is not None:
        num_users = int(rec.train_ratings['user_id'].nunique())
        num_ratings = int(len(rec.train_ratings))
        avg_rating = float(rec.train_ratings['rating'].mean())
        
    if rec.movies_df is not None:
        num_movies = int(len(rec.movies_df))
        
    # Get a list of genres to populate the filter dropdown
    genres_set = set()
    if rec.movies_df is not None:
        for val in rec.movies_df['genres'].dropna():
            for g in val.split('|'):
                genres_set.add(g.strip())
    sorted_genres = sorted(list(genres_set))
    
    # Get list of top popular movies to display in dashboard
    popular_movies = rec.get_popular_movies(top_n=5).to_dict(orient='records')
    
    return render_template(
        'index.html',
        num_users=num_users,
        num_movies=num_movies,
        num_ratings=num_ratings,
        avg_rating=round(avg_rating, 2),
        genres=sorted_genres,
        popular_movies=popular_movies
    )

@app.route('/recommendations')
def recommendations():
    rec = get_recommender()
    user_id = request.args.get('user_id', 1, type=int)
    method = request.args.get('method', 'hybrid', type=str)
    genre = request.args.get('genre', '', type=str)
    
    if genre == 'All' or not genre:
        genre = None
        
    recs = rec.get_recommendations(user_id=user_id, top_n=10, method=method, genre=genre)
    
    # Get user rating history
    user_history = []
    if rec.train_ratings is not None and rec.movies_df is not None:
        history_df = rec.train_ratings[rec.train_ratings['user_id'] == user_id]
        history_df = history_df.merge(rec.movies_df, on='movie_id')
        history_df = history_df.sort_values(by='rating', ascending=False)
        user_history = history_df.head(8)[['title', 'genres', 'rating']].to_dict(orient='records')
        
    return jsonify({
        'user_id': user_id,
        'method': method,
        'genre': genre or 'All',
        'recommendations': recs,
        'history': user_history
    })

@app.route('/movies/search')
def search_movies():
    rec = get_recommender()
    query = request.args.get('q', '', type=str)
    results = rec.search_movies(query) if query else []
    return jsonify(results[:10])

@app.route('/rate', methods=['POST'])
def add_rate():
    """Simulates adding a live rating from a user, updating the model memory."""
    rec = get_recommender()
    data = request.get_json()
    
    user_id = int(data.get('user_id', 1))
    movie_id = int(data.get('movie_id'))
    rating = float(data.get('rating', 5.0))
    
    if rec.train_ratings is not None:
        # Check if user already rated it
        mask = (rec.train_ratings['user_id'] == user_id) & (rec.train_ratings['movie_id'] == movie_id)
        if mask.any():
            rec.train_ratings.loc[mask, 'rating'] = rating
        else:
            # Append new rating
            import time
            new_row = {
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': rating,
                'timestamp': int(time.time())
            }
            # Convert to DataFrame properly
            rec.train_ratings = pd.concat([rec.train_ratings, pd.DataFrame([new_row])], ignore_index=True)
            
        # Re-fit Collaborative Filtering model in-memory for instant feedback
        if rec.cf_model:
            rec.cf_model.fit(rec.train_ratings)
            
        return jsonify({'status': 'success', 'message': f'Rating of {rating} stars added successfully!'})
        
    return jsonify({'status': 'error', 'message': 'Data not loaded yet'}), 400

if __name__ == '__main__':
    console.print(f"[success]🚀 Starting Flask Dashboard on http://{config.FLASK_HOST}:{config.FLASK_PORT}...[/success]")
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
