import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
GRAPHS_DIR = os.path.join(OUTPUTS_DIR, 'graphs')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# Create directories if they do not exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, GRAPHS_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data File Paths
MOVIES_CSV = os.path.join(RAW_DATA_DIR, 'movies.csv')
USERS_CSV = os.path.join(RAW_DATA_DIR, 'users.csv')
RATINGS_CSV = os.path.join(RAW_DATA_DIR, 'ratings.csv')

PREPROCESSED_RATINGS_CSV = os.path.join(PROCESSED_DATA_DIR, 'preprocessed_ratings.csv')
ENCODED_DATA_PKL = os.path.join(PROCESSED_DATA_DIR, 'encoded_data.pkl')

# Model File Paths
USER_SIMILARITY_PKL = os.path.join(MODELS_DIR, 'user_similarity.pkl')
SVD_MODEL_PKL = os.path.join(MODELS_DIR, 'svd_model.pkl')

# Algorithm Configuration
SVD_K = 15              # Number of latent factors
SVD_LR = 0.005           # Learning rate
SVD_REG = 0.02          # Regularization parameter
SVD_EPOCHS = 20         # Number of epochs for training SVD

CF_MIN_SIMILARITY = 0.1 # Minimum user similarity to consider for recommendations
CF_K_NEIGHBORS = 10     # Number of nearest neighbors to aggregate ratings from

HYBRID_WEIGHT_CF = 0.4  # Weight for Collaborative Filtering in Hybrid model
HYBRID_WEIGHT_SVD = 0.6 # Weight for SVD Matrix Factorization in Hybrid model

# Flask App Configuration
FLASK_HOST = '127.0.0.1'
FLASK_PORT = 5000
FLASK_DEBUG = True
