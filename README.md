# CineMatch AI: Hybrid Movie Recommendation Engine

CineMatch AI is a modern, high-performance hybrid movie recommendation engine that merges the predictive strengths of **User-Based Collaborative Filtering** and **Matrix Factorization (Singular Value Decomposition)** optimized with **Stochastic Gradient Descent (SGD)**.

This repository features:
- **Synthetic/Sample Data Simulator**: Generates movie metadata, user demographics, and realistic ratings structured around hidden preference categories.
- **Dual Recommender Machine Learning Engine**: Written from scratch in Python/NumPy/Pandas.
- **Stunning Terminal Console Dashboard**: Built using `rich` with live training tables, interactive panels, progress bars, and execution spinners.
- **Premium Glassmorphism Web App**: A luxury dark-mode Flask web dashboard incorporating responsive visual blocks, animated search catalogs, star sliders, and a live rating feedback loop.

---

## 📂 File Architecture

```text
recommendation-system-project/
├── data/
│   ├── raw/                  # Generated CSV catalogs (movies, users, ratings)
│   └── processed/            # Cleaned data splits & cache serialization
├── notebooks/                # Exploratory Jupyter Notebook guides
│   ├── EDA.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
├── src/                      # Machine learning and preprocessing source
│   ├── preprocessing.py
│   ├── collaborative_filtering.py
│   ├── matrix_factorization.py
│   ├── recommendation.py
│   ├── evaluation.py
│   └── utils.py
├── app/                      # Flask dashboard web application
│   ├── app.py
│   ├── templates/            # HTML glassmorphism screens (index, result)
│   └── static/               # Style animations & front-end controller scripts
├── models/                   # Pickled mathematical weights and structures
├── outputs/                  # Exported performance charts & visualization
├── reports/                  # Generated model markdown validation reports
├── config.py                 # Hyper-parameter configurations & directory paths
├── create_sample_data.py     # High-fidelity synthetic generator
├── requirements.txt          # Python dependencies
├── README.md                 # Complete documentation
├── train.py                  # ML training and evaluation script
├── predict.py                # Command-line query tool
├── main.py                   # Master startup script
└── run.bat                   # Setup and boot script for Windows
```

---

## ⚡ Setup & Execution

### Option A: Fully Automated Boot (Recommended for Windows)
If you are on Windows, simply double-click the **`run.bat`** file or run it in your terminal:
```cmd
.\run.bat
```
This script will automatically:
1. Initialize a Python virtual environment (`venv`).
2. Upgrade pip and install all project dependencies from `requirements.txt`.
3. Generate the movie datasets, train the SVD/CF models, print terminal charts, and start the web dashboard server.

---

### Option B: Step-by-Step Shell Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Synthetic Data**:
   ```bash
   python create_sample_data.py
   ```

3. **Train and Evaluate Machine Learning Models**:
   ```bash
   python train.py
   ```

4. **Query Recommendations from CLI**:
   ```bash
   python predict.py --user 1 --method hybrid --top_n 10
   ```

5. **Start Flask Web App**:
   ```bash
   python app/app.py
   ```
   Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## 🧠 ML Engine Architecture

### 1. Collaborative Filtering (40% Weight)
Using **Mean-Centered User-Based Cosine Similarity** to handle rating scale offsets:
$$\hat{r}_{u,i} = \bar{r}_u + \frac{\sum_{v \in N_i(u)} sim(u, v) \cdot (r_{v,i} - \bar{r}_v)}{\sum_{v \in N_i(u)} |sim(u, v)|}$$

### 2. SVD Matrix Factorization (60% Weight)
Singular Value Decomposition from scratch. Resolves matrix sparsity by projecting ratings into 15-dimensional latent vectors ($P$ and $Q$) using Stochastic Gradient Descent (SGD):
$$e_{u,i} = r_{u,i} - (\mu + b_u + b_i + P_u \cdot Q_i^T)$$
$$\text{Objective function updates weights for biases } b_u, b_i \text{ and vectors } P_u, Q_i.$$

### 3. Hybrid Ensemble
Combines the neighborhood specificity of CF with the dimensional robustness of SVD, offering highly resilient recommendations and mitigating cold start scenarios.
