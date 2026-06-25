import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
import config
from src.utils import get_rich_console, print_section, save_pkl, load_pkl
from src.preprocessing import preprocess_pipeline
from src.collaborative_filtering import UserCollaborativeFiltering
from src.matrix_factorization import SVDMatrixFactorization
from src.recommendation import HybridRecommender
from src.evaluation import evaluate_predictions, evaluate_precision_recall_at_k


console = get_rich_console()

def plot_and_save_comparison(results_df):
    """Generates comparison bar charts and saves them to outputs/graphs directory."""
    sns.set_theme(style="darkgrid")
    
    # Melt dataframe for easy plotting with seaborn
    melted = results_df.melt(id_vars='Method', var_name='Metric', value_name='Score')
    
    # 1. Error Metrics (RMSE, MAE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    error_data = melted[melted['Metric'].isin(['RMSE', 'MAE'])]
    sns.barplot(x='Method', y='Score', hue='Metric', data=error_data, ax=axes[0], palette='crest')
    axes[0].set_title('Error Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Error Value')
    axes[0].set_ylim(0, 1.2)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() - 0.08),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', color='white', fontweight='bold')
        
    # 2. Ranking Metrics (Precision@10, Recall@10, F1)
    ranking_data = melted[melted['Metric'].isin(['Precision@10', 'Recall@10', 'F1-Score@10'])]
    sns.barplot(x='Method', y='Score', hue='Metric', data=ranking_data, ax=axes[1], palette='flare')
    axes[1].set_title('Recommendation Accuracy (Higher is Better)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Score Ratio')
    axes[1].set_ylim(0, 1.1)
    for p in axes[1].patches:
        if p.get_height() > 0:
            axes[1].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() - 0.08),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', color='white', fontweight='bold')
            
    plt.tight_layout()
    graph_path = os.path.join(config.GRAPHS_DIR, 'model_comparison.png')
    plt.savefig(graph_path, dpi=300)
    plt.close()
    
    console.print(f"[success]✔ Evaluation graphs exported to: [bold]{graph_path}[/bold][/success]")

def train_and_evaluate():
    """Main runner to load data, fit recommendation models, calculate metrics, and serialize."""
    print_section("RECOMMENDATION SYSTEM ENGINE TRAINING PIPELINE", "★")
    
    # 1. Run Preprocessing
    data = preprocess_pipeline()
    train_ratings = data['train_ratings']
    test_ratings = data['test_ratings']
    
    # 2. Fit User Collaborative Filtering
    cf_model = UserCollaborativeFiltering()
    cf_model.fit(train_ratings)
    save_pkl(cf_model, config.USER_SIMILARITY_PKL)
    
    # 3. Fit SVD Matrix Factorization
    svd_model = SVDMatrixFactorization()
    svd_model.fit(train_ratings)
    save_pkl(svd_model, config.SVD_MODEL_PKL)
    
    # 4. Initialize Hybrid Recommender wrapper
    hybrid_rec = HybridRecommender(
        cf_model=cf_model,
        svd_model=svd_model,
        movies_df=data['movies'],
        train_ratings=train_ratings
    )
    
    # 5. Evaluate All Approaches
    console.print("\n[info]Evaluating models on held-out test dataset...[/info]")
    
    results = []
    
    # Collaborative Filtering
    cf_rmse, cf_mae = evaluate_predictions(cf_model, test_ratings)
    cf_prec, cf_rec, cf_f1 = evaluate_precision_recall_at_k(cf_model, test_ratings, k=10)
    results.append({
        'Method': 'User-Based CF', 'RMSE': cf_rmse, 'MAE': cf_mae, 
        'Precision@10': cf_prec, 'Recall@10': cf_rec, 'F1-Score@10': cf_f1
    })
    
    # SVD Matrix Factorization
    svd_rmse, svd_mae = evaluate_predictions(svd_model, test_ratings)
    svd_prec, svd_rec, svd_f1 = evaluate_precision_recall_at_k(svd_model, test_ratings, k=10)
    results.append({
        'Method': 'Matrix Fact (SVD)', 'RMSE': svd_rmse, 'MAE': svd_mae, 
        'Precision@10': svd_prec, 'Recall@10': svd_rec, 'F1-Score@10': svd_f1
    })
    
    # Hybrid Model
    hybrid_rmse, hybrid_mae = evaluate_predictions(hybrid_rec, test_ratings, method='hybrid')
    hybrid_prec, hybrid_rec_val, hybrid_f1 = evaluate_precision_recall_at_k(hybrid_rec, test_ratings, k=10, method='hybrid')
    results.append({
        'Method': 'Hybrid (CF + SVD)', 'RMSE': hybrid_rmse, 'MAE': hybrid_mae, 
        'Precision@10': hybrid_prec, 'Recall@10': hybrid_rec_val, 'F1-Score@10': hybrid_f1
    })
    
    df_results = pd.DataFrame(results)
    
    # 6. Render Terminal Table
    table = Table(title="Model Evaluation Performance Metrics", header_style="bold magenta")
    table.add_column("Recommender Method", style="cyan", justify="left")
    table.add_column("RMSE (lower is better)", style="green", justify="right")
    table.add_column("MAE (lower is better)", style="green", justify="right")
    table.add_column("Precision@10 (higher is better)", style="yellow", justify="right")
    table.add_column("Recall@10 (higher is better)", style="yellow", justify="right")
    table.add_column("F1-Score (higher is better)", style="orange1", justify="right")
    
    for _, row in df_results.iterrows():
        table.add_row(
            row['Method'],
            f"{row['RMSE']:.4f}",
            f"{row['MAE']:.4f}",
            f"{row['Precision@10']:.4f}",
            f"{row['Recall@10']:.4f}",
            f"{row['F1-Score@10']:.4f}"
        )
        
    console.print("\n")
    console.print(table)
    
    # 7. Generate & Save Comparison Plot
    plot_and_save_comparison(df_results)
    
    # 8. Write Evaluation Report
    write_markdown_report(df_results)
    
    console.print("\n[bold success]🎉 Model training, evaluation, and logging finished successfully![/bold success]\n")

def write_markdown_report(results_df):
    """Writes a beautifully formatted markdown evaluation report."""
    report_path = os.path.join(config.REPORTS_DIR, 'evaluation_report.md')
    
    markdown_content = f"""# Model Evaluation & Validation Report

This report summarizes the comparative results of our Recommendation System algorithms: Mean-Centered User-Based Collaborative Filtering, Matrix Factorization SVD optimized with SGD, and an ensemble Hybrid model.

## Performance Metrics Table

| Recommender Method | RMSE (↓) | MAE (↓) | Precision@10 (↑) | Recall@10 (↑) | F1-Score@10 (↑) |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for _, r in results_df.iterrows():
        markdown_content += f"| **{r['Method']}** | {r['RMSE']:.4f} | {r['MAE']:.4f} | {r['Precision@10']:.4f} | {r['Recall@10']:.4f} | {r['F1-Score@10']:.4f} |\n"
        
    markdown_content += """
## Key Insights & Rationale

1. **SVD Matrix Factorization** yields low reconstruction errors (RMSE/MAE) because it learns rich, lower-dimensional latent factors representing underlying user behavior and movie attributes.
2. **User-Based Collaborative Filtering** excels at finding immediate similarities in rating scales, offering high-fidelity predictions but suffering from the sparsity constraint.
3. **The Hybrid Model** (combining CF and SVD predictions) balances localized similarity structures with latent theme dimensions, providing robust recommendations and stabilizing the F1-score performance.

---
Report generated automatically by the AI Engine.
"""
    with open(report_path, 'w') as f:
        f.write(markdown_content)
        
    console.print(f"[success]✔ Markdown performance report written to: [bold]{report_path}[/bold][/success]")

if __name__ == "__main__":
    train_and_evaluate()
