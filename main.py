import os
import sys
import time
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
import config

console = Console()

def print_banner():
    banner_text = """
 [bold magenta]██████╗██╗███╗   ██╗███████╗███╗   ███╗ █████╗ ████████╗ ██████╗██╗  ██╗[/bold magenta]
[bold purple]██╔════╝██║████╗  ██║██╔════╝████╗ Tell ██╔══██╗╚══██╔══╝██╔════╝██║  ██║[/bold purple]
[bold cyan]██║     ██║██╔██╗ ██║█████╗  ██╔██╗ ██╔███████║   ██║   ██║     ███████║[/bold cyan]
[bold blue]██║     ██║██║╚██╗██║██╔══╝  ██║╚██╔╝██║██╔══██║   ██║   ██║     ██╔══██║[/bold blue]
[bold teal]╚██████╗██║██║ ╚████║███████╗██║ ╚═╝ ██║██║  ██║   ██║   ╚██████╗██║  ██║[/bold teal]
[bold green] ╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝    ╚══════╝╚═╝  ╚═╝[/bold green]
           [bold yellow]★ INTEGRATED HYBRID RECOMMENDATION ENGINE ★[/bold yellow]
    """
    console.print(banner_text)

def show_recommendation_showcase(recommender):
    """Prints baseline recommendation predictions and rating history for a sample user in the CLI."""
    console.print("\n")
    console.print(Panel("[bold white]🎯 Live Terminal Recommendation Showcase (User 1)[/bold white]", border_style="cyan"))
    
    # User taste profile
    if recommender.train_ratings is not None and recommender.movies_df is not None:
        history_df = recommender.train_ratings[recommender.train_ratings['user_id'] == 1]
        history_df = history_df.merge(recommender.movies_df, on='movie_id')
        history_df = history_df.sort_values(by='rating', ascending=False)
        
        hist_table = Table(title="User 1: Historical Top Rated Movies", header_style="bold magenta")
        hist_table.add_column("Movie Title", style="bold white")
        hist_table.add_column("Genres", style="green")
        hist_table.add_column("Rating Given", style="bold yellow", justify="right")
        
        for _, row in history_df.head(5).iterrows():
            stars = "★" * int(round(row['rating'])) + "☆" * (5 - int(round(row['rating'])))
            hist_table.add_row(row['title'], row['genres'].replace('|', ' • '), f"{row['rating']:.1f} ({stars})")
        
        console.print(hist_table)
        console.print("\n")
        
    # SVD Recommendations
    recs_svd = recommender.get_recommendations(user_id=1, top_n=5, method='svd')
    svd_table = Table(title="Matrix Factorization (SVD) Top Picks", header_style="bold purple")
    svd_table.add_column("Rank", justify="center")
    svd_table.add_column("Title", style="bold white")
    svd_table.add_column("Score", style="bold yellow")
    
    for i, r in enumerate(recs_svd):
        svd_table.add_row(str(i+1), r['title'], f"{r['predicted_rating']:.2f}")
        
    # CF Recommendations
    recs_cf = recommender.get_recommendations(user_id=1, top_n=5, method='cf')
    cf_table = Table(title="Collaborative Filtering (CF) Top Picks", header_style="bold blue")
    cf_table.add_column("Rank", justify="center")
    cf_table.add_column("Title", style="bold white")
    cf_table.add_column("Score", style="bold yellow")
    
    for i, r in enumerate(recs_cf):
        cf_table.add_row(str(i+1), r['title'], f"{r['predicted_rating']:.2f}")
        
    # Hybrid Recommendations
    recs_hybrid = recommender.get_recommendations(user_id=1, top_n=5, method='hybrid')
    hybrid_table = Table(title="Hybrid Engine Ensemble Top Picks", header_style="bold green")
    hybrid_table.add_column("Rank", justify="center")
    hybrid_table.add_column("Title", style="bold white")
    hybrid_table.add_column("Score", style="bold yellow")
    
    for i, r in enumerate(recs_hybrid):
        hybrid_table.add_row(str(i+1), r['title'], f"{r['predicted_rating']:.2f}")
        
    # Print side-by-side tables
    console.print(Columns([svd_table, cf_table, hybrid_table]))
    console.print("\n")
    input("Press Enter to return to main menu...")

def run_cli_simulator(recommender):
    """Runs interactive CLI simulator allowing live rating additions and instant metric retraining."""
    console.print("\n")
    console.print(Panel("[bold purple]🎬 Interactive CLI Rating Simulator[/bold purple]", border_style="purple"))
    
    # 1. Get User ID
    try:
        user_input = input("Enter User ID to simulate (e.g. 1 to 200, or a new user like 999): ").strip()
        if not user_input:
            user_id = 1
        else:
            user_id = int(user_input)
    except ValueError:
        console.print("[red]Invalid user ID. Using User 1 by default.[/red]")
        user_id = 1
        
    console.print(f"[info]Selected target simulation user: [bold yellow]User {user_id}[/bold yellow][/info]")
    
    # 2. Get recommendations BEFORE rating
    console.print("\n[info]Fetching baseline hybrid recommendations...[/info]")
    recs_before = recommender.get_recommendations(user_id=user_id, top_n=5, method='hybrid')
    
    # 3. Choose movie to rate
    movie_query = input("\nSearch a movie to rate (e.g. Inception, Matrix, Finding Nemo): ").strip()
    matches = recommender.search_movies(movie_query)
    
    if not matches:
        console.print("[red]No movies found matching that query. Exiting simulator.[/red]")
        input("\nPress Enter to return to main menu...")
        return
        
    console.print("\n[bold cyan]Select movie from matches:[/bold cyan]")
    for idx, movie in enumerate(matches[:5]):
        console.print(f" [cyan][{idx + 1}][/cyan] {movie['title']} ({movie['genres'].replace('|', ' • ')})")
        
    try:
        selection_input = input("\nEnter selection index (1-5, default 1): ").strip()
        selection_idx = int(selection_input) - 1 if selection_input else 0
        if selection_idx < 0 or selection_idx >= len(matches[:5]):
            console.print("[red]Invalid selection. Selecting option 1 by default.[/red]")
            selection_idx = 0
    except ValueError:
        selection_idx = 0
        
    target_movie = matches[selection_idx]
    movie_id = target_movie['movie_id']
    movie_title = target_movie['title']
    console.print(f"\n[success]Selected Movie: [bold white]{movie_title}[/bold white][/success]")
    
    # 4. Get Rating value
    try:
        rating_input = input("Enter rating (1.0 to 5.0 stars, e.g. 5.0): ").strip()
        rating_val = float(rating_input) if rating_input else 5.0
        if rating_val < 1.0 or rating_val > 5.0:
            console.print("[red]Rating out of bounds. Assigning 5.0 stars.[/red]")
            rating_val = 5.0
    except ValueError:
        rating_val = 5.0
        
    # 5. Insert rating and retrain model in-memory
    console.print("\n[yellow]⚙ Simulating rating submission & re-fitting models in memory...[/yellow]")
    
    # Append or update training set
    if recommender.train_ratings is not None:
        mask = (recommender.train_ratings['user_id'] == user_id) & (recommender.train_ratings['movie_id'] == movie_id)
        if mask.any():
            recommender.train_ratings.loc[mask, 'rating'] = rating_val
        else:
            new_row = {
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': rating_val,
                'timestamp': int(time.time())
            }
            recommender.train_ratings = pd.concat([recommender.train_ratings, pd.DataFrame([new_row])], ignore_index=True)
            
        # Re-fit Collaborative Filtering
        if recommender.cf_model:
            recommender.cf_model.fit(recommender.train_ratings)
            
    console.print("[success]✔ In-memory Collaborative Filtering model retrained successfully![/success]")
    
    # 6. Fetch recommendations AFTER rating
    console.print("\n[info]Fetching updated hybrid recommendations...[/info]")
    recs_after = recommender.get_recommendations(user_id=user_id, top_n=5, method='hybrid')
    
    # 7. Print side-by-side comparison tables
    table_before = Table(title="Baseline Predictions (Before Rating)", header_style="bold red")
    table_before.add_column("Rank", justify="center")
    table_before.add_column("Movie Title", style="bold white")
    table_before.add_column("Score", style="yellow")
    
    for i, r in enumerate(recs_before):
        table_before.add_row(str(i+1), r['title'], f"{r['predicted_rating']:.2f}")
        
    table_after = Table(title=f"Updated Predictions (After {rating_val}★ for {movie_title})", header_style="bold green")
    table_after.add_column("Rank", justify="center")
    table_after.add_column("Movie Title", style="bold white")
    table_after.add_column("Score", style="yellow")
    
    for i, r in enumerate(recs_after):
        table_after.add_row(str(i+1), r['title'], f"{r['predicted_rating']:.2f}")
        
    console.print("\n")
    console.print(Columns([table_before, table_after]))
    console.print("\n[success]✔ CLI Simulation step complete! Recommendation weights successfully shifted.[/success]\n")
    input("Press Enter to return to main menu...")

def main():
    print_banner()
    
    # 1. Check for data and generate if missing
    if not (os.path.exists(config.RATINGS_CSV) and 
            os.path.exists(config.MOVIES_CSV) and 
            os.path.exists(config.USERS_CSV)):
        console.print("[yellow]⚠ Sample raw datasets not found! Launching synthetic generator...[/yellow]")
        from create_sample_data import generate_data
        generate_data()
    else:
        console.print("[green]✔ Existing raw movie and ratings datasets verified.[/green]")
        
    # 2. Check for trained models
    if not (os.path.exists(config.USER_SIMILARITY_PKL) and 
            os.path.exists(config.SVD_MODEL_PKL) and
            os.path.exists(config.ENCODED_DATA_PKL)):
        console.print("[yellow]⚠ Cache or trained models missing. Executing ML optimization pipeline...[/yellow]")
        from train import train_and_evaluate
        train_and_evaluate()
    else:
        console.print("[green]✔ Pre-trained SVD matrices and user similarity weights verified.[/green]\n")
        
    # 3. Load Engine for CLI Demo
    from src.recommendation import HybridRecommender
    recommender = HybridRecommender()
    
    # 4. Interactive Controller Menu Loop
    while True:
        console.print("\n")
        console.print(Panel(
            "[bold white]Select CineMatch Engine Operations Mode[/bold white]\n\n"
            "[bold cyan][1][/bold cyan] Launch Premium Web Dashboard (Auto-opens website)\n"
            "[bold purple][2][/bold purple] Run Interactive CLI Rating Simulator\n"
            "[bold yellow][3][/bold yellow] View Target User Recommendation Showcase (CLI)\n"
            "[bold red][4][/bold red] Shutdown Engine Console",
            title="★ CineMatch AI Control Console ★",
            border_style="magenta",
            expand=False
        ))
        
        try:
            choice = input("\nEnter choice number (1-4, default 1): ").strip()
            if not choice:
                choice = '1'
        except KeyboardInterrupt:
            console.print("\n[red]Exiting. Good bye![/red]")
            sys.exit(0)
            
        if choice == '1':
            console.print("\n")
            console.print(Panel("[bold green]🚀 Launching Premium Flask Sandbox Web Dashboard...[/bold green]", border_style="green"))
            
            # Boot Web Application with Auto-Open Browser Trigger
            import webbrowser
            import threading
            
            def open_browser():
                time.sleep(1.5)
                webbrowser.open(f"http://{config.FLASK_HOST}:{config.FLASK_PORT}")

            threading.Thread(target=open_browser, daemon=True).start()

            from app.app import app as flask_app
            flask_app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=False)
            break
            
        elif choice == '2':
            run_cli_simulator(recommender)
            
        elif choice == '3':
            show_recommendation_showcase(recommender)
            
        elif choice == '4':
            console.print("[red]Shutting down engine console. Good bye![/red]")
            sys.exit(0)
        else:
            console.print("[red]Invalid choice. Please select 1, 2, 3 or 4.[/red]")

if __name__ == "__main__":
    main()
