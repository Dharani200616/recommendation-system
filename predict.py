import argparse
from rich.table import Table
from rich.console import Console
from src.utils import get_rich_console, print_section
from src.recommendation import HybridRecommender

console = get_rich_console()

def main():
    parser = argparse.ArgumentParser(description="Query the Recommendation System Engine from CLI.")
    parser.add_argument("--user", type=int, default=1, help="ID of the user (e.g. 1 to 200)")
    parser.add_argument("--method", type=str, choices=['cf', 'svd', 'hybrid'], default='hybrid', help="Recommendation algorithm")
    parser.add_argument("--top_n", type=int, default=10, help="Number of recommendations to fetch")
    parser.add_argument("--genre", type=str, default=None, help="Filter recommendations by genre (e.g. Sci-Fi, Romance)")
    
    args = parser.parse_args()
    
    print_section(f"RECOMMENDATION QUERY FOR USER {args.user}", "◆")
    
    # Initialize recommender
    recommender = HybridRecommender()
    
    # Fetch recommendations
    recommendations = recommender.get_recommendations(
        user_id=args.user,
        top_n=args.top_n,
        method=args.method,
        genre=args.genre
    )
    
    if not recommendations:
        console.print("[warning]No recommendations found! Check if models are trained and files exist.[/warning]")
        return
        
    # Render recommendations table
    table = Table(
        title=f"Top {args.top_n} {args.method.upper()} Recommendations" + (f" (Genre: {args.genre})" if args.genre else ""),
        header_style="bold magenta"
    )
    table.add_column("Rank", style="dim cyan", justify="center")
    table.add_column("Movie Title", style="bold white", justify="left")
    table.add_column("Genres", style="green", justify="left")
    table.add_column("Predicted Score", style="bold yellow", justify="right")
    
    for i, rec in enumerate(recommendations):
        score_val = rec.get('predicted_rating', 0.0)
        # Visual star scale
        stars = "★" * int(round(score_val)) + "☆" * (5 - int(round(score_val)))
        table.add_row(
            str(i + 1),
            rec.get('title', 'Unknown'),
            rec.get('genres', 'N/A'),
            f"{score_val:.2f} ({stars})"
        )
        
    console.print(table)
    console.print(f"\n[info]Retrieved {len(recommendations)} recommendations. Query complete![/info]\n")

if __name__ == "__main__":
    main()
