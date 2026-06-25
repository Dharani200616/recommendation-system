import os
import random
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import config

console = Console()

# Movie seed data
MOVIE_TEMPLATES = [
    # Sci-Fi / Action
    ("Inception", "Sci-Fi|Action|Thriller", 2010),
    ("The Matrix", "Sci-Fi|Action", 1999),
    ("Interstellar", "Sci-Fi|Drama|Adventure", 2014),
    ("Blade Runner 2049", "Sci-Fi|Action|Drama", 2017),
    ("Star Wars: A New Hope", "Sci-Fi|Action|Adventure", 1977),
    ("The Dark Knight", "Action|Crime|Drama", 2008),
    ("Avengers: Endgame", "Action|Sci-Fi|Adventure", 2019),
    ("Terminator 2: Judgment Day", "Action|Sci-Fi", 1991),
    ("Spider-Man: Into the Spider-Verse", "Animation|Action|Adventure", 2018),
    ("Mad Max: Fury Road", "Action|Sci-Fi|Adventure", 2015),
    ("Gladiator", "Action|Drama", 2000),
    ("Die Hard", "Action|Thriller", 1988),
    ("Aliens", "Sci-Fi|Action|Horror", 1986),
    ("Jurassic Park", "Sci-Fi|Adventure|Action", 1993),
    ("Avatar", "Sci-Fi|Action|Adventure", 2009),
    # Drama / Classic
    ("The Shawshank Redemption", "Drama", 1994),
    ("The Godfather", "Drama|Crime", 1972),
    ("Pulp Fiction", "Drama|Crime", 1994),
    ("Forrest Gump", "Drama|Romance", 1994),
    ("Fight Club", "Drama|Thriller", 1999),
    ("Good Will Hunting", "Drama|Romance", 1997),
    ("The Green Mile", "Drama|Fantasy|Crime", 1999),
    ("Schindler's List", "Drama|History", 1993),
    ("Whiplash", "Drama|Music", 2014),
    ("The Social Network", "Drama", 2010),
    ("Parasite", "Drama|Thriller|Comedy", 2019),
    ("American Beauty", "Drama", 1999),
    ("Casablanca", "Drama|Romance|War", 1942),
    ("12 Angry Men", "Drama", 1957),
    ("Citizen Kane", "Drama|Mystery", 1941),
    # Comedy / Lighthearted
    ("Superbad", "Comedy", 2007),
    ("The Hangover", "Comedy", 2009),
    ("Monty Python and the Holy Grail", "Comedy|Fantasy", 1975),
    ("Office Space", "Comedy", 1999),
    ("Step Brothers", "Comedy", 2008),
    ("Zoolander", "Comedy", 2001),
    ("Anchor-Man", "Comedy", 2004),
    ("Shaun of the Dead", "Comedy|Horror", 2004),
    ("Hot Fuzz", "Comedy|Action|Mystery", 2007),
    ("Groundhog Day", "Comedy|Romance|Fantasy", 1993),
    ("The Big Lebowski", "Comedy|Crime", 1998),
    ("Mean Girls", "Comedy", 2004),
    ("Dumb and Dumber", "Comedy", 1994),
    ("Ferris Bueller's Day Off", "Comedy", 1986),
    ("Airplane!", "Comedy", 1980),
    # Romance
    ("Titanic", "Romance|Drama", 1997),
    ("The Notebook", "Romance|Drama", 2004),
    ("Pride & Prejudice", "Romance|Drama", 2005),
    ("La La Land", "Romance|Drama|Musical", 2016),
    ("Before Sunrise", "Romance|Drama", 1995),
    ("Before Sunset", "Romance|Drama", 2004),
    ("Notting Hill", "Romance|Comedy", 1999),
    ("Crazy Rich Asians", "Romance|Comedy", 2018),
    ("About Time", "Romance|Drama|Fantasy", 2013),
    ("500 Days of Summer", "Romance|Comedy|Drama", 2009),
    ("Eternal Sunshine of the Spotless Mind", "Romance|Sci-Fi|Drama", 2004),
    ("Amélie", "Romance|Comedy", 2001),
    ("The Princess Bride", "Romance|Adventure|Comedy", 1987),
    ("A Star Is Born", "Romance|Drama|Music", 2018),
    ("Pretty Woman", "Romance|Comedy", 1990),
    # Animation / Kids / Family
    ("Toy Story", "Animation|Adventure|Comedy|Family", 1995),
    ("Finding Nemo", "Animation|Adventure|Comedy|Family", 2003),
    ("The Lion King", "Animation|Adventure|Drama|Family", 1994),
    ("Spirited Away", "Animation|Fantasy|Adventure|Family", 2001),
    ("Wall-E", "Animation|Sci-Fi|Family", 2008),
    ("Up", "Animation|Adventure|Comedy|Family", 2009),
    ("Shrek", "Animation|Adventure|Comedy|Family|Fantasy", 2001),
    ("Inside Out", "Animation|Adventure|Comedy|Family|Drama", 2015),
    ("Monsters, Inc.", "Animation|Comedy|Family|Fantasy", 2001),
    ("Ratatouille", "Animation|Comedy|Family", 2007),
    ("Coco", "Animation|Adventure|Family|Music", 2017),
    ("How to Train Your Dragon", "Animation|Adventure|Family|Fantasy", 2010),
    ("My Neighbor Totoro", "Animation|Fantasy|Family", 1988),
    ("Aladdin", "Animation|Adventure|Comedy|Family|Musical", 1992),
    ("Beauty and the Beast", "Animation|Fantasy|Family|Musical|Romance", 1991),
    # Horror / Thriller
    ("The Shining", "Horror|Drama|Thriller", 1980),
    ("Get Out", "Horror|Mystery|Thriller", 2017),
    ("Psycho", "Horror|Thriller|Mystery", 1960),
    ("A Quiet Place", "Horror|Sci-Fi|Thriller", 2018),
    ("The Silence of the Lambs", "Thriller|Crime|Drama", 1991),
    ("Se7en", "Thriller|Crime|Drama|Mystery", 1995),
    ("The Sixth Sense", "Thriller|Mystery|Drama", 1999),
    ("Hereditary", "Horror|Mystery|Drama", 2018),
    ("The Conjuring", "Horror|Thriller", 2013),
    ("A Nightmare on Elm Street", "Horror", 1984),
    ("Alien", "Horror|Sci-Fi", 1979),
    ("The Exorcist", "Horror", 1973),
    ("Saw", "Horror|Mystery|Thriller", 2004),
    ("Midsommar", "Horror|Mystery|Drama|Thriller", 2019),
    ("Halloween", "Horror|Thriller", 1978),
    # Fantasy / Adventure
    ("The Lord of the Rings: The Fellowship of the Ring", "Fantasy|Adventure|Action", 2001),
    ("Harry Potter and the Sorcerer's Stone", "Fantasy|Adventure|Family", 2001),
    ("The Hobbit: An Unexpected Journey", "Fantasy|Adventure|Action", 2012),
    ("Pirates of the Caribbean: The Curse of the Black Pearl", "Adventure|Action|Fantasy", 2003),
    ("The Chronicles of Narnia: The Lion, the Witch and the Wardrobe", "Adventure|Family|Fantasy", 2005),
    ("Stardust", "Fantasy|Romance|Adventure", 2007),
    ("Pan's Labyrinth", "Fantasy|Drama|War", 2006),
    ("Alice in Wonderland", "Fantasy|Adventure|Family", 2010),
    ("Jumanji", "Adventure|Family|Fantasy", 1995),
    ("Clash of the Titans", "Fantasy|Action|Adventure", 2010)
]

OCCUPATIONS = [
    "Student", "Engineer", "Artist", "Doctor", "Teacher", "Writer", 
    "Lawyer", "Executive", "Scientist", "Sales", "Technician", "Retired"
]

GENDERS = ["M", "F", "O"]

def generate_data(num_users=200):
    console.print("[bold cyan]=====================================================[/bold cyan]")
    console.print("[bold cyan]       RECOMMENDATION SYSTEM DATA GENERATOR          [/bold cyan]")
    console.print("[bold cyan]=====================================================[/bold cyan]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        # 1. Generate Movies
        task1 = progress.add_task("[yellow]Generating movie catalog...[/yellow]", total=len(MOVIE_TEMPLATES))
        movies = []
        for i, (title, genres, year) in enumerate(MOVIE_TEMPLATES):
            movies.append({
                "movie_id": i + 1,
                "title": title,
                "genres": genres,
                "release_year": year
            })
            progress.update(task1, advance=1)
        
        df_movies = pd.DataFrame(movies)
        os.makedirs(os.path.dirname(config.MOVIES_CSV), exist_ok=True)
        df_movies.to_csv(config.MOVIES_CSV, index=False)
        
        # 2. Generate Users
        task2 = progress.add_task("[yellow]Generating user profiles...[/yellow]", total=num_users)
        users = []
        # Let's assign users a hidden preference style to make the recommendation patterns clear.
        # Preference types:
        # 0: Sci-Fi / Action / Adventure
        # 1: Romance / Drama
        # 2: Animation / Kids / Family
        # 3: Horror / Thriller
        # 4: Comedy
        user_pref_profiles = []
        for i in range(num_users):
            user_id = i + 1
            age = random.choices(
                [random.randint(15, 22), random.randint(23, 35), random.randint(36, 55), random.randint(56, 75)],
                weights=[0.2, 0.5, 0.2, 0.1]
            )[0]
            gender = random.choices(GENDERS, weights=[0.48, 0.48, 0.04])[0]
            occupation = random.choice(OCCUPATIONS)
            pref_type = random.randint(0, 4)
            
            users.append({
                "user_id": user_id,
                "age": age,
                "gender": gender,
                "occupation": occupation
            })
            user_pref_profiles.append(pref_type)
            progress.update(task2, advance=1)
            
        df_users = pd.DataFrame(users)
        df_users.to_csv(config.USERS_CSV, index=False)

        # 3. Generate Ratings (Structured around preference profiles to enable ML models to discover themes)
        # We will generate between 30 and 80 ratings per user, totaling about 10,000 ratings.
        total_ratings_to_gen = num_users * 40
        task3 = progress.add_task("[yellow]Simulating user ratings...[/yellow]", total=total_ratings_to_gen)
        
        ratings = []
        
        for u_idx in range(num_users):
            user_id = u_idx + 1
            pref = user_pref_profiles[u_idx]
            
            # Select target movie indices based on preferences
            num_ratings = random.randint(30, 60)
            rated_movies = set()
            
            # Populate rated movies
            while len(rated_movies) < num_ratings:
                # 70% chance to rate a movie aligned with preference profile, 30% chance to rate a random movie
                if random.random() < 0.7:
                    # Filter templates matching pref
                    matching_indices = []
                    for m_idx, (title, genres, year) in enumerate(MOVIE_TEMPLATES):
                        g_lower = genres.lower()
                        if pref == 0 and ("sci-fi" in g_lower or "action" in g_lower):
                            matching_indices.append(m_idx + 1)
                        elif pref == 1 and ("romance" in g_lower or "drama" in g_lower) and "animation" not in g_lower:
                            matching_indices.append(m_idx + 1)
                        elif pref == 2 and ("animation" in g_lower or "family" in g_lower):
                            matching_indices.append(m_idx + 1)
                        elif pref == 3 and ("horror" in g_lower or "thriller" in g_lower):
                            matching_indices.append(m_idx + 1)
                        elif pref == 4 and "comedy" in g_lower and "animation" not in g_lower:
                            matching_indices.append(m_idx + 1)
                            
                    if matching_indices:
                        movie_id = random.choice(matching_indices)
                    else:
                        movie_id = random.randint(1, len(MOVIE_TEMPLATES))
                else:
                    movie_id = random.randint(1, len(MOVIE_TEMPLATES))
                    
                rated_movies.add(movie_id)
                
            for movie_id in rated_movies:
                # Calculate structured rating
                # Base rating depends on user preference match
                genres = MOVIE_TEMPLATES[movie_id - 1][1].lower()
                
                base_rating = 3.0
                if pref == 0 and ("sci-fi" in genres or "action" in genres):
                    base_rating = 4.2
                elif pref == 1 and ("romance" in genres or "drama" in genres) and "animation" not in genres:
                    base_rating = 4.2
                elif pref == 2 and ("animation" in genres or "family" in genres):
                    base_rating = 4.3
                elif pref == 3 and ("horror" in genres or "thriller" in genres):
                    base_rating = 4.1
                elif pref == 4 and "comedy" in genres and "animation" not in genres:
                    base_rating = 4.0
                    
                # Add random noise
                rating_val = np.clip(np.random.normal(base_rating, 0.8), 1.0, 5.0)
                # Round to half or whole star
                rating = round(rating_val * 2) / 2
                
                # Timestamp
                timestamp = 1609459200 + random.randint(0, 31536000) # Year 2021
                
                ratings.append({
                    "user_id": user_id,
                    "movie_id": movie_id,
                    "rating": rating,
                    "timestamp": timestamp
                })
                progress.update(task3, advance=1)
                
        df_ratings = pd.DataFrame(ratings)
        df_ratings.to_csv(config.RATINGS_CSV, index=False)
        
    console.print(f"[bold green]✔ Done! Generated files successfully:[/bold green]")
    console.print(f"  - Movies: [cyan]{len(df_movies)}[/cyan] records saved to [bold]{config.MOVIES_CSV}[/bold]")
    console.print(f"  - Users: [cyan]{len(df_users)}[/cyan] records saved to [bold]{config.USERS_CSV}[/bold]")
    console.print(f"  - Ratings: [cyan]{len(df_ratings)}[/cyan] ratings (mean score: [yellow]{df_ratings['rating'].mean():.2f}[/yellow]) saved to [bold]{config.RATINGS_CSV}[/bold]")
    console.print("\n[bold green]Ready for preprocessing and modeling![/bold green]\n")

if __name__ == "__main__":
    generate_data()
