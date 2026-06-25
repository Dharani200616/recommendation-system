import os
import pickle
import pandas as pd
from rich.console import Console
from rich.theme import Theme

# Set up beautiful theme for terminal logs
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red",
    "success": "bold green",
    "highlight": "bold yellow"
})

console = Console(theme=custom_theme)

def load_pkl(path):
    """Load a serialized object from a pickle file."""
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pkl(obj, path):
    """Serialize and save an object to a pickle file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def print_section(title, character="="):
    """Print a clean visual section header to the terminal."""
    width = 60
    console.print(f"\n[bold cyan]{character * width}[/bold cyan]")
    console.print(f"[bold white]{title.center(width)}[/bold white]")
    console.print(f"[bold cyan]{character * width}[/bold cyan]\n")

def get_rich_console():
    """Access the configured global rich console."""
    return console
