import os
import yaml
import time
import json

def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, 'r', encoding = 'utf-8') as file:
        return yaml.safe_load(file)

def save_json(data: dict, path: str):
    """Save dictionary as JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def read_json(path: str) -> dict:
    """Read JSON file and return as dictionary."""
    with open(path, 'r') as f:
        return json.load(f)

def ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def format_time(seconds):
    """Format seconds into hh:mm:ss."""
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

