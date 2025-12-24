import pandas as pd
import json
from pathlib import Path

def read_csv(path: str, **kwargs):
    """
    Read CSV into DataFrame safely.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, **kwargs)

def write_csv(df, path: str, index: bool = False, **kwargs):
    """
    Write DataFrame to CSV safely.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, **kwargs)

def read_json(path: str):
    """
    Read JSON file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(data, path: str, indent: int = 4):
    """
    Write data to JSON file safely.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def ensure_dir(path: str):
    """
    Ensure a directory exists.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path