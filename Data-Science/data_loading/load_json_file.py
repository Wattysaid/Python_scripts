"""
JSON File Loader
----------------
Simple functions to load JSON files with nested structure support.
"""

import pandas as pd
import json


def detect_json_structure(file_path):
    """Detect the structure of a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        f.seek(0)
        
        # Check if JSON Lines format
        try:
            json.loads(first_line)
            second_line = f.readline().strip()
            if second_line:
                try:
                    json.loads(second_line)
                    return 'jsonl'
                except:
                    pass
        except:
            pass
        
        # Regular JSON
        f.seek(0)
        data = json.load(f)
        if isinstance(data, list):
            return 'array'
        elif isinstance(data, dict):
            return 'object'
    
    return 'unknown'


def flatten_json(nested_json, prefix=''):
    """Flatten a nested JSON object."""
    flat_dict = {}
    
    if isinstance(nested_json, dict):
        for key, value in nested_json.items():
            new_key = f"{prefix}{key}" if prefix else key
            if isinstance(value, dict):
                flat_dict.update(flatten_json(value, f"{new_key}_"))
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    for i, item in enumerate(value):
                        flat_dict.update(flatten_json(item, f"{new_key}_{i}_"))
                else:
                    flat_dict[new_key] = value
            else:
                flat_dict[new_key] = value
    
    return flat_dict


def load_json(file_path, orient='records', flatten=False):
    """
    Load a JSON file into a DataFrame.
    
    Parameters:
    -----------
    file_path : str - Path to the JSON file
    orient : str - JSON orientation ('records', 'columns', 'index', 'split', 'values')
    flatten : bool - Whether to flatten nested structures
    
    Returns:
    --------
    pandas.DataFrame
    """
    structure = detect_json_structure(file_path)
    
    if structure == 'jsonl':
        df = pd.read_json(file_path, lines=True)
    else:
        try:
            df = pd.read_json(file_path, orient=orient)
        except ValueError:
            # Try loading as raw JSON and normalizing
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and flatten:
                # Flatten nested structure
                if any(isinstance(v, (dict, list)) for v in data.values()):
                    flat_data = flatten_json(data)
                    df = pd.DataFrame([flat_data])
                else:
                    df = pd.DataFrame([data])
            elif isinstance(data, list):
                if flatten and data and isinstance(data[0], dict):
                    flat_records = [flatten_json(record) for record in data]
                    df = pd.DataFrame(flat_records)
                else:
                    df = pd.DataFrame(data)
            else:
                df = pd.json_normalize(data)
    
    return df


def preview_data(df):
    """Display preview of the dataframe."""
    print(f"\n{'='*60}")
    print("DATA PREVIEW")
    print(f"{'='*60}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")
    print(f"\nFirst 5 rows:")
    print(df.head())