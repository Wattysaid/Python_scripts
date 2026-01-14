"""
CSV to JSON Converter
---------------------
Convert CSV files to JSON format with options.
"""

import pandas as pd
import json
import sys

def ensure_packages():
    try:
        import pandas
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])

def convert_csv_to_json(file_path, orient='records'):
    df = pd.read_csv(file_path)
    
    if orient == 'records':
        json_data = df.to_json(orient='records', indent=2)
    elif orient == 'values':
        json_data = df.to_json(orient='values', indent=2)
    elif orient == 'index':
        json_data = df.to_json(orient='index', indent=2)
    
    return json_data