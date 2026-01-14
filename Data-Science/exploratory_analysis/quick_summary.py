"""
Quick Data Summary
------------------
Generate instant data overview and summary statistics.
"""

import pandas as pd
import sys

def ensure_packages():
    try:
        import pandas
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])

def quick_summary(file_path):
    df = pd.read_csv(file_path)
    print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(f"Missing: {df.isnull().sum().sum()} values")
    print("\nColumn Types:")
    print(df.dtypes.value_counts())
    print("\nFirst 5 rows:")
    print(df.head())
    return df