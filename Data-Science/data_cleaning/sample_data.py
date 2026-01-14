"""
Data Sampler
------------
Create random samples from large datasets.
"""

import pandas as pd
import sys

def ensure_packages():
    try:
        import pandas
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])

def sample_data(df, method='random', size=1000):
    if method == 'random':
        sample = df.sample(n=min(size, len(df)), random_state=42)
    elif method == 'first':
        sample = df.head(size)
    elif method == 'last':
        sample = df.tail(size)
    elif method == 'systematic':
        step = len(df) // size
        sample = df.iloc[::max(step, 1)][:size]
    
    return sample