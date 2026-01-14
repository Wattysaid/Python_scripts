"""
Missing Values Handler
----------------------
Simple script to detect and handle missing values in datasets.

"""

import pandas as pd
import sys

def analyze_missing(df):
    """Analyze missing values in DataFrame."""
    missing_info = df.isnull().sum()
    missing_percent = (missing_info / len(df)) * 100
    
    result = pd.DataFrame({
        'Missing Count': missing_info,
        'Missing %': missing_percent
    })
    return result[result['Missing Count'] > 0].sort_values('Missing %', ascending=False)


def handle_missing(df, method='drop', fill_value=None):
    """Handle missing values using specified method."""
    if method == 'drop':
        return df.dropna()
    elif method == 'fill':
        return df.fillna(fill_value)
    elif method == 'ffill':
        return df.fillna(method='ffill')
    elif method == 'bfill':
        return df.fillna(method='bfill')
    elif method == 'mean':
        numeric_cols = df.select_dtypes(include=[float, int]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        return df
    else:
        return df