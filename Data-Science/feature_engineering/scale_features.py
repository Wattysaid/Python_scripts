"""
Feature Scaler
---------------
Simple functions to scale numerical features.
"""

import pandas as pd
import numpy as np


def min_max_scale(data):
    """Apply min-max scaling (0-1 range)."""
    return (data - data.min()) / (data.max() - data.min())


def standard_scale(data):
    """Apply standard scaling (z-score normalization)."""
    return (data - data.mean()) / data.std()


def robust_scale(data):
    """Apply robust scaling using median and IQR."""
    median = data.median()
    q75 = data.quantile(0.75)
    q25 = data.quantile(0.25)
    iqr = q75 - q25
    return (data - median) / iqr


def scale_columns(df, columns, method='standard'):
    """Scale specified columns using chosen method."""
    df_scaled = df.copy()
    
    for col in columns:
        if method == 'minmax':
            df_scaled[col] = min_max_scale(df[col])
        elif method == 'standard':
            df_scaled[col] = standard_scale(df[col])
        elif method == 'robust':
            df_scaled[col] = robust_scale(df[col])
    
    return df_scaled