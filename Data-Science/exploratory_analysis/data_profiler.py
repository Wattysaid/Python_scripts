"""
Data Profiler
-------------
Simple functions to create comprehensive data profiles.
"""

import pandas as pd
import numpy as np


def profile_column(df, column):
    """Create detailed profile for a single column."""
    col_data = df[column]
    
    profile = {
        'Column': column,
        'Data Type': str(col_data.dtype),
        'Non-Null Count': col_data.notna().sum(),
        'Null Count': col_data.isnull().sum(),
        'Null %': (col_data.isnull().sum() / len(col_data)) * 100,
        'Unique Values': col_data.nunique(),
        'Unique %': (col_data.nunique() / len(col_data)) * 100
    }
    
    if pd.api.types.is_numeric_dtype(col_data):
        profile.update({
            'Mean': col_data.mean(),
            'Median': col_data.median(),
            'Min': col_data.min(),
            'Max': col_data.max(),
            'Std Dev': col_data.std()
        })
    else:
        top_value = col_data.value_counts().index[0] if not col_data.value_counts().empty else None
        profile.update({
            'Most Frequent': top_value,
            'Frequency': col_data.value_counts().iloc[0] if not col_data.value_counts().empty else 0
        })
    
    return profile


def create_data_profile(df):
    """Create comprehensive profile for entire dataset."""
    profiles = []
    
    for column in df.columns:
        profile = profile_column(df, column)
        profiles.append(profile)
    
    return pd.DataFrame(profiles)