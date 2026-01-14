"""
Basic Statistics Generator
--------------------------
Simple functions to generate descriptive statistics.
"""

import pandas as pd
import numpy as np


def generate_basic_stats(df):
    """Generate basic descriptive statistics."""
    stats = {
        'Shape': f"{df.shape[0]} rows × {df.shape[1]} columns",
        'Memory Usage': f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB",
        'Missing Values': df.isnull().sum().sum(),
        'Duplicate Rows': df.duplicated().sum()
    }
    return stats


def numeric_statistics(df):
    """Generate statistics for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return None
    
    stats = df[numeric_cols].describe()
    
    # Add additional statistics
    additional_stats = pd.DataFrame({
        'skewness': df[numeric_cols].skew(),
        'kurtosis': df[numeric_cols].kurtosis(),
        'variance': df[numeric_cols].var()
    }).T
    
    return pd.concat([stats, additional_stats])


def categorical_statistics(df):
    """Generate statistics for categorical columns."""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) == 0:
        return None
    
    stats = []
    for col in cat_cols:
        col_stats = {
            'Column': col,
            'Unique Values': df[col].nunique(),
            'Most Frequent': df[col].mode().iloc[0] if not df[col].mode().empty else 'None',
            'Frequency': df[col].value_counts().iloc[0] if not df[col].empty else 0,
            'Missing': df[col].isnull().sum()
        }
        stats.append(col_stats)
    
    return pd.DataFrame(stats)