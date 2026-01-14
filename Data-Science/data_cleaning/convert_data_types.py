"""
Data Type Converter
-------------------
Simple functions to convert and standardize data types.
"""

import pandas as pd
from datetime import datetime


def convert_to_numeric(df, columns):
    """Convert columns to numeric, handling errors."""
    df_copy = df.copy()
    for col in columns:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    return df_copy


def convert_to_datetime(df, columns, date_format=None):
    """Convert columns to datetime."""
    df_copy = df.copy()
    for col in columns:
        if date_format:
            df_copy[col] = pd.to_datetime(df_copy[col], format=date_format, errors='coerce')
        else:
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
    return df_copy


def convert_to_category(df, columns):
    """Convert columns to categorical."""
    df_copy = df.copy()
    for col in columns:
        df_copy[col] = df_copy[col].astype('category')
    return df_copy