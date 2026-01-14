"""
Categorical Encoder
------------------
Simple functions to encode categorical variables.
"""

import pandas as pd
import numpy as np


def one_hot_encode(df, columns):
    """Apply one-hot encoding to categorical columns."""
    return pd.get_dummies(df, columns=columns, dummy_na=False)


def label_encode(df, columns):
    """Apply label encoding to categorical columns."""
    df_encoded = df.copy()
    
    for col in columns:
        unique_values = df[col].dropna().unique()
        value_map = {value: idx for idx, value in enumerate(unique_values)}
        df_encoded[col] = df[col].map(value_map)
    
    return df_encoded


def ordinal_encode(df, column, order):
    """Apply ordinal encoding with specified order."""
    df_encoded = df.copy()
    value_map = {value: idx for idx, value in enumerate(order)}
    df_encoded[column] = df[column].map(value_map)
    return df_encoded