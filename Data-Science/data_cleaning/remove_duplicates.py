"""
Duplicate Detector and Remover
-------------------------------
Simple script to find and remove duplicate rows in datasets.

"""

import pandas as pd
import sys

def find_duplicates(df, subset=None):
    """Find duplicate rows in DataFrame."""
    duplicates = df[df.duplicated(subset=subset, keep=False)]
    return duplicates.sort_values(by=subset if subset else df.columns.tolist())


def remove_duplicates(df, subset=None, keep='first'):
    """Remove duplicate rows from DataFrame."""
    return df.drop_duplicates(subset=subset, keep=keep)