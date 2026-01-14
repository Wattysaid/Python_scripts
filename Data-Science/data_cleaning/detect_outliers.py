"""
Outlier Detector
----------------
Simple script to detect and handle outliers using statistical methods.

"""

import pandas as pd
import numpy as np
import sys

def detect_outliers_iqr(df, column):
    """Detect outliers using Interquartile Range method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound


def detect_outliers_zscore(df, column, threshold=3):
    """Detect outliers using Z-score method."""
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    outliers = df[z_scores > threshold]
    return outliers