"""
Correlation Analysis
--------------------
Simple functions to analyze correlations between variables.
"""

import pandas as pd
import numpy as np


def find_correlations(df, threshold=0.5, method='pearson'):
    """Find strong correlations between numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        return None, None
    
    corr_matrix = numeric_df.corr(method=method)
    
    # Find strong correlations
    strong_corrs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                strong_corrs.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Correlation': corr_val,
                    'Abs Correlation': abs(corr_val)
                })
    
    return corr_matrix, pd.DataFrame(strong_corrs).sort_values('Abs Correlation', ascending=False)