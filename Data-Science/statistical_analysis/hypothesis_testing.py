"""
Hypothesis Testing
-----------------
Functions for statistical hypothesis testing.
"""

import pandas as pd
import numpy as np
from scipy import stats

def ttest_one_sample(data, pop_mean, alpha=0.05):
    """Perform one-sample t-test."""
    statistic, p_value = stats.ttest_1samp(data, pop_mean)
    result = {
        'statistic': statistic,
        'p_value': p_value,
        'alpha': alpha,
        'reject_null': p_value < alpha,
        'mean': np.mean(data),
        'std': np.std(data, ddof=1),
        'n': len(data)
    }
    return result

def ttest_independent(group1, group2, alpha=0.05):
    """Perform independent samples t-test."""
    statistic, p_value = stats.ttest_ind(group1, group2)
    result = {
        'statistic': statistic,
        'p_value': p_value,
        'alpha': alpha,
        'reject_null': p_value < alpha,
        'group1_mean': np.mean(group1),
        'group2_mean': np.mean(group2),
        'group1_std': np.std(group1, ddof=1),
        'group2_std': np.std(group2, ddof=1),
        'effect_size': (np.mean(group1) - np.mean(group2)) / np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
    }
    return result

def chi_square_test(observed, expected=None, alpha=0.05):
    """Perform chi-square goodness of fit test."""
    if expected is None:
        statistic, p_value = stats.chisquare(observed)
    else:
        statistic, p_value = stats.chisquare(observed, expected)
    
    result = {
        'statistic': statistic,
        'p_value': p_value,
        'alpha': alpha,
        'reject_null': p_value < alpha,
        'degrees_freedom': len(observed) - 1
    }
    return result