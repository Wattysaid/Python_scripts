"""
Distribution Analysis
--------------------
Functions for analyzing data distributions.
"""

import pandas as pd
import numpy as np
from scipy import stats

def test_normality(data, alpha=0.05):
    """Test if data follows normal distribution."""
    shapiro_stat, shapiro_p = stats.shapiro(data)
    kstest_stat, kstest_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
    
    result = {
        'shapiro_statistic': shapiro_stat,
        'shapiro_p_value': shapiro_p,
        'shapiro_normal': shapiro_p > alpha,
        'kstest_statistic': kstest_stat,
        'kstest_p_value': kstest_p,
        'kstest_normal': kstest_p > alpha,
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data),
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data)
    }
    return result

def fit_distribution(data, distributions=['norm', 'expon', 'gamma', 'beta']):
    """Fit multiple distributions and find best fit."""
    results = {}
    
    for dist_name in distributions:
        try:
            dist = getattr(stats, dist_name)
            params = dist.fit(data)
            ks_stat, p_value = stats.kstest(data, dist.cdf, args=params)
            
            results[dist_name] = {
                'parameters': params,
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'aic': 2 * len(params) - 2 * np.sum(dist.logpdf(data, *params))
            }
        except:
            continue
    
    # Find best fit by highest p-value
    if results:
        best_fit = max(results.keys(), key=lambda x: results[x]['p_value'])
        results['best_fit'] = best_fit
    
    return results

def describe_distribution(data):
    """Get comprehensive distribution statistics."""
    result = {
        'count': len(data),
        'mean': np.mean(data),
        'median': np.median(data),
        'mode': stats.mode(data)[0][0],
        'std': np.std(data),
        'variance': np.var(data),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data),
        'min': np.min(data),
        'max': np.max(data),
        'range': np.max(data) - np.min(data),
        'q1': np.percentile(data, 25),
        'q3': np.percentile(data, 75),
        'iqr': np.percentile(data, 75) - np.percentile(data, 25)
    }
    return result