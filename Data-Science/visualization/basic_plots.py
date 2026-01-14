"""
Basic Plotting Functions
-----------------------
Basic visualization functions with statistical summaries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def plot_histogram_with_stats(data, bins=30, title=None):
    """Create histogram with statistical summary."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    ax1.hist(data, bins=bins, alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(data), color='red', linestyle='--', label=f'Mean: {np.mean(data):.2f}')
    ax1.axvline(np.median(data), color='green', linestyle='--', label=f'Median: {np.median(data):.2f}')
    ax1.set_title(title or 'Histogram with Statistics')
    ax1.legend()
    
    # Statistical summary
    stats_text = f"""
    Count: {len(data)}
    Mean: {np.mean(data):.3f}
    Median: {np.median(data):.3f}
    Std: {np.std(data):.3f}
    Min: {np.min(data):.3f}
    Max: {np.max(data):.3f}
    Skewness: {stats.skew(data):.3f}
    Kurtosis: {stats.kurtosis(data):.3f}
    """
    
    ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray'))
    ax2.axis('off')
    ax2.set_title('Statistical Summary')
    
    plt.tight_layout()
    return fig

def plot_scatter_with_correlation(x, y, title=None):
    """Create scatter plot with correlation analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Scatter plot with regression line
    ax1.scatter(x, y, alpha=0.6)
    
    # Add regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax1.plot(x, p(x), "r--", alpha=0.8)
    
    ax1.set_title(title or 'Scatter Plot with Regression Line')
    ax1.set_xlabel('X Variable')
    ax1.set_ylabel('Y Variable')
    
    # Correlation analysis
    corr_pearson, p_pearson = stats.pearsonr(x, y)
    corr_spearman, p_spearman = stats.spearmanr(x, y)
    
    corr_text = f"""
    Pearson Correlation: {corr_pearson:.3f}
    Pearson P-value: {p_pearson:.3f}
    
    Spearman Correlation: {corr_spearman:.3f}
    Spearman P-value: {p_spearman:.3f}
    
    R-squared: {corr_pearson**2:.3f}
    
    Regression Equation:
    y = {z[0]:.3f}x + {z[1]:.3f}
    """
    
    ax2.text(0.1, 0.9, corr_text, transform=ax2.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax2.axis('off')
    ax2.set_title('Correlation Analysis')
    
    plt.tight_layout()
    return fig

def plot_boxplot_with_outliers(data, title=None):
    """Create boxplot with outlier analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Boxplot
    box_plot = ax1.boxplot(data, patch_artist=True)
    ax1.set_title(title or 'Boxplot with Outlier Analysis')
    
    # Outlier analysis
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    outlier_text = f"""
    Quartiles:
    Q1 (25%): {Q1:.3f}
    Q2 (50%): {np.median(data):.3f}
    Q3 (75%): {Q3:.3f}
    
    IQR: {IQR:.3f}
    
    Outlier Bounds:
    Lower: {lower_bound:.3f}
    Upper: {upper_bound:.3f}
    
    Number of Outliers: {len(outliers)}
    Outlier Percentage: {len(outliers)/len(data)*100:.1f}%
    """
    
    ax2.text(0.1, 0.9, outlier_text, transform=ax2.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax2.axis('off')
    ax2.set_title('Outlier Analysis')
    
    plt.tight_layout()
    return fig