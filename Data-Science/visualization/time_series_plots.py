"""
Time Series Plotting Functions
-----------------------------
Time series visualization with trend analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks

def plot_time_series_analysis(data, date_col=None, value_col=None, title=None):
    """Comprehensive time series analysis plot."""
    if isinstance(data, pd.DataFrame):
        if date_col and value_col:
            ts_data = data.set_index(date_col)[value_col]
        else:
            ts_data = data.iloc[:, 0]  # First column
    else:
        ts_data = pd.Series(data)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # Original time series
    ax1.plot(ts_data.index, ts_data.values, linewidth=1.5)
    ax1.set_title('Time Series Data')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    
    # Add trend line
    x_numeric = np.arange(len(ts_data))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, ts_data.values)
    trend_line = slope * x_numeric + intercept
    ax1.plot(ts_data.index, trend_line, color='red', linestyle='--', 
             label=f'Trend (slope: {slope:.4f})')
    ax1.legend()
    
    # Moving averages
    if len(ts_data) > 20:
        ma_7 = ts_data.rolling(window=7).mean()
        ma_30 = ts_data.rolling(window=min(30, len(ts_data)//3)).mean()
        
        ax2.plot(ts_data.index, ts_data.values, alpha=0.6, label='Original')
        ax2.plot(ma_7.index, ma_7.values, label='7-period MA')
        ax2.plot(ma_30.index, ma_30.values, label=f'{min(30, len(ts_data)//3)}-period MA')
        ax2.set_title('Moving Averages')
        ax2.legend()
    else:
        ax2.plot(ts_data.index, ts_data.values)
        ax2.set_title('Time Series (Insufficient data for MA)')
    
    # Autocorrelation
    lags = min(40, len(ts_data)//4)
    autocorr = [ts_data.autocorr(lag=i) for i in range(1, lags+1)]
    ax3.bar(range(1, lags+1), autocorr)
    ax3.set_title('Autocorrelation')
    ax3.set_xlabel('Lag')
    ax3.set_ylabel('Autocorrelation')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Statistical summary
    # Detect seasonality
    peaks, _ = find_peaks(ts_data.values, height=np.mean(ts_data.values))
    
    ts_summary = f"""
    Time Series Statistics:
    
    Count: {len(ts_data)}
    Mean: {np.mean(ts_data.values):.3f}
    Std: {np.std(ts_data.values):.3f}
    Min: {np.min(ts_data.values):.3f}
    Max: {np.max(ts_data.values):.3f}
    
    Trend Analysis:
    Slope: {slope:.6f}
    R-squared: {r_value**2:.4f}
    P-value: {p_value:.4f}
    
    Pattern Analysis:
    Peaks detected: {len(peaks)}
    Variance: {np.var(ts_data.values):.3f}
    Range: {np.max(ts_data.values) - np.min(ts_data.values):.3f}
    """
    
    ax4.text(0.1, 0.9, ts_summary, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightsteelblue'))
    ax4.axis('off')
    ax4.set_title('Statistical Summary')
    
    plt.suptitle(title or 'Time Series Analysis')
    plt.tight_layout()
    return fig

def plot_seasonal_decomposition(data, period=12, title=None):
    """Plot seasonal decomposition of time series."""
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    if isinstance(data, pd.DataFrame):
        ts_data = data.iloc[:, 0]
    else:
        ts_data = pd.Series(data)
    
    # Perform decomposition
    decomposition = seasonal_decompose(ts_data, model='additive', period=period)
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    
    # Original series
    axes[0, 0].plot(decomposition.observed)
    axes[0, 0].set_title('Original Time Series')
    
    # Trend
    axes[1, 0].plot(decomposition.trend)
    axes[1, 0].set_title('Trend Component')
    
    # Seasonal
    axes[2, 0].plot(decomposition.seasonal)
    axes[2, 0].set_title('Seasonal Component')
    
    # Residual
    axes[3, 0].plot(decomposition.resid)
    axes[3, 0].set_title('Residual Component')
    
    # Statistics for each component
    components = ['Original', 'Trend', 'Seasonal', 'Residual']
    component_data = [decomposition.observed, decomposition.trend, 
                     decomposition.seasonal, decomposition.resid]
    
    for i, (comp_name, comp_data) in enumerate(zip(components, component_data)):
        comp_stats = f"""
        {comp_name} Statistics:
        
        Mean: {np.nanmean(comp_data.values):.4f}
        Std: {np.nanstd(comp_data.values):.4f}
        Min: {np.nanmin(comp_data.values):.4f}
        Max: {np.nanmax(comp_data.values):.4f}
        
        Non-null count: {comp_data.count()}
        Variance: {np.nanvar(comp_data.values):.4f}
        """
        
        axes[i, 1].text(0.1, 0.9, comp_stats, transform=axes[i, 1].transAxes, 
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f'{comp_name} Statistics')
    
    plt.suptitle(title or f'Seasonal Decomposition (Period={period})')
    plt.tight_layout()
    return fig, decomposition