"""
Performance Visualization
------------------------
Functions to visualize process performance metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats

def plot_case_duration_analysis(case_durations, title=None):
    """Analyze and plot case duration distributions."""
    durations_hours = [d.total_seconds() / 3600 for d in case_durations.values()]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # Histogram
    ax1.hist(durations_hours, bins=30, alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(durations_hours), color='red', linestyle='--', 
               label=f'Mean: {np.mean(durations_hours):.2f}h')
    ax1.axvline(np.median(durations_hours), color='green', linestyle='--', 
               label=f'Median: {np.median(durations_hours):.2f}h')
    ax1.set_title('Case Duration Distribution')
    ax1.set_xlabel('Duration (hours)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # Box plot
    box_plot = ax2.boxplot(durations_hours, patch_artist=True)
    ax2.set_title('Case Duration Box Plot')
    ax2.set_ylabel('Duration (hours)')
    
    # Q-Q plot for normality
    stats.probplot(durations_hours, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot (Normal Distribution)')
    
    # Statistical summary
    Q1 = np.percentile(durations_hours, 25)
    Q3 = np.percentile(durations_hours, 75)
    IQR = Q3 - Q1
    outliers = [d for d in durations_hours if d < Q1 - 1.5*IQR or d > Q3 + 1.5*IQR]
    
    # Test for normality
    shapiro_stat, shapiro_p = stats.shapiro(durations_hours[:5000])  # Shapiro limited to 5000
    
    duration_stats = f"""
    Case Duration Analysis:
    
    Total Cases: {len(case_durations)}
    
    Duration Statistics (hours):
    Mean: {np.mean(durations_hours):.2f}
    Median: {np.median(durations_hours):.2f}
    Std: {np.std(durations_hours):.2f}
    Min: {min(durations_hours):.2f}
    Max: {max(durations_hours):.2f}
    
    Quartiles:
    Q1: {Q1:.2f}
    Q3: {Q3:.2f}
    IQR: {IQR:.2f}
    
    Outliers: {len(outliers)} ({len(outliers)/len(durations_hours)*100:.1f}%)
    
    Distribution Tests:
    Shapiro-Wilk p-value: {shapiro_p:.4f}
    Normal Distribution: {"Yes" if shapiro_p > 0.05 else "No"}
    
    Performance Insights:
    Fastest Case: {min(durations_hours):.2f}h
    Slowest Case: {max(durations_hours):.2f}h
    95th Percentile: {np.percentile(durations_hours, 95):.2f}h
    """
    
    ax4.text(0.1, 0.9, duration_stats, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightsteelblue'))
    ax4.axis('off')
    ax4.set_title('Statistical Summary')
    
    plt.suptitle(title or 'Case Duration Analysis')
    plt.tight_layout()
    return fig

def plot_throughput_analysis(event_log, time_col='timestamp', case_col='case_id', title=None):
    """Analyze process throughput over time."""
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(event_log[time_col]):
        event_log[time_col] = pd.to_datetime(event_log[time_col])
    
    # Calculate daily throughput (cases started and completed)
    daily_starts = event_log.groupby(case_col)[time_col].min().dt.date.value_counts().sort_index()
    daily_completions = event_log.groupby(case_col)[time_col].max().dt.date.value_counts().sort_index()
    
    # Create common date range
    all_dates = pd.date_range(start=min(daily_starts.index.min(), daily_completions.index.min()),
                             end=max(daily_starts.index.max(), daily_completions.index.max()))
    
    daily_starts = daily_starts.reindex(all_dates, fill_value=0)
    daily_completions = daily_completions.reindex(all_dates, fill_value=0)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # Throughput over time
    ax1.plot(daily_starts.index, daily_starts.values, label='Cases Started', marker='o', alpha=0.7)
    ax1.plot(daily_completions.index, daily_completions.values, label='Cases Completed', marker='s', alpha=0.7)
    ax1.set_title('Daily Throughput')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Number of Cases')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Cumulative throughput
    cumulative_starts = daily_starts.cumsum()
    cumulative_completions = daily_completions.cumsum()
    
    ax2.plot(cumulative_starts.index, cumulative_starts.values, label='Cumulative Starts')
    ax2.plot(cumulative_completions.index, cumulative_completions.values, label='Cumulative Completions')
    ax2.set_title('Cumulative Throughput')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative Cases')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    
    # Work in progress (WIP)
    wip = cumulative_starts - cumulative_completions
    ax3.plot(wip.index, wip.values, color='red', linewidth=2)
    ax3.set_title('Work in Progress (WIP)')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Cases in Progress')
    ax3.axhline(y=wip.mean(), color='blue', linestyle='--', label=f'Average WIP: {wip.mean():.1f}')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    
    # Throughput statistics
    throughput_stats = f"""
    Throughput Analysis:
    
    Analysis Period:
    Start: {all_dates[0].strftime('%Y-%m-%d')}
    End: {all_dates[-1].strftime('%Y-%m-%d')}
    Duration: {len(all_dates)} days
    
    Case Statistics:
    Total Started: {daily_starts.sum()}
    Total Completed: {daily_completions.sum()}
    Currently in Progress: {wip.iloc[-1]}
    
    Daily Averages:
    Cases Started: {daily_starts.mean():.2f}
    Cases Completed: {daily_completions.mean():.2f}
    
    Peak Performance:
    Max Started (day): {daily_starts.max()}
    Max Completed (day): {daily_completions.max()}
    Max WIP: {wip.max()}
    Min WIP: {wip.min()}
    
    Efficiency Metrics:
    Completion Rate: {daily_completions.sum()/daily_starts.sum()*100:.1f}%
    Avg WIP: {wip.mean():.1f}
    WIP Std: {wip.std():.1f}
    """
    
    ax4.text(0.1, 0.9, throughput_stats, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen'))
    ax4.axis('off')
    ax4.set_title('Throughput Statistics')
    
    plt.suptitle(title or 'Process Throughput Analysis')
    plt.tight_layout()
    return fig