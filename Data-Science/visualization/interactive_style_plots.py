"""
Interactive-Style and Real-Time Visualizations
----------------------------------------------
Functions for creating interactive-style plots and real-time data visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
import seaborn as sns
from datetime import datetime, timedelta

def create_dashboard_style_plot(data_dict, title=None):
    """Create multi-panel dashboard-style visualization."""
    fig = plt.figure(figsize=(20, 12))
    
    # Define a sophisticated grid layout
    gs = fig.add_gridspec(4, 6, height_ratios=[1, 1.5, 1, 1], 
                         width_ratios=[1, 1, 1, 1, 1, 1])
    
    # Color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 1. Main KPI boxes (top row)
    kpi_ax = fig.add_subplot(gs[0, :])
    kpi_ax.axis('off')
    
    kpis = list(data_dict.keys())[:6]  # First 6 metrics as KPIs
    box_width = 0.15
    
    for i, kpi in enumerate(kpis):
        value = data_dict[kpi] if isinstance(data_dict[kpi], (int, float)) else len(data_dict[kpi])
        x_pos = i * 0.16 + 0.05
        
        # Create KPI box
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((x_pos, 0.2), box_width, 0.6,
                            boxstyle="round,pad=0.02",
                            facecolor=colors[i % len(colors)],
                            alpha=0.7, edgecolor='black')
        kpi_ax.add_patch(box)
        
        # Add text
        kpi_ax.text(x_pos + box_width/2, 0.7, f'{value:,.0f}' if isinstance(value, (int, float)) else f'{value}',
                   ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        kpi_ax.text(x_pos + box_width/2, 0.3, kpi.replace('_', ' ').title(),
                   ha='center', va='center', fontsize=10, color='white')
    
    kpi_ax.set_xlim(0, 1)
    kpi_ax.set_ylim(0, 1)
    kpi_ax.set_title('Key Performance Indicators', fontsize=14, fontweight='bold', pad=20)
    
    # 2. Main trend chart (second row, spans 4 columns)
    trend_ax = fig.add_subplot(gs[1, :4])
    
    # Find time series data
    time_series_key = None
    for key, value in data_dict.items():
        if isinstance(value, (list, np.ndarray, pd.Series)) and len(value) > 1:
            time_series_key = key
            break
    
    if time_series_key:
        ts_data = data_dict[time_series_key]
        x_vals = range(len(ts_data))
        
        # Create multiple trend lines with confidence intervals
        trend_ax.plot(x_vals, ts_data, linewidth=3, color=colors[0], label='Actual')
        trend_ax.fill_between(x_vals, ts_data, alpha=0.3, color=colors[0])
        
        # Add moving average if enough data
        if len(ts_data) > 7:
            ma_7 = pd.Series(ts_data).rolling(window=7).mean()
            trend_ax.plot(x_vals, ma_7, '--', linewidth=2, color=colors[1], label='7-period MA')
        
        # Add trend line
        z = np.polyfit(x_vals, ts_data, 1)
        p = np.poly1d(z)
        trend_ax.plot(x_vals, p(x_vals), ':', linewidth=2, color=colors[2], label='Trend')
        
        trend_ax.set_title(f'{time_series_key.replace("_", " ").title()} Trend Analysis', 
                          fontweight='bold')
        trend_ax.grid(True, alpha=0.3)
        trend_ax.legend()
    else:
        trend_ax.text(0.5, 0.5, 'No time series data available', 
                     ha='center', va='center', transform=trend_ax.transAxes)
        trend_ax.set_title('Trend Analysis', fontweight='bold')
    
    # 3. Gauge chart (second row, right side)
    gauge_ax = fig.add_subplot(gs[1, 4:])
    
    # Create a gauge for the first numeric value
    numeric_keys = [k for k, v in data_dict.items() if isinstance(v, (int, float))]
    if numeric_keys:
        gauge_key = numeric_keys[0]
        gauge_value = data_dict[gauge_key]
        
        # Assume max value for gauge (can be customized)
        max_val = gauge_value * 1.5 if gauge_value > 0 else 100
        
        # Create semicircle gauge
        theta = np.linspace(0, np.pi, 100)
        r = 1
        
        # Background arc
        gauge_ax.plot(r * np.cos(theta), r * np.sin(theta), 'lightgray', linewidth=10)
        
        # Value arc
        progress = min(gauge_value / max_val, 1.0)
        progress_theta = np.linspace(0, np.pi * progress, 100)
        color = colors[0] if progress < 0.8 else 'red'
        gauge_ax.plot(r * np.cos(progress_theta), r * np.sin(progress_theta), 
                     color, linewidth=10)
        
        # Add needle
        needle_angle = np.pi * (1 - progress)
        gauge_ax.plot([0, 0.8 * np.cos(needle_angle)], [0, 0.8 * np.sin(needle_angle)], 
                     'black', linewidth=4)
        
        # Add text
        gauge_ax.text(0, -0.3, f'{gauge_value:,.0f}', ha='center', va='center', 
                     fontsize=16, fontweight='bold')
        gauge_ax.text(0, -0.5, gauge_key.replace('_', ' ').title(), ha='center', va='center', 
                     fontsize=10)
    
    gauge_ax.set_xlim(-1.2, 1.2)
    gauge_ax.set_ylim(-0.6, 1.2)
    gauge_ax.set_aspect('equal')
    gauge_ax.axis('off')
    gauge_ax.set_title('Performance Gauge', fontweight='bold')
    
    # 4. Distribution plots (third row)
    dist_ax1 = fig.add_subplot(gs[2, :2])
    dist_ax2 = fig.add_subplot(gs[2, 2:4])
    dist_ax3 = fig.add_subplot(gs[2, 4:])
    
    # Find array-like data for distributions
    array_keys = [k for k, v in data_dict.items() if isinstance(v, (list, np.ndarray, pd.Series))]
    
    if len(array_keys) >= 1:
        data1 = data_dict[array_keys[0]]
        dist_ax1.hist(data1, bins=20, alpha=0.7, color=colors[0], edgecolor='black')
        dist_ax1.set_title(f'{array_keys[0].replace("_", " ").title()} Distribution')
        dist_ax1.grid(True, alpha=0.3)
    
    if len(array_keys) >= 2:
        data2 = data_dict[array_keys[1]]
        dist_ax2.boxplot(data2)
        dist_ax2.set_title(f'{array_keys[1].replace("_", " ").title()} Box Plot')
        dist_ax2.grid(True, alpha=0.3)
    
    if len(array_keys) >= 3:
        data3 = data_dict[array_keys[2]]
        if len(data3) > 1:
            dist_ax3.plot(data3, marker='o', linewidth=2, color=colors[2])
            dist_ax3.set_title(f'{array_keys[2].replace("_", " ").title()} Trend')
            dist_ax3.grid(True, alpha=0.3)
    
    # 5. Summary statistics (bottom row)
    stats_ax = fig.add_subplot(gs[3, :])
    stats_ax.axis('off')
    
    # Calculate summary statistics
    summary_text = "Data Summary:\n\n"
    for key, value in list(data_dict.items())[:10]:  # Limit to first 10 for space
        if isinstance(value, (int, float)):
            summary_text += f"{key.replace('_', ' ').title()}: {value:,.2f}\n"
        elif isinstance(value, (list, np.ndarray, pd.Series)):
            arr = np.array(value)
            summary_text += f"{key.replace('_', ' ').title()}: Mean={np.mean(arr):.2f}, Std={np.std(arr):.2f}\n"
        else:
            summary_text += f"{key.replace('_', ' ').title()}: {len(str(value))} chars\n"
    
    stats_ax.text(0.05, 0.95, summary_text, transform=stats_ax.transAxes, 
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle(title or 'Interactive-Style Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_real_time_simulation(data_length=100, update_interval=100):
    """Create a real-time data simulation visualization."""
    # Initialize data
    x_data = []
    y_data = []
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # Lines for different metrics
    line1, = ax1.plot([], [], 'b-', linewidth=2, label='Metric 1')
    line2, = ax1.plot([], [], 'r-', linewidth=2, label='Metric 2')
    ax1.set_xlim(0, data_length)
    ax1.set_ylim(-2, 2)
    ax1.set_title('Real-Time Signal Monitoring')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram that updates
    ax2.set_title('Data Distribution (Live)')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(0, 20)
    
    # Scrolling buffer display
    line3, = ax3.plot([], [], 'g-', linewidth=2)
    ax3.set_xlim(0, 50)  # Show last 50 points
    ax3.set_ylim(-3, 3)
    ax3.set_title('Recent Data (Scrolling)')
    ax3.grid(True, alpha=0.3)
    
    # Statistics display
    ax4.axis('off')
    ax4.set_title('Live Statistics')
    stats_text = ax4.text(0.1, 0.8, '', transform=ax4.transAxes, fontsize=12,
                         verticalalignment='top')
    
    def update_simulation(frame):
        # Generate new data points
        if len(x_data) >= data_length:
            x_data.pop(0)
            y_data.pop(0)
        
        x_data.append(frame)
        # Simulate real-time data with some patterns
        noise = np.random.normal(0, 0.1)
        trend = 0.01 * frame
        seasonal = 0.5 * np.sin(2 * np.pi * frame / 20)
        y_data.append(trend + seasonal + noise)
        
        # Generate second metric
        y_data2 = [np.sin(x * 0.1) + np.random.normal(0, 0.05) for x in x_data]
        
        # Update main plot
        line1.set_data(x_data, y_data)
        line2.set_data(x_data, y_data2)
        
        # Update histogram
        ax2.clear()
        if len(y_data) > 10:
            ax2.hist(y_data, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_title('Data Distribution (Live)')
            ax2.set_xlim(-3, 3)
        
        # Update scrolling plot (last 50 points)
        recent_x = x_data[-50:] if len(x_data) > 50 else x_data
        recent_y = y_data[-50:] if len(y_data) > 50 else y_data
        line3.set_data(range(len(recent_y)), recent_y)
        
        if len(recent_y) > 0:
            ax3.set_xlim(0, len(recent_y))
            ax3.set_ylim(min(recent_y) - 0.5, max(recent_y) + 0.5)
        
        # Update statistics
        if len(y_data) > 0:
            current_stats = f"""
            Live Data Statistics:
            
            Current Value: {y_data[-1]:.3f}
            Mean: {np.mean(y_data):.3f}
            Std Dev: {np.std(y_data):.3f}
            Min: {np.min(y_data):.3f}
            Max: {np.max(y_data):.3f}
            
            Data Points: {len(y_data)}
            Latest Trend: {np.polyfit(range(len(y_data)), y_data, 1)[0]:.4f}
            
            Time: {datetime.now().strftime('%H:%M:%S')}
            """
            stats_text.set_text(current_stats)
        
        return line1, line2, line3, stats_text
    
    # Create animation
    ani = animation.FuncAnimation(fig, update_simulation, frames=range(data_length),
                                 interval=update_interval, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.suptitle('Real-Time Data Simulation Dashboard', fontsize=14, fontweight='bold')
    
    return fig, ani

def create_comparison_dashboard(datasets_dict, title=None):
    """Create side-by-side comparison dashboard."""
    n_datasets = len(datasets_dict)
    fig = plt.figure(figsize=(6 * n_datasets, 12))
    
    dataset_names = list(datasets_dict.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, n_datasets))
    
    # Create subplots for each comparison type
    gs = fig.add_gridspec(4, n_datasets, height_ratios=[1, 1.5, 1, 1])
    
    # 1. Summary statistics row
    for i, (name, data) in enumerate(datasets_dict.items()):
        ax = fig.add_subplot(gs[0, i])
        ax.axis('off')
        
        if isinstance(data, (list, np.ndarray, pd.Series)):
            stats_text = f"""
            {name} Summary:
            
            Count: {len(data)}
            Mean: {np.mean(data):.2f}
            Median: {np.median(data):.2f}
            Std: {np.std(data):.2f}
            Min: {np.min(data):.2f}
            Max: {np.max(data):.2f}
            """
        else:
            stats_text = f"{name}:\n{data}"
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor=colors[i], alpha=0.7))
    
    # 2. Distribution comparison
    ax_dist = fig.add_subplot(gs[1, :])
    
    for i, (name, data) in enumerate(datasets_dict.items()):
        if isinstance(data, (list, np.ndarray, pd.Series)):
            ax_dist.hist(data, bins=20, alpha=0.6, label=name, 
                        color=colors[i], edgecolor='black')
    
    ax_dist.set_title('Distribution Comparison')
    ax_dist.legend()
    ax_dist.grid(True, alpha=0.3)
    
    # 3. Box plots comparison
    for i, (name, data) in enumerate(datasets_dict.items()):
        ax = fig.add_subplot(gs[2, i])
        if isinstance(data, (list, np.ndarray, pd.Series)):
            ax.boxplot(data, patch_artist=True, 
                      boxprops=dict(facecolor=colors[i], alpha=0.7))
            ax.set_title(f'{name} Distribution')
            ax.grid(True, alpha=0.3)
    
    # 4. Time series comparison (if applicable)
    ax_time = fig.add_subplot(gs[3, :])
    
    for i, (name, data) in enumerate(datasets_dict.items()):
        if isinstance(data, (list, np.ndarray, pd.Series)):
            ax_time.plot(data, label=name, color=colors[i], linewidth=2, alpha=0.8)
    
    ax_time.set_title('Time Series Comparison')
    ax_time.legend()
    ax_time.grid(True, alpha=0.3)
    
    plt.suptitle(title or 'Multi-Dataset Comparison Dashboard', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig