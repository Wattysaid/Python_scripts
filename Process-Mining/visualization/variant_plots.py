"""
Variant Analysis Visualization
-----------------------------
Functions to visualize process variants and patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def plot_variant_analysis(traces, title=None, top_n=10):
    """Analyze and plot process variants."""
    # Convert traces to string patterns for analysis
    trace_patterns = {}
    for case_id, trace in traces.items():
        pattern = ' → '.join(trace)
        trace_patterns[case_id] = pattern
    
    # Count variant frequencies
    variant_counts = Counter(trace_patterns.values())
    top_variants = dict(variant_counts.most_common(top_n))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # Variant frequency bar chart
    variants = list(top_variants.keys())
    frequencies = list(top_variants.values())
    
    # Truncate long variant names for display
    display_variants = [v[:50] + '...' if len(v) > 50 else v for v in variants]
    
    bars = ax1.barh(range(len(display_variants)), frequencies)
    ax1.set_yticks(range(len(display_variants)))
    ax1.set_yticklabels(display_variants)
    ax1.set_title(f'Top {top_n} Process Variants')
    ax1.set_xlabel('Frequency')
    
    # Add frequency labels
    for i, (bar, freq) in enumerate(zip(bars, frequencies)):
        ax1.text(bar.get_width() + max(frequencies)*0.01, bar.get_y() + bar.get_height()/2,
                str(freq), ha='left', va='center')
    
    # Variant length distribution
    variant_lengths = [len(trace) for trace in traces.values()]
    ax2.hist(variant_lengths, bins=range(1, max(variant_lengths)+2), alpha=0.7, edgecolor='black')
    ax2.set_title('Process Variant Length Distribution')
    ax2.set_xlabel('Number of Activities')
    ax2.set_ylabel('Frequency')
    ax2.axvline(np.mean(variant_lengths), color='red', linestyle='--', 
               label=f'Mean: {np.mean(variant_lengths):.1f}')
    ax2.legend()
    
    # Cumulative frequency of variants
    total_cases = sum(variant_counts.values())
    cumulative_freq = []
    cumulative_sum = 0
    
    for freq in frequencies:
        cumulative_sum += freq
        cumulative_freq.append(cumulative_sum / total_cases * 100)
    
    ax3.plot(range(1, len(cumulative_freq)+1), cumulative_freq, marker='o')
    ax3.set_title('Cumulative Variant Coverage')
    ax3.set_xlabel('Number of Top Variants')
    ax3.set_ylabel('Cumulative Coverage (%)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=80, color='red', linestyle='--', label='80% Coverage')
    ax3.legend()
    
    # Variant analysis statistics
    unique_variants = len(variant_counts)
    total_traces = len(traces)
    
    # Calculate complexity metrics
    avg_variant_length = np.mean(variant_lengths)
    std_variant_length = np.std(variant_lengths)
    
    # Find 80% coverage point
    coverage_80_variants = next((i for i, cum_freq in enumerate(cumulative_freq, 1) 
                               if cum_freq >= 80), len(cumulative_freq))
    
    variant_stats = f"""
    Process Variant Analysis:
    
    Overview:
    Total Cases: {total_traces:,}
    Unique Variants: {unique_variants:,}
    Variant Diversity: {unique_variants/total_traces*100:.1f}%
    
    Variant Length Statistics:
    Average: {avg_variant_length:.2f} activities
    Std Dev: {std_variant_length:.2f}
    Min: {min(variant_lengths)}
    Max: {max(variant_lengths)}
    
    Top Variant Coverage:
    Most Common: {frequencies[0]} cases ({frequencies[0]/total_traces*100:.1f}%)
    Top 3: {sum(frequencies[:3])} cases ({sum(frequencies[:3])/total_traces*100:.1f}%)
    Top {min(5, len(frequencies))}: {sum(frequencies[:5])} cases ({sum(frequencies[:5])/total_traces*100:.1f}%)
    
    Complexity Metrics:
    80% Coverage: {coverage_80_variants} variants
    Complexity Ratio: {unique_variants/total_traces:.3f}
    
    Process Insights:
    {'High' if unique_variants/total_traces > 0.5 else 'Medium' if unique_variants/total_traces > 0.2 else 'Low'} Process Complexity
    {'Standard' if avg_variant_length < 10 else 'Complex'} Process Length
    """
    
    ax4.text(0.1, 0.9, variant_stats, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightpink'))
    ax4.axis('off')
    ax4.set_title('Variant Statistics')
    
    plt.suptitle(title or 'Process Variant Analysis')
    plt.tight_layout()
    return fig

def plot_activity_positions(traces, title=None):
    """Analyze positions of activities in process variants."""
    # Collect activity positions
    activity_positions = {}
    max_length = 0
    
    for case_id, trace in traces.items():
        max_length = max(max_length, len(trace))
        for position, activity in enumerate(trace):
            if activity not in activity_positions:
                activity_positions[activity] = []
            activity_positions[activity].append(position + 1)  # 1-indexed
    
    # Create position matrix for heatmap
    activities = sorted(activity_positions.keys())
    position_matrix = np.zeros((len(activities), max_length))
    
    for i, activity in enumerate(activities):
        positions = activity_positions[activity]
        for pos in positions:
            if pos <= max_length:
                position_matrix[i, pos-1] += 1
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Position heatmap
    sns.heatmap(position_matrix, 
                xticklabels=[f'Pos {i+1}' for i in range(max_length)],
                yticklabels=activities,
                cmap='YlOrRd', 
                annot=False,
                ax=ax1,
                cbar_kws={'label': 'Frequency'})
    ax1.set_title('Activity Position Frequency Heatmap')
    ax1.set_xlabel('Position in Process')
    ax1.set_ylabel('Activities')
    
    # Activity position statistics
    position_stats = f"""
    Activity Position Analysis:
    
    Process Statistics:
    Total Activities: {len(activities)}
    Max Process Length: {max_length}
    Total Cases Analyzed: {len(traces)}
    
    Position Analysis per Activity:
    """
    
    for activity in activities[:10]:  # Show first 10 activities
        positions = activity_positions[activity]
        avg_pos = np.mean(positions)
        std_pos = np.std(positions)
        min_pos = min(positions)
        max_pos = max(positions)
        frequency = len(positions)
        
        position_stats += f"""
    
    {activity}:
      Frequency: {frequency}
      Avg Position: {avg_pos:.1f}
      Std: {std_pos:.1f}
      Range: {min_pos}-{max_pos}
    """
    
    if len(activities) > 10:
        position_stats += f"\n    ... and {len(activities)-10} more activities"
    
    ax2.text(0.1, 0.9, position_stats, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan'))
    ax2.axis('off')
    ax2.set_title('Position Statistics')
    
    plt.suptitle(title or 'Activity Position Analysis')
    plt.tight_layout()
    return fig