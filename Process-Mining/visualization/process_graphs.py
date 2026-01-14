"""
Process Visualization
--------------------
Functions to visualize process mining results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import networkx as nx

def plot_process_flow_diagram(df_relations, start_activities, end_activities, min_frequency=1):
    """Create process flow diagram with statistics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Filter relations by frequency
    filtered_relations = {k: v for k, v in df_relations.items() if v >= min_frequency}
    
    # Create directed graph
    G = nx.DiGraph()
    
    for (source, target), frequency in filtered_relations.items():
        G.add_edge(source, target, weight=frequency)
    
    # Calculate node positions
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Draw nodes
    node_sizes = []
    node_colors = []
    for node in G.nodes():
        if node in start_activities:
            node_colors.append('lightgreen')
            node_sizes.append(1000)
        elif node in end_activities:
            node_colors.append('lightcoral')
            node_sizes.append(1000)
        else:
            node_colors.append('lightblue')
            node_sizes.append(800)
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_sizes, ax=ax1)
    
    # Draw edges with thickness based on frequency
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [weight/max_weight * 5 + 0.5 for weight in edge_weights]
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, 
                          edge_color='gray', alpha=0.7, ax=ax1)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax1)
    
    # Add edge labels for high-frequency flows
    edge_labels = {(u, v): str(d['weight']) 
                  for u, v, d in G.edges(data=True) if d['weight'] >= min_frequency * 2}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, ax=ax1)
    
    ax1.set_title('Process Flow Diagram')
    ax1.axis('off')
    
    # Statistics summary
    total_relations = sum(filtered_relations.values())
    unique_activities = len(set([act for relation in filtered_relations.keys() 
                               for act in relation]))
    
    stats_text = f"""
    Process Flow Statistics:
    
    Unique Activities: {unique_activities}
    Total Relations: {len(filtered_relations)}
    Total Flow Frequency: {total_relations}
    
    Start Activities:
    {chr(10).join([f'  {act}: {freq}' for act, freq in start_activities.items()])}
    
    End Activities:
    {chr(10).join([f'  {act}: {freq}' for act, freq in end_activities.items()])}
    
    Most Frequent Flows:
    """
    
    top_flows = sorted(filtered_relations.items(), key=lambda x: x[1], reverse=True)[:5]
    for (source, target), freq in top_flows:
        stats_text += f"\n  {source} → {target}: {freq}"
    
    ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan'))
    ax2.axis('off')
    ax2.set_title('Flow Statistics')
    
    plt.tight_layout()
    return fig

def plot_activity_frequency(activity_counts, title=None):
    """Plot activity frequency analysis."""
    activities = list(activity_counts.keys())
    frequencies = list(activity_counts.values())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart
    bars = ax1.bar(activities, frequencies)
    ax1.set_title('Activity Frequency')
    ax1.set_xlabel('Activities')
    ax1.set_ylabel('Frequency')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, freq in zip(bars, frequencies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(frequencies)*0.01,
                str(freq), ha='center', va='bottom')
    
    # Statistical analysis
    freq_stats = f"""
    Activity Frequency Analysis:
    
    Total Activities: {len(activities)}
    Total Occurrences: {sum(frequencies)}
    
    Frequency Statistics:
    Mean: {np.mean(frequencies):.2f}
    Median: {np.median(frequencies):.2f}
    Std: {np.std(frequencies):.2f}
    Min: {min(frequencies)}
    Max: {max(frequencies)}
    
    Most Frequent Activities:
    """
    
    sorted_activities = sorted(activity_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (activity, freq) in enumerate(sorted_activities[:5]):
        percentage = (freq / sum(frequencies)) * 100
        freq_stats += f"\n  {i+1}. {activity}: {freq} ({percentage:.1f}%)"
    
    ax2.text(0.1, 0.9, freq_stats, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax2.axis('off')
    ax2.set_title('Frequency Statistics')
    
    plt.suptitle(title or 'Activity Frequency Analysis')
    plt.tight_layout()
    return fig