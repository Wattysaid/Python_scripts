"""
Network and Graph Visualizations
-------------------------------
Functions for creating network graphs and relationship visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import seaborn as sns

def create_social_network_graph(connections_df, source_col='source', target_col='target', 
                                weight_col=None, title=None):
    """Create social network graph with community detection."""
    # Create graph
    G = nx.Graph()
    
    # Add edges
    if weight_col and weight_col in connections_df.columns:
        for _, row in connections_df.iterrows():
            G.add_edge(row[source_col], row[target_col], weight=row[weight_col])
    else:
        for _, row in connections_df.iterrows():
            G.add_edge(row[source_col], row[target_col])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Main network graph
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Node sizes based on degree centrality
    node_sizes = [G.degree(node) * 100 + 300 for node in G.nodes()]
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', 
                          alpha=0.7, ax=ax1)
    
    if weight_col:
        # Edge widths based on weights
        edges = G.edges(data=True)
        weights = [edge[2].get('weight', 1) for edge in edges]
        max_weight = max(weights) if weights else 1
        edge_widths = [weight/max_weight * 5 + 0.5 for weight in weights]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, ax=ax1)
    else:
        nx.draw_networkx_edges(G, pos, alpha=0.6, ax=ax1)
    
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax1)
    ax1.set_title('Network Graph')
    ax1.axis('off')
    
    # 2. Degree distribution
    degrees = [G.degree(node) for node in G.nodes()]
    ax2.hist(degrees, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_title('Degree Distribution')
    ax2.set_xlabel('Degree')
    ax2.set_ylabel('Frequency')
    ax2.axvline(np.mean(degrees), color='red', linestyle='--', 
               label=f'Mean: {np.mean(degrees):.2f}')
    ax2.legend()
    
    # 3. Network metrics visualization
    ax3.axis('off')
    
    # Calculate network metrics
    try:
        clustering_coeff = nx.average_clustering(G)
        avg_path_length = nx.average_shortest_path_length(G) if nx.is_connected(G) else 'Disconnected'
        diameter = nx.diameter(G) if nx.is_connected(G) else 'Disconnected'
        density = nx.density(G)
        
        # Centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        # Most central nodes
        top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
        top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
        
    except:
        clustering_coeff = 0
        avg_path_length = 'Error'
        diameter = 'Error'
        density = nx.density(G)
        top_degree = []
        top_betweenness = []
    
    network_stats = f"""
    Network Statistics:
    
    Nodes: {G.number_of_nodes()}
    Edges: {G.number_of_edges()}
    Density: {density:.3f}
    Avg Clustering: {clustering_coeff:.3f}
    Avg Path Length: {avg_path_length}
    Diameter: {diameter}
    
    Top Degree Centrality:
    """
    
    for node, centrality in top_degree:
        network_stats += f"\n  {node}: {centrality:.3f}"
    
    network_stats += "\n\nTop Betweenness:\n"
    for node, centrality in top_betweenness:
        network_stats += f"\n  {node}: {centrality:.3f}"
    
    ax3.text(0.1, 0.9, network_stats, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray'))
    ax3.set_title('Network Metrics')
    
    # 4. Community detection (if possible)
    ax4.set_title('Community Structure')
    try:
        communities = nx.community.greedy_modularity_communities(G)
        colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
        
        for i, community in enumerate(communities):
            nx.draw_networkx_nodes(G, pos, nodelist=community, 
                                 node_color=[colors[i]], alpha=0.8, ax=ax4)
        
        nx.draw_networkx_edges(G, pos, alpha=0.4, ax=ax4)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax4)
        
        modularity = nx.community.modularity(G, communities)
        ax4.text(0.02, 0.98, f'Communities: {len(communities)}\nModularity: {modularity:.3f}',
                transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    except:
        ax4.text(0.5, 0.5, 'Community detection\nnot available', 
                ha='center', va='center', transform=ax4.transAxes)
    
    ax4.axis('off')
    
    plt.suptitle(title or 'Social Network Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_hierarchy_tree(hierarchy_df, parent_col='parent', child_col='child', 
                         value_col=None, title=None):
    """Create hierarchical tree visualization."""
    # Create directed graph
    G = nx.DiGraph()
    
    # Add edges
    for _, row in hierarchy_df.iterrows():
        if pd.notna(row[parent_col]) and pd.notna(row[child_col]):
            weight = row[value_col] if value_col and value_col in hierarchy_df.columns else 1
            G.add_edge(row[parent_col], row[child_col], weight=weight)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Tree layout
    try:
        # Find root nodes (nodes with no predecessors)
        root_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]
        
        if root_nodes:
            # Use hierarchical layout
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        else:
            # Fallback to spring layout
            pos = nx.spring_layout(G)
    except:
        pos = nx.spring_layout(G)
    
    # Calculate node levels for coloring
    node_levels = {}
    if root_nodes:
        for root in root_nodes:
            try:
                lengths = nx.single_source_shortest_path_length(G, root)
                for node, level in lengths.items():
                    node_levels[node] = level
            except:
                pass
    
    # Color nodes by level
    if node_levels:
        max_level = max(node_levels.values())
        colors = [plt.cm.viridis(node_levels.get(node, 0) / max_level) for node in G.nodes()]
    else:
        colors = 'lightblue'
    
    # Node sizes based on out-degree (number of children)
    node_sizes = [G.out_degree(node) * 200 + 300 for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=node_sizes, 
                          alpha=0.8, ax=ax1)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          arrowsize=20, alpha=0.6, ax=ax1)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax1)
    
    ax1.set_title('Hierarchy Tree')
    ax1.axis('off')
    
    # 2. Hierarchy statistics
    ax2.axis('off')
    
    # Calculate hierarchy metrics
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_roots = len(root_nodes) if 'root_nodes' in locals() else 0
    
    # Count nodes at each level
    level_counts = Counter(node_levels.values()) if node_levels else {}
    
    # Find leaf nodes (nodes with no successors)
    leaf_nodes = [node for node in G.nodes() if G.out_degree(node) == 0]
    
    # Calculate branching factor
    branching_factors = [G.out_degree(node) for node in G.nodes() if G.out_degree(node) > 0]
    avg_branching_factor = np.mean(branching_factors) if branching_factors else 0
    
    hierarchy_stats = f"""
    Hierarchy Statistics:
    
    Total Nodes: {n_nodes}
    Total Edges: {n_edges}
    Root Nodes: {n_roots}
    Leaf Nodes: {len(leaf_nodes)}
    
    Levels in Hierarchy: {max(node_levels.values()) + 1 if node_levels else 'Unknown'}
    Avg Branching Factor: {avg_branching_factor:.2f}
    
    Nodes per Level:
    """
    
    for level in sorted(level_counts.keys()):
        hierarchy_stats += f"\n  Level {level}: {level_counts[level]} nodes"
    
    if value_col and value_col in hierarchy_df.columns:
        total_value = hierarchy_df[value_col].sum()
        hierarchy_stats += f"\n\nTotal Value: {total_value:,.2f}"
    
    ax2.text(0.1, 0.9, hierarchy_stats, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax2.set_title('Hierarchy Analysis')
    
    plt.suptitle(title or 'Hierarchical Structure Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_flow_diagram(flow_data, source_col='source', target_col='target', 
                       value_col='value', title=None):
    """Create Sankey-style flow diagram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Network representation of flows
    G = nx.DiGraph()
    
    for _, row in flow_data.iterrows():
        G.add_edge(row[source_col], row[target_col], weight=row[value_col])
    
    # Calculate positions
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Node sizes based on total flow (in + out)
    node_flows = {}
    for node in G.nodes():
        in_flow = sum([G[source][node]['weight'] for source in G.predecessors(node)])
        out_flow = sum([G[node][target]['weight'] for target in G.successors(node)])
        node_flows[node] = max(in_flow, out_flow)
    
    max_flow = max(node_flows.values()) if node_flows else 1
    node_sizes = [node_flows.get(node, 0) / max_flow * 1000 + 200 for node in G.nodes()]
    
    # Edge widths based on flow values
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [weight / max_weight * 10 + 1 for weight in edge_weights]
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightcoral',
                          alpha=0.7, ax=ax1)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='darkblue',
                          arrows=True, arrowsize=20, alpha=0.6, ax=ax1)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax1)
    
    # Add edge labels for major flows
    edge_labels = {}
    for u, v in G.edges():
        weight = G[u][v]['weight']
        if weight >= max_weight * 0.5:  # Only show major flows
            edge_labels[(u, v)] = f'{weight:,.0f}'
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax1)
    
    ax1.set_title('Flow Network Diagram')
    ax1.axis('off')
    
    # 2. Flow analysis
    ax2.axis('off')
    
    # Calculate flow statistics
    total_flow = flow_data[value_col].sum()
    unique_sources = flow_data[source_col].nunique()
    unique_targets = flow_data[target_col].nunique()
    unique_nodes = len(set(flow_data[source_col]) | set(flow_data[target_col]))
    
    # Top flows
    top_flows = flow_data.nlargest(5, value_col)
    
    # Source and target analysis
    source_totals = flow_data.groupby(source_col)[value_col].sum().sort_values(ascending=False)
    target_totals = flow_data.groupby(target_col)[value_col].sum().sort_values(ascending=False)
    
    flow_stats = f"""
    Flow Analysis:
    
    Total Flow Volume: {total_flow:,.0f}
    Number of Connections: {len(flow_data)}
    Unique Nodes: {unique_nodes}
    Source Nodes: {unique_sources}
    Target Nodes: {unique_targets}
    
    Top Sources:
    """
    
    for source, total in source_totals.head(3).items():
        percentage = (total / total_flow) * 100
        flow_stats += f"\n  {source}: {total:,.0f} ({percentage:.1f}%)"
    
    flow_stats += "\n\nTop Targets:"
    for target, total in target_totals.head(3).items():
        percentage = (total / total_flow) * 100
        flow_stats += f"\n  {target}: {total:,.0f} ({percentage:.1f}%)"
    
    flow_stats += "\n\nLargest Flows:"
    for _, flow in top_flows.head(3).iterrows():
        percentage = (flow[value_col] / total_flow) * 100
        flow_stats += f"\n  {flow[source_col]} → {flow[target_col]}: {flow[value_col]:,.0f} ({percentage:.1f}%)"
    
    ax2.text(0.1, 0.9, flow_stats, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan'))
    ax2.set_title('Flow Statistics')
    
    plt.suptitle(title or 'Flow Analysis Diagram', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig