# Social Network Analysis
import pandas as pd
import sys
from collections import defaultdict, Counter

def build_network(df):
    # Build network from handoffs between resources
    network = defaultdict(list)
    
    for case_id, group in df.groupby('case_id'):
        if 'resource' in df.columns:
            resources = group.sort_values('timestamp')['resource'].tolist()
        else:
            resources = group.sort_values('timestamp')['activity'].tolist()
        
        # Create edges between consecutive resources
        for i in range(len(resources) - 1):
            source = resources[i]
            target = resources[i + 1]
            if source != target:  # Only if different resources
                network[source].append(target)
    
    return network

def calculate_centrality(network):
    # Simple centrality measures
    centrality = {}
    all_nodes = set(network.keys())
    for edges in network.values():
        all_nodes.update(edges)
    
    for node in all_nodes:
        # Out-degree centrality
        out_degree = len(network[node])
        
        # In-degree centrality
        in_degree = sum(1 for edges in network.values() if node in edges)
        
        centrality[node] = {
            'out_degree': out_degree,
            'in_degree': in_degree,
            'total_degree': out_degree + in_degree
        }
    
    return centrality