"""
Geographic and Spatial Visualizations
------------------------------------
Functions for creating geographic maps and spatial data visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

def create_choropleth_simulation(region_data, value_col, region_col='region', title=None):
    """Create choropleth-style visualization using synthetic geographic data."""
    # Since we don't have actual geographic boundaries, we'll create a grid-based representation
    regions = region_data[region_col].unique()
    n_regions = len(regions)
    
    # Create a grid layout for regions
    grid_size = int(np.ceil(np.sqrt(n_regions)))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Grid-based choropleth
    region_values = dict(zip(region_data[region_col], region_data[value_col]))
    
    # Create value matrix for heatmap
    value_matrix = np.zeros((grid_size, grid_size))
    region_matrix = np.empty((grid_size, grid_size), dtype=object)
    
    for i, region in enumerate(regions):
        row = i // grid_size
        col = i % grid_size
        if row < grid_size and col < grid_size:
            value_matrix[row, col] = region_values.get(region, 0)
            region_matrix[row, col] = region
    
    # Create heatmap
    sns.heatmap(value_matrix, annot=region_matrix, fmt='', cmap='YlOrRd', ax=ax1,
                cbar_kws={'label': value_col.title()})
    ax1.set_title(f'Regional {value_col.title()} Distribution')
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    
    # 2. Bar chart by region
    sorted_data = region_data.sort_values(value_col, ascending=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_data)))
    
    bars = ax2.barh(range(len(sorted_data)), sorted_data[value_col], color=colors)
    ax2.set_yticks(range(len(sorted_data)))
    ax2.set_yticklabels(sorted_data[region_col])
    ax2.set_xlabel(value_col.title())
    ax2.set_title(f'{value_col.title()} by Region (Ranked)')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, sorted_data[value_col])):
        ax2.text(value + max(sorted_data[value_col]) * 0.01, i, f'{value:,.1f}', 
                va='center', fontsize=9)
    
    # 3. Statistical distribution
    ax3.hist(region_data[value_col], bins=15, alpha=0.7, edgecolor='black', color='skyblue')
    ax3.axvline(region_data[value_col].mean(), color='red', linestyle='--', 
               label=f'Mean: {region_data[value_col].mean():.2f}')
    ax3.axvline(region_data[value_col].median(), color='green', linestyle='--', 
               label=f'Median: {region_data[value_col].median():.2f}')
    ax3.set_xlabel(value_col.title())
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'{value_col.title()} Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Regional statistics
    ax4.axis('off')
    
    # Calculate regional statistics
    stats = {
        'total_regions': len(regions),
        'total_value': region_data[value_col].sum(),
        'mean_value': region_data[value_col].mean(),
        'median_value': region_data[value_col].median(),
        'std_value': region_data[value_col].std(),
        'min_value': region_data[value_col].min(),
        'max_value': region_data[value_col].max(),
        'range_value': region_data[value_col].max() - region_data[value_col].min()
    }
    
    # Find top and bottom regions
    top_region = region_data.loc[region_data[value_col].idxmax()]
    bottom_region = region_data.loc[region_data[value_col].idxmin()]
    
    stats_text = f"""
    Regional Statistics:
    
    Total Regions: {stats['total_regions']}
    Total {value_col.title()}: {stats['total_value']:,.2f}
    
    Distribution:
    Mean: {stats['mean_value']:.2f}
    Median: {stats['median_value']:.2f}
    Std Dev: {stats['std_value']:.2f}
    Range: {stats['range_value']:.2f}
    
    Extremes:
    Highest: {top_region[region_col]}
    Value: {top_region[value_col]:.2f}
    
    Lowest: {bottom_region[region_col]}
    Value: {bottom_region[value_col]:.2f}
    
    Coefficient of Variation:
    {(stats['std_value']/stats['mean_value']*100):.1f}%
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen'))
    ax4.set_title('Regional Analysis')
    
    plt.suptitle(title or f'Geographic Analysis of {value_col.title()}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_spatial_clustering_plot(coordinates_df, lat_col='latitude', lon_col='longitude', 
                                  value_col=None, cluster_col=None, title=None):
    """Create spatial clustering visualization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    x = coordinates_df[lon_col]
    y = coordinates_df[lat_col]
    
    # 1. Basic spatial plot
    if value_col and value_col in coordinates_df.columns:
        scatter = ax1.scatter(x, y, c=coordinates_df[value_col], 
                             cmap='viridis', alpha=0.6, s=50)
        plt.colorbar(scatter, ax=ax1, label=value_col.title())
        ax1.set_title(f'Spatial Distribution of {value_col.title()}')
    else:
        ax1.scatter(x, y, alpha=0.6, s=50)
        ax1.set_title('Spatial Point Distribution')
    
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.grid(True, alpha=0.3)
    
    # 2. Density plot
    try:
        ax2.hexbin(x, y, gridsize=20, cmap='Blues', mincnt=1)
        ax2.set_title('Spatial Density (Hexbin)')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
    except:
        # Fallback to regular scatter if hexbin fails
        ax2.scatter(x, y, alpha=0.6, s=50)
        ax2.set_title('Spatial Distribution')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
    
    # 3. Clusters (if available) or distance analysis
    if cluster_col and cluster_col in coordinates_df.columns:
        unique_clusters = coordinates_df[cluster_col].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster in enumerate(unique_clusters):
            mask = coordinates_df[cluster_col] == cluster
            ax3.scatter(x[mask], y[mask], c=[colors[i]], label=f'Cluster {cluster}', 
                       alpha=0.7, s=50)
        
        ax3.set_title('Spatial Clusters')
        ax3.legend()
    else:
        # Create simple distance-based analysis
        # Calculate distances from centroid
        centroid_x = x.mean()
        centroid_y = y.mean()
        
        distances = np.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
        scatter = ax3.scatter(x, y, c=distances, cmap='coolwarm', alpha=0.7, s=50)
        plt.colorbar(scatter, ax=ax3, label='Distance from Centroid')
        ax3.scatter(centroid_x, centroid_y, c='red', s=100, marker='x', label='Centroid')
        ax3.set_title('Distance from Centroid')
        ax3.legend()
    
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.grid(True, alpha=0.3)
    
    # 4. Spatial statistics
    ax4.axis('off')
    
    # Calculate spatial metrics
    centroid_x = x.mean()
    centroid_y = y.mean()
    
    # Bounding box
    min_x, max_x = x.min(), x.max()
    min_y, max_y = y.min(), y.max()
    area = (max_x - min_x) * (max_y - min_y)
    
    # Distances from centroid
    distances_from_centroid = np.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
    
    # Calculate pairwise distances for spatial analysis
    if len(coordinates_df) < 1000:  # Only for reasonable sized datasets
        coords = coordinates_df[[lon_col, lat_col]].values
        pairwise_distances = pdist(coords)
        avg_nearest_neighbor = np.mean(pairwise_distances)
    else:
        avg_nearest_neighbor = 'Too many points'
    
    spatial_stats = f"""
    Spatial Statistics:
    
    Number of Points: {len(coordinates_df)}
    
    Bounding Box:
    Longitude: {min_x:.4f} to {max_x:.4f}
    Latitude: {min_y:.4f} to {max_y:.4f}
    Area: {area:.6f} sq degrees
    
    Centroid:
    Longitude: {centroid_x:.4f}
    Latitude: {centroid_y:.4f}
    
    Dispersion:
    Mean distance from centroid: {distances_from_centroid.mean():.4f}
    Max distance from centroid: {distances_from_centroid.max():.4f}
    Std distance from centroid: {distances_from_centroid.std():.4f}
    """
    
    if isinstance(avg_nearest_neighbor, (int, float)):
        spatial_stats += f"\nAvg pairwise distance: {avg_nearest_neighbor:.4f}"
    
    if cluster_col and cluster_col in coordinates_df.columns:
        n_clusters = coordinates_df[cluster_col].nunique()
        spatial_stats += f"\n\nClusters: {n_clusters}"
    
    ax4.text(0.1, 0.9, spatial_stats, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax4.set_title('Spatial Analysis')
    
    plt.suptitle(title or 'Spatial Data Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_route_optimization_plot(locations_df, lat_col='latitude', lon_col='longitude',
                                  location_col='location', title=None):
    """Create route optimization and distance analysis visualization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    coordinates = locations_df[[lat_col, lon_col]].values
    
    # 1. Location map with connections
    ax1.scatter(locations_df[lon_col], locations_df[lat_col], 
               c='red', s=100, alpha=0.7, zorder=5)
    
    # Add location labels
    if location_col in locations_df.columns:
        for _, location in locations_df.iterrows():
            ax1.annotate(location[location_col], 
                        (location[lon_col], location[lat_col]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Add simple connecting lines (not optimized route)
    for i in range(len(coordinates) - 1):
        ax1.plot([coordinates[i][1], coordinates[i+1][1]], 
                [coordinates[i][0], coordinates[i+1][0]], 
                'b--', alpha=0.5)
    
    ax1.set_title('Location Map with Connections')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.grid(True, alpha=0.3)
    
    # 2. Distance matrix heatmap
    from scipy.spatial.distance import euclidean
    n_locations = len(locations_df)
    distance_matrix = np.zeros((n_locations, n_locations))
    
    for i in range(n_locations):
        for j in range(n_locations):
            if i != j:
                distance_matrix[i, j] = euclidean(coordinates[i], coordinates[j])
    
    labels = locations_df[location_col].tolist() if location_col in locations_df.columns else [f'Loc {i+1}' for i in range(n_locations)]
    
    sns.heatmap(distance_matrix, annot=True, fmt='.3f', 
                xticklabels=labels, yticklabels=labels,
                cmap='viridis', ax=ax2)
    ax2.set_title('Distance Matrix')
    
    # 3. Route analysis - nearest neighbor heuristic
    def nearest_neighbor_route(distance_matrix, start=0):
        n = len(distance_matrix)
        unvisited = set(range(n))
        current = start
        route = [current]
        unvisited.remove(current)
        total_distance = 0
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: distance_matrix[current][x])
            total_distance += distance_matrix[current][nearest]
            route.append(nearest)
            current = nearest
            unvisited.remove(current)
        
        # Return to start
        total_distance += distance_matrix[current][start]
        route.append(start)
        
        return route, total_distance
    
    # Calculate optimized route
    route, total_distance = nearest_neighbor_route(distance_matrix)
    
    # Plot optimized route
    route_coords = [coordinates[i] for i in route]
    route_lons = [coord[1] for coord in route_coords]
    route_lats = [coord[0] for coord in route_coords]
    
    ax3.scatter(locations_df[lon_col], locations_df[lat_col], 
               c='red', s=100, alpha=0.7, zorder=5)
    ax3.plot(route_lons, route_lats, 'g-', linewidth=2, alpha=0.7)
    
    # Add route order numbers
    for i, (lat, lon) in enumerate(zip(route_lats[:-1], route_lons[:-1])):
        ax3.annotate(str(i+1), (lon, lat), xytext=(10, 10), 
                    textcoords='offset points', fontsize=10, 
                    bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
    
    ax3.set_title('Optimized Route (Nearest Neighbor)')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.grid(True, alpha=0.3)
    
    # 4. Route statistics
    ax4.axis('off')
    
    # Calculate route statistics
    segment_distances = []
    for i in range(len(route) - 1):
        segment_distances.append(distance_matrix[route[i]][route[i+1]])
    
    route_stats = f"""
    Route Optimization Analysis:
    
    Total Locations: {n_locations}
    
    Optimized Route:
    """
    
    if location_col in locations_df.columns:
        for i, loc_idx in enumerate(route[:-1]):
            route_stats += f"\n  {i+1}. {labels[loc_idx]}"
    else:
        for i, loc_idx in enumerate(route[:-1]):
            route_stats += f"\n  {i+1}. Location {loc_idx + 1}"
    
    route_stats += f"""
    
    Route Metrics:
    Total Distance: {total_distance:.4f}
    Average Segment: {np.mean(segment_distances):.4f}
    Longest Segment: {max(segment_distances):.4f}
    Shortest Segment: {min(segment_distances):.4f}
    
    Efficiency Analysis:
    Std Dev Segments: {np.std(segment_distances):.4f}
    CV Segments: {np.std(segment_distances)/np.mean(segment_distances)*100:.1f}%
    """
    
    ax4.text(0.1, 0.9, route_stats, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax4.set_title('Route Statistics')
    
    plt.suptitle(title or 'Route Optimization Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig