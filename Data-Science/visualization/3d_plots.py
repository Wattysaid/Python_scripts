"""
3D Visualizations
----------------
Functions for creating 3D plots and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.interpolate import griddata
from sklearn.decomposition import PCA

def create_3d_scatter_plot(x, y, z, colors=None, sizes=None, title=None):
    """Create 3D scatter plot with statistical analysis."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot
    if colors is not None and sizes is not None:
        scatter = ax.scatter(x, y, z, c=colors, s=sizes, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, ax=ax, shrink=0.5)
    elif colors is not None:
        scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, ax=ax, shrink=0.5)
    else:
        ax.scatter(x, y, z, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title or '3D Scatter Plot')
    
    # Statistical analysis
    stats = {
        'x_stats': {'mean': np.mean(x), 'std': np.std(x), 'min': np.min(x), 'max': np.max(x)},
        'y_stats': {'mean': np.mean(y), 'std': np.std(y), 'min': np.min(y), 'max': np.max(y)},
        'z_stats': {'mean': np.mean(z), 'std': np.std(z), 'min': np.min(z), 'max': np.max(z)},
        'correlations': {
            'x_y': np.corrcoef(x, y)[0, 1],
            'x_z': np.corrcoef(x, z)[0, 1],
            'y_z': np.corrcoef(y, z)[0, 1]
        }
    }
    
    return fig, stats

def create_3d_surface_plot(x, y, z, title=None):
    """Create 3D surface plot."""
    # Create meshgrid
    if isinstance(x, (list, np.ndarray)) and len(x.shape) == 1:
        xi = np.linspace(x.min(), x.max(), 50)
        yi = np.linspace(y.min(), y.max(), 50)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), z, (xi, yi), method='cubic')
    else:
        xi, yi, zi = x, y, z
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot
    surface = ax.plot_surface(xi, yi, zi, cmap='viridis', alpha=0.7)
    
    # Add contour lines
    ax.contour(xi, yi, zi, zdir='z', offset=zi.min(), cmap='viridis', alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title or '3D Surface Plot')
    
    plt.colorbar(surface, ax=ax, shrink=0.5)
    
    return fig

def create_3d_bar_chart(categories, values, groups=None, title=None):
    """Create 3D bar chart."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if groups is not None:
        unique_groups = list(set(groups))
        x_pos = []
        y_pos = []
        z_pos = []
        dx = []
        dy = []
        dz = []
        colors = []
        
        for i, group in enumerate(unique_groups):
            group_indices = [j for j, g in enumerate(groups) if g == group]
            for j, idx in enumerate(group_indices):
                x_pos.append(j)
                y_pos.append(i)
                z_pos.append(0)
                dx.append(0.8)
                dy.append(0.8)
                dz.append(values[idx])
                colors.append(plt.cm.viridis(i / len(unique_groups)))
        
        ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, alpha=0.7)
        
        # Set labels
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories)
        ax.set_yticks(range(len(unique_groups)))
        ax.set_yticklabels(unique_groups)
    else:
        x_pos = range(len(categories))
        y_pos = [0] * len(categories)
        z_pos = [0] * len(categories)
        dx = [0.8] * len(categories)
        dy = [0.8] * len(categories)
        dz = values
        
        ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, alpha=0.7)
        
        # Set labels
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories)
    
    ax.set_xlabel('Categories')
    ax.set_ylabel('Groups' if groups else '')
    ax.set_zlabel('Values')
    ax.set_title(title or '3D Bar Chart')
    
    return fig

def create_3d_pca_visualization(data, labels=None, title=None):
    """Create 3D PCA visualization."""
    # Perform PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if labels is not None:
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            ax.scatter(pca_result[mask, 0], pca_result[mask, 1], pca_result[mask, 2], 
                      c=[colors[i]], label=label, s=50, alpha=0.6)
        ax.legend()
    else:
        ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], alpha=0.6)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)')
    ax.set_title(title or f'3D PCA Visualization ({pca.explained_variance_ratio_.sum():.2%} total variance)')
    
    pca_stats = {
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'components': pca.components_
    }
    
    return fig, pca_stats