"""
Correlation and Relationship Analysis Visualizations
---------------------------------------------------
Functions for creating correlation analysis and relationship visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def create_correlation_matrix_heatmap(data, method='pearson', title=None):
    """Create comprehensive correlation matrix visualization."""
    # Calculate correlation matrix
    if method == 'pearson':
        corr_matrix = data.corr(method='pearson')
    elif method == 'spearman':
        corr_matrix = data.corr(method='spearman')
    elif method == 'kendall':
        corr_matrix = data.corr(method='kendall')
    else:
        corr_matrix = data.corr()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Correlation heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, ax=ax1, cbar_kws={'shrink': 0.8})
    ax1.set_title(f'{method.title()} Correlation Matrix')

    # 2. Correlation significance heatmap
    p_values = pd.DataFrame(np.zeros_like(corr_matrix), columns=corr_matrix.columns, index=corr_matrix.index)

    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            if i != j:
                if method == 'pearson':
                    _, p_val = stats.pearsonr(data.iloc[:, i], data.iloc[:, j])
                elif method == 'spearman':
                    _, p_val = stats.spearmanr(data.iloc[:, i], data.iloc[:, j])
                else:
                    _, p_val = stats.kendalltau(data.iloc[:, i], data.iloc[:, j])
                p_values.iloc[i, j] = p_val

    # Create significance mask
    sig_mask = p_values < 0.05
    sns.heatmap(corr_matrix, mask=~sig_mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, ax=ax2, cbar=False)
    ax2.set_title('Significant Correlations (p < 0.05)')

    # 3. Correlation distribution
    correlations = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]
    ax3.hist(correlations, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(np.mean(correlations), color='red', linestyle='--',
               label=f'Mean: {np.mean(correlations):.3f}')
    ax3.axvline(np.median(correlations), color='green', linestyle='--',
               label=f'Median: {np.median(correlations):.3f}')
    ax3.set_xlabel('Correlation Coefficient')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Correlation Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Correlation network
    ax4.axis('off')

    # Create correlation network visualization
    threshold = 0.3  # Only show strong correlations
    strong_corr = corr_matrix.abs() > threshold

    # Calculate network statistics
    n_variables = len(corr_matrix)
    n_connections = (strong_corr.sum().sum() - n_variables) // 2  # Subtract diagonal
    avg_correlation = np.mean(correlations)
    max_correlation = np.max(np.abs(correlations))

    # Find most correlated pairs
    corr_pairs = []
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j],
                             corr_matrix.iloc[i, j]))

    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    top_pairs = corr_pairs[:5]

    network_stats = f"""
    Correlation Network Analysis:

    Variables: {n_variables}
    Strong Connections (|r| > {threshold}): {n_connections}
    Average Correlation: {avg_correlation:.3f}
    Maximum |Correlation|: {max_correlation:.3f}

    Top Correlated Pairs:
    """

    for var1, var2, corr in top_pairs:
        network_stats += f"""
    {var1} ↔ {var2}: {corr:.3f}"""

    # Add clustering information
    try:
        from scipy.cluster.hierarchy import fcluster
        linkage_matrix = linkage(1 - corr_matrix.abs(), method='average')
        clusters = fcluster(linkage_matrix, t=2, criterion='maxclust')
        n_clusters = len(set(clusters))
        network_stats += f"""

    Hierarchical Clusters: {n_clusters}"""
    except:
        pass

    ax4.text(0.1, 0.9, network_stats, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen'))

    plt.suptitle(title or f'Correlation Analysis Dashboard ({method.title()})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig

def create_scatterplot_matrix(data, hue=None, title=None):
    """Create enhanced scatterplot matrix with correlation coefficients."""
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    n_vars = len(numeric_cols)

    if n_vars < 2:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'Need at least 2 numeric variables for scatterplot matrix',
               ha='center', va='center', transform=ax.transAxes)
        return fig

    fig, axes = plt.subplots(n_vars, n_vars, figsize=(4*n_vars, 4*n_vars))
    if n_vars == 1:
        axes = np.array([[axes]])

    # Calculate correlations
    corr_matrix = data[numeric_cols].corr()

    for i in range(n_vars):
        for j in range(n_vars):
            ax = axes[i, j]

            if i == j:
                # Diagonal: histograms
                if hue and hue in data.columns:
                    for hue_val in data[hue].unique():
                        subset = data[data[hue] == hue_val]
                        ax.hist(subset[numeric_cols[i]], alpha=0.5, label=str(hue_val))
                    ax.legend()
                else:
                    ax.hist(data[numeric_cols[i]], alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_title(f'{numeric_cols[i]}')
            elif i > j:
                # Lower triangle: scatter plots
                x_col, y_col = numeric_cols[j], numeric_cols[i]

                if hue and hue in data.columns:
                    sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue, ax=ax, alpha=0.6)
                else:
                    ax.scatter(data[x_col], data[y_col], alpha=0.6, color='blue')

                # Add correlation coefficient
                corr = corr_matrix.loc[x_col, y_col]
                ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            else:
                # Upper triangle: correlation coefficients
                corr = corr_matrix.iloc[i, j]
                ax.text(0.5, 0.5, f'{corr:.3f}', ha='center', va='center',
                       fontsize=12, fontweight='bold')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')

            # Only show labels on outer edges
            if i == n_vars - 1:
                ax.set_xlabel(numeric_cols[j])
            else:
                ax.set_xlabel('')

            if j == 0:
                ax.set_ylabel(numeric_cols[i])
            else:
                ax.set_ylabel('')

    plt.suptitle(title or 'Scatterplot Matrix with Correlations', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig

def create_pca_biplot(data, n_components=2, scale=True, title=None):
    """Create PCA biplot with variable contributions."""
    # Prepare data
    numeric_data = data.select_dtypes(include=[np.number])
    feature_names = numeric_data.columns

    # Standardize data
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_data)
    else:
        X_scaled = numeric_data.values

    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. PCA scatter plot
    if n_components >= 2:
        ax1.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, color='blue', s=50)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax1.set_title('PCA Scatter Plot')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        # Add variable vectors (biplot)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        scale_factor = np.max(np.abs(X_pca)) / np.max(np.abs(loadings)) * 0.8

        for i, feature in enumerate(feature_names):
            ax1.arrow(0, 0, loadings[i, 0] * scale_factor, loadings[i, 1] * scale_factor,
                     head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.7)
            ax1.text(loadings[i, 0] * scale_factor * 1.1, loadings[i, 1] * scale_factor * 1.1,
                    feature, fontsize=9, ha='center', va='center')

    # 2. Explained variance
    ax2.bar(range(1, len(pca.explained_variance_ratio_) + 1),
            pca.explained_variance_ratio_, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_), 'r-o', linewidth=2, label='Cumulative')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_title('PCA Explained Variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Variable contributions (loadings)
    if n_components >= 2:
        loadings_df = pd.DataFrame(pca.components_.T,
                                  columns=[f'PC{i+1}' for i in range(n_components)],
                                  index=feature_names)

        sns.heatmap(loadings_df, annot=True, fmt='.3f', cmap='coolwarm', center=0, ax=ax3)
        ax3.set_title('Variable Loadings (Contributions)')

    # 4. PCA summary statistics
    ax4.axis('off')

    pca_summary = f"""
    PCA Analysis Summary:

    Number of Components: {n_components}
    Original Variables: {len(feature_names)}
    Data Scaled: {scale}

    Explained Variance:
    """

    for i, var_ratio in enumerate(pca.explained_variance_ratio_):
        pca_summary += f"""
    PC{i+1}: {var_ratio:.1%} ({pca.explained_variance_[i]:.3f} eigenvalue)"""

    pca_summary += f"""

    Cumulative Variance:
    First 2 PCs: {np.sum(pca.explained_variance_ratio_[:2]):.1%}
    First 3 PCs: {np.sum(pca.explained_variance_ratio_[:min(3, n_components)]):.1%}

    Variable Contributions (PC1):
    """

    # Top contributing variables to PC1
    pc1_loadings = pd.Series(pca.components_[0], index=feature_names)
    top_contributors = pc1_loadings.abs().sort_values(ascending=False).head(5)

    for var, loading in top_contributors.items():
        pca_summary += f"""
    {var}: {loading:.3f}"""

    ax4.text(0.1, 0.9, pca_summary, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle(title or 'PCA Biplot and Analysis Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig

def create_correlation_clustermap(data, method='average', metric='euclidean', title=None):
    """Create hierarchical clustering of correlation matrix."""
    # Calculate correlation matrix
    corr_matrix = data.corr()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Clustered heatmap
    try:
        g = sns.clustermap(corr_matrix, method=method, metric=metric,
                          cmap='coolwarm', center=0, figsize=(8, 8))
        plt.close(g.fig)  # Close the clustermap figure

        # Recreate the heatmap manually for our subplot
        linkage_matrix = linkage(1 - corr_matrix.abs(), method=method)
        dendro = dendrogram(linkage_matrix, no_plot=True)
        ordered_corr = corr_matrix.iloc[dendro['leaves'], dendro['leaves']]

        sns.heatmap(ordered_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax1)
        ax1.set_title('Clustered Correlation Matrix')
    except:
        # Fallback to regular heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax1)
        ax1.set_title('Correlation Matrix (Clustering failed)')

    # 2. Dendrogram
    try:
        linkage_matrix = linkage(1 - corr_matrix.abs(), method=method)
        dendrogram(linkage_matrix, labels=corr_matrix.columns, ax=ax2, leaf_rotation=90)
        ax2.set_title('Hierarchical Clustering Dendrogram')
    except:
        ax2.text(0.5, 0.5, 'Dendrogram not available', ha='center', va='center', transform=ax2.transAxes)

    # 3. Correlation clusters
    ax3.axis('off')

    try:
        from scipy.cluster.hierarchy import fcluster

        # Try different numbers of clusters
        cluster_results = {}
        for n_clusters in range(2, min(6, len(corr_matrix) + 1)):
            clusters = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
            cluster_results[n_clusters] = clusters

        # Display cluster membership for 3 clusters (if possible)
        if 3 in cluster_results:
            clusters = cluster_results[3]
            cluster_membership = {}

            for i, var in enumerate(corr_matrix.columns):
                cluster_id = clusters[i]
                if cluster_id not in cluster_membership:
                    cluster_membership[cluster_id] = []
                cluster_membership[cluster_id].append(var)

            cluster_text = """
            Variable Clusters (3 clusters):

            """

            for cluster_id, variables in cluster_membership.items():
                cluster_text += f"""
            Cluster {cluster_id}: {', '.join(variables)}"""

        else:
            cluster_text = "Clustering analysis not available"

    except:
        cluster_text = "Clustering analysis not available"

    ax3.text(0.1, 0.9, cluster_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue'))

    # 4. Correlation strength analysis
    ax4.axis('off')

    # Analyze correlation strengths
    correlations = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]

    strength_analysis = f"""
    Correlation Strength Analysis:

    Total Correlations: {len(correlations)}

    Very Strong (|r| ≥ 0.8): {np.sum(np.abs(correlations) >= 0.8)}
    Strong (0.6 ≤ |r| < 0.8): {np.sum((np.abs(correlations) >= 0.6) & (np.abs(correlations) < 0.8))}
    Moderate (0.3 ≤ |r| < 0.6): {np.sum((np.abs(correlations) >= 0.3) & (np.abs(correlations) < 0.6))}
    Weak (|r| < 0.3): {np.sum(np.abs(correlations) < 0.3)}

    Summary Statistics:
    Mean |r|: {np.mean(np.abs(correlations)):.3f}
    Median |r|: {np.median(np.abs(correlations)):.3f}
    Max |r|: {np.max(np.abs(correlations)):.3f}
    Min |r|: {np.min(np.abs(correlations)):.3f}

    Positive Correlations: {np.sum(correlations > 0)} ({np.sum(correlations > 0)/len(correlations)*100:.1f}%)
    Negative Correlations: {np.sum(correlations < 0)} ({np.sum(correlations < 0)/len(correlations)*100:.1f}%)
    """

    ax4.text(0.1, 0.9, strength_analysis, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen'))

    plt.suptitle(title or 'Correlation Clustering Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig