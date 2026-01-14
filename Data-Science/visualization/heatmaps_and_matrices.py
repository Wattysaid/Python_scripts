"""
Heatmaps and Matrix Visualizations
---------------------------------
Functions for creating heatmaps, correlation matrices, and related visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import pearsonr

def create_correlation_heatmap_advanced(df, method='pearson', annot=True, title=None):
    """Create advanced correlation heatmap with clustering and significance testing."""
    # Calculate correlations
    if method == 'pearson':
        corr_matrix = df.corr(method='pearson')
    elif method == 'spearman':
        corr_matrix = df.corr(method='spearman')
    else:
        corr_matrix = df.corr()
    
    # Calculate p-values for significance
    n = len(df)
    p_values = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)
    
    for i in corr_matrix.index:
        for j in corr_matrix.columns:
            if i != j:
                _, p_val = pearsonr(df[i].dropna(), df[j].dropna())
                p_values.loc[i, j] = p_val
            else:
                p_values.loc[i, j] = 0
    
    # Create mask for non-significant correlations
    mask_insignificant = p_values > 0.05
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Full correlation heatmap
    sns.heatmap(corr_matrix, annot=annot, cmap='coolwarm', center=0, ax=ax1,
                square=True, cbar_kws={'label': 'Correlation Coefficient'})
    ax1.set_title('Correlation Matrix (All)')
    
    # 2. Significant correlations only
    corr_significant = corr_matrix.copy()
    corr_significant[mask_insignificant] = 0
    sns.heatmap(corr_significant, annot=annot, cmap='coolwarm', center=0, ax=ax2,
                square=True, cbar_kws={'label': 'Significant Correlations (p<0.05)'})
    ax2.set_title('Significant Correlations Only')
    
    # 3. P-values heatmap
    sns.heatmap(p_values.astype(float), annot=True, cmap='Reds_r', ax=ax3,
                square=True, cbar_kws={'label': 'P-values'})
    ax3.set_title('P-values Matrix')
    
    # 4. Clustered heatmap
    if len(corr_matrix) > 2:
        try:
            # Hierarchical clustering
            linkage_matrix = linkage(corr_matrix, method='ward')
            cluster_order = dendrogram(linkage_matrix, no_plot=True)['leaves']
            corr_clustered = corr_matrix.iloc[cluster_order, cluster_order]
            
            sns.heatmap(corr_clustered, annot=annot, cmap='coolwarm', center=0, ax=ax4,
                       square=True, cbar_kws={'label': 'Correlation (Clustered)'})
            ax4.set_title('Clustered Correlation Matrix')
        except:
            ax4.text(0.5, 0.5, 'Clustering failed\nInsufficient data', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Clustering Error')
    
    plt.suptitle(title or f'{method.title()} Correlation Analysis')
    plt.tight_layout()
    
    # Statistical summary
    strong_correlations = corr_matrix[(abs(corr_matrix) > 0.7) & (corr_matrix != 1)]
    stats_summary = {
        'total_pairs': len(corr_matrix) * (len(corr_matrix) - 1) // 2,
        'strong_correlations': strong_correlations.count().sum() // 2,
        'significant_correlations': (~mask_insignificant).sum().sum() // 2 - len(corr_matrix) // 2,
        'avg_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean(),
        'max_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max(),
        'min_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min()
    }
    
    return fig, stats_summary

def create_confusion_matrix_heatmap(y_true, y_pred, labels=None, title=None):
    """Create detailed confusion matrix heatmap with performance metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Raw confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # 2. Normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax2)
    ax2.set_title('Normalized Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    # 3. Error analysis heatmap
    error_matrix = cm.copy()
    np.fill_diagonal(error_matrix, 0)  # Remove correct predictions
    sns.heatmap(error_matrix, annot=True, fmt='d', cmap='Reds',
                xticklabels=labels, yticklabels=labels, ax=ax3)
    ax3.set_title('Error Matrix (Misclassifications Only)')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    # 4. Performance metrics table
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    
    metrics_text = "Classification Performance:\n\n"
    for label in labels:
        if str(label) in report:
            metrics = report[str(label)]
            metrics_text += f"Class {label}:\n"
            metrics_text += f"  Precision: {metrics['precision']:.3f}\n"
            metrics_text += f"  Recall: {metrics['recall']:.3f}\n"
            metrics_text += f"  F1-Score: {metrics['f1-score']:.3f}\n\n"
    
    metrics_text += f"Overall Accuracy: {report['accuracy']:.3f}\n"
    metrics_text += f"Macro Avg F1: {report['macro avg']['f1-score']:.3f}\n"
    metrics_text += f"Weighted Avg F1: {report['weighted avg']['f1-score']:.3f}"
    
    ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray'))
    ax4.axis('off')
    ax4.set_title('Performance Metrics')
    
    plt.suptitle(title or 'Confusion Matrix Analysis')
    plt.tight_layout()
    
    return fig, report

def create_feature_importance_heatmap(feature_names, importance_scores, groups=None, title=None):
    """Create feature importance heatmap with grouping."""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    })
    
    if groups is not None:
        importance_df['group'] = groups
        # Create pivot table for heatmap
        pivot_df = importance_df.pivot_table(index='group', columns='feature', 
                                            values='importance', fill_value=0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Grouped heatmap
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='viridis', ax=ax1)
        ax1.set_title('Feature Importance by Group')
        
        # Group summary
        group_summary = importance_df.groupby('group')['importance'].agg(['mean', 'sum', 'count'])
        sns.heatmap(group_summary.T, annot=True, fmt='.3f', cmap='plasma', ax=ax2)
        ax2.set_title('Group Summary Statistics')
        
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Single heatmap
        importance_matrix = importance_scores.reshape(-1, 1)
        sns.heatmap(importance_matrix, 
                   yticklabels=feature_names,
                   xticklabels=['Importance'],
                   annot=True, fmt='.3f', cmap='viridis', ax=ax1)
        ax1.set_title('Feature Importance')
        
        # Top features bar plot
        top_features = importance_df.nlargest(10, 'importance')
        ax2.barh(range(len(top_features)), top_features['importance'])
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(top_features['feature'])
        ax2.set_xlabel('Importance Score')
        ax2.set_title('Top 10 Most Important Features')
        ax2.invert_yaxis()
    
    plt.suptitle(title or 'Feature Importance Analysis')
    plt.tight_layout()
    
    return fig

def create_distance_matrix_heatmap(data, metric='euclidean', labels=None, title=None):
    """Create distance matrix heatmap for similarity analysis."""
    from scipy.spatial.distance import pdist, squareform
    
    # Calculate distance matrix
    distances = pdist(data, metric=metric)
    distance_matrix = squareform(distances)
    
    if labels is None:
        labels = [f'Sample {i+1}' for i in range(len(data))]
    
    distance_df = pd.DataFrame(distance_matrix, index=labels, columns=labels)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Distance matrix heatmap
    sns.heatmap(distance_df, annot=True, fmt='.2f', cmap='viridis', ax=ax1)
    ax1.set_title(f'Distance Matrix ({metric})')
    
    # 2. Similarity matrix (inverse of distance)
    max_distance = distance_matrix.max()
    similarity_matrix = max_distance - distance_matrix
    similarity_df = pd.DataFrame(similarity_matrix, index=labels, columns=labels)
    
    sns.heatmap(similarity_df, annot=True, fmt='.2f', cmap='plasma', ax=ax2)
    ax2.set_title('Similarity Matrix')
    
    # 3. Clustered distance matrix
    try:
        linkage_matrix = linkage(distances, method='ward')
        cluster_order = dendrogram(linkage_matrix, no_plot=True)['leaves']
        distance_clustered = distance_df.iloc[cluster_order, cluster_order]
        
        sns.heatmap(distance_clustered, annot=True, fmt='.2f', cmap='viridis', ax=ax3)
        ax3.set_title('Clustered Distance Matrix')
    except:
        ax3.text(0.5, 0.5, 'Clustering failed', ha='center', va='center', transform=ax3.transAxes)
    
    # 4. Distance distribution
    ax4.hist(distances, bins=20, alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(distances), color='red', linestyle='--', label=f'Mean: {np.mean(distances):.2f}')
    ax4.axvline(np.median(distances), color='green', linestyle='--', label=f'Median: {np.median(distances):.2f}')
    ax4.set_xlabel('Distance')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distance Distribution')
    ax4.legend()
    
    plt.suptitle(title or f'Distance Analysis ({metric} metric)')
    plt.tight_layout()
    
    distance_stats = {
        'mean_distance': np.mean(distances),
        'median_distance': np.median(distances),
        'std_distance': np.std(distances),
        'min_distance': np.min(distances[distances > 0]),  # Exclude self-distances
        'max_distance': np.max(distances),
        'total_pairs': len(distances)
    }
    
    return fig, distance_stats