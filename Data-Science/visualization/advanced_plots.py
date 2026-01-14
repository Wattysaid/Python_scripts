"""
Advanced Plotting Functions
--------------------------
Advanced visualization functions with comprehensive statistics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA

def plot_distribution_comparison(data1, data2, labels=['Group 1', 'Group 2'], title=None):
    """Compare two distributions with statistical tests."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Histograms
    ax1.hist(data1, alpha=0.7, label=labels[0], bins=30)
    ax1.hist(data2, alpha=0.7, label=labels[1], bins=30)
    ax1.legend()
    ax1.set_title('Distribution Comparison')
    
    # Box plots
    ax2.boxplot([data1, data2], labels=labels)
    ax2.set_title('Box Plot Comparison')
    
    # Q-Q plot
    stats.probplot(data1, dist="norm", plot=ax3)
    ax3.set_title(f'Q-Q Plot - {labels[0]}')
    
    # Statistical comparison
    t_stat, t_pval = stats.ttest_ind(data1, data2)
    ks_stat, ks_pval = stats.ks_2samp(data1, data2)
    
    comparison_text = f"""
    {labels[0]} Statistics:
    Mean: {np.mean(data1):.3f}
    Std: {np.std(data1):.3f}
    Skewness: {stats.skew(data1):.3f}
    
    {labels[1]} Statistics:
    Mean: {np.mean(data2):.3f}
    Std: {np.std(data2):.3f}
    Skewness: {stats.skew(data2):.3f}
    
    Statistical Tests:
    T-test p-value: {t_pval:.4f}
    KS-test p-value: {ks_pval:.4f}
    
    Effect Size (Cohen's d):
    {(np.mean(data1) - np.mean(data2)) / np.sqrt((np.var(data1) + np.var(data2)) / 2):.3f}
    """
    
    ax4.text(0.1, 0.9, comparison_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan'))
    ax4.axis('off')
    ax4.set_title('Statistical Comparison')
    
    plt.suptitle(title or 'Distribution Comparison Analysis')
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df, title=None):
    """Create correlation heatmap with significance tests."""
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Correlation heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax1)
    ax1.set_title('Correlation Matrix')
    
    # Create significance matrix
    n = len(numeric_df)
    significance_matrix = pd.DataFrame(index=correlation_matrix.index, columns=correlation_matrix.columns)
    
    for i in correlation_matrix.index:
        for j in correlation_matrix.columns:
            if i != j:
                r = correlation_matrix.loc[i, j]
                t_stat = r * np.sqrt((n-2)/(1-r**2))
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
                significance_matrix.loc[i, j] = p_val
            else:
                significance_matrix.loc[i, j] = 0
    
    significance_matrix = significance_matrix.astype(float)
    sns.heatmap(significance_matrix, annot=True, cmap='Reds_r', ax=ax2, 
                cbar_kws={'label': 'P-value'})
    ax2.set_title('Significance Matrix (P-values)')
    
    plt.suptitle(title or 'Correlation Analysis with Significance Testing')
    plt.tight_layout()
    return fig

def plot_pca_analysis(df, title=None):
    """Perform and plot PCA analysis."""
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    
    # Standardize the data
    standardized_data = (numeric_df - numeric_df.mean()) / numeric_df.std()
    
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(standardized_data)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Scree plot
    ax1.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             pca.explained_variance_ratio_, 'bo-')
    ax1.set_title('Scree Plot')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    
    # Cumulative explained variance
    ax2.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_), 'ro-')
    ax2.set_title('Cumulative Explained Variance')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.axhline(y=0.95, color='k', linestyle='--', label='95% Variance')
    ax2.legend()
    
    # First two principal components
    ax3.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
    ax3.set_title('First Two Principal Components')
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    
    # Component loadings
    n_components_to_show = min(5, len(pca.explained_variance_ratio_))
    pca_summary = f"""
    PCA Summary:
    
    Total Components: {len(pca.explained_variance_ratio_)}
    
    Explained Variance by Component:
    """
    
    for i in range(n_components_to_show):
        pca_summary += f"PC{i+1}: {pca.explained_variance_ratio_[i]*100:.2f}%\n"
    
    pca_summary += f"""
    
    Cumulative Variance (First {n_components_to_show} PCs):
    {np.sum(pca.explained_variance_ratio_[:n_components_to_show])*100:.2f}%
    """
    
    ax4.text(0.1, 0.9, pca_summary, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen'))
    ax4.axis('off')
    ax4.set_title('PCA Summary')
    
    plt.suptitle(title or 'Principal Component Analysis')
    plt.tight_layout()
    return fig, pca