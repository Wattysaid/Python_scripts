"""
Dimensionality Reduction
-----------------------
Functions for reducing dataset dimensions while preserving information.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, FactorAnalysis
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif

def apply_pca(data, n_components=None, explained_variance_threshold=0.95):
    """Apply Principal Component Analysis with detailed analysis."""
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Determine number of components if not specified
    if n_components is None:
        pca_temp = PCA()
        pca_temp.fit(data_scaled)
        cumvar = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = np.argmax(cumvar >= explained_variance_threshold) + 1
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)
    
    # Calculate metrics
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Create component interpretation
    if hasattr(data, 'columns'):
        feature_names = data.columns
        components_df = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=feature_names
        )
    else:
        components_df = None
    
    return {
        'transformed_data': data_pca,
        'pca_model': pca,
        'scaler': scaler,
        'explained_variance_ratio': explained_variance,
        'cumulative_variance': cumulative_variance,
        'n_components': n_components,
        'total_variance_explained': cumulative_variance[-1],
        'components_matrix': components_df
    }

def apply_tsne(data, n_components=2, perplexity=30, random_state=42):
    """Apply t-SNE for non-linear dimensionality reduction."""
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Apply t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    data_tsne = tsne.fit_transform(data_scaled)
    
    return {
        'transformed_data': data_tsne,
        'tsne_model': tsne,
        'scaler': scaler,
        'perplexity': perplexity,
        'n_components': n_components
    }

def feature_selection_statistical(X, y, k=10, task_type='classification'):
    """Select features using statistical tests."""
    if task_type == 'classification':
        selector = SelectKBest(score_func=f_classif, k=k)
    else:
        selector = SelectKBest(score_func=f_regression, k=k)
    
    X_selected = selector.fit_transform(X, y)
    
    # Get feature scores and rankings
    feature_scores = selector.scores_
    selected_features = selector.get_support()
    
    if hasattr(X, 'columns'):
        feature_names = X.columns
        feature_ranking = pd.DataFrame({
            'feature': feature_names,
            'score': feature_scores,
            'selected': selected_features,
            'rank': np.argsort(-feature_scores) + 1
        }).sort_values('score', ascending=False)
    else:
        feature_ranking = None
    
    return {
        'selected_features': X_selected,
        'selector': selector,
        'feature_scores': feature_scores,
        'selected_mask': selected_features,
        'feature_ranking': feature_ranking,
        'n_features_selected': k
    }

def mutual_information_selection(X, y, k=10):
    """Select features using mutual information."""
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get feature scores
    mi_scores = selector.scores_
    selected_features = selector.get_support()
    
    if hasattr(X, 'columns'):
        feature_names = X.columns
        mi_ranking = pd.DataFrame({
            'feature': feature_names,
            'mi_score': mi_scores,
            'selected': selected_features,
            'rank': np.argsort(-mi_scores) + 1
        }).sort_values('mi_score', ascending=False)
    else:
        mi_ranking = None
    
    return {
        'selected_features': X_selected,
        'selector': selector,
        'mi_scores': mi_scores,
        'selected_mask': selected_features,
        'mi_ranking': mi_ranking
    }

def compare_dimensionality_methods(data, methods=['pca', 'tsne'], n_components=2):
    """Compare different dimensionality reduction methods."""
    results = {}
    
    # Standardize data once
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    for method in methods:
        try:
            if method == 'pca':
                reducer = PCA(n_components=n_components)
                transformed = reducer.fit_transform(data_scaled)
                results[method] = {
                    'transformed_data': transformed,
                    'model': reducer,
                    'explained_variance': getattr(reducer, 'explained_variance_ratio_', None)
                }
                
            elif method == 'tsne':
                reducer = TSNE(n_components=n_components, random_state=42)
                transformed = reducer.fit_transform(data_scaled)
                results[method] = {
                    'transformed_data': transformed,
                    'model': reducer
                }
                
            elif method == 'ica':
                reducer = FastICA(n_components=n_components, random_state=42)
                transformed = reducer.fit_transform(data_scaled)
                results[method] = {
                    'transformed_data': transformed,
                    'model': reducer
                }
                
            elif method == 'factor_analysis':
                reducer = FactorAnalysis(n_components=n_components, random_state=42)
                transformed = reducer.fit_transform(data_scaled)
                results[method] = {
                    'transformed_data': transformed,
                    'model': reducer
                }
                
        except Exception as e:
            results[method] = {'error': str(e)}
    
    return {
        'results': results,
        'scaler': scaler,
        'methods_compared': methods,
        'n_components': n_components
    }