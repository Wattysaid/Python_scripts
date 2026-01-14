"""
Anomaly Detection
----------------
Functions for detecting anomalies and outliers in datasets.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

def detect_outliers_zscore(data, threshold=3, columns=None):
    """Detect outliers using Z-score method."""
    if isinstance(data, pd.Series):
        data = data.to_frame()
    
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    outliers_info = {}
    all_outliers = set()
    
    for col in columns:
        if col in data.columns:
            z_scores = np.abs(stats.zscore(data[col].dropna()))
            outlier_indices = data[~data[col].isna()].index[z_scores > threshold]
            
            outliers_info[col] = {
                'outlier_indices': outlier_indices.tolist(),
                'outlier_count': len(outlier_indices),
                'outlier_percentage': len(outlier_indices) / len(data) * 100,
                'threshold': threshold,
                'max_zscore': z_scores.max(),
                'outlier_values': data.loc[outlier_indices, col].tolist()
            }
            
            all_outliers.update(outlier_indices)
    
    return {
        'outliers_by_column': outliers_info,
        'all_outlier_indices': list(all_outliers),
        'total_outliers': len(all_outliers),
        'outlier_percentage': len(all_outliers) / len(data) * 100
    }

def detect_outliers_iqr(data, factor=1.5, columns=None):
    """Detect outliers using Interquartile Range (IQR) method."""
    if isinstance(data, pd.Series):
        data = data.to_frame()
    
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    outliers_info = {}
    all_outliers = set()
    
    for col in columns:
        if col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
            outlier_indices = data[outlier_mask].index
            
            outliers_info[col] = {
                'outlier_indices': outlier_indices.tolist(),
                'outlier_count': len(outlier_indices),
                'outlier_percentage': len(outlier_indices) / len(data) * 100,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_values': data.loc[outlier_indices, col].tolist()
            }
            
            all_outliers.update(outlier_indices)
    
    return {
        'outliers_by_column': outliers_info,
        'all_outlier_indices': list(all_outliers),
        'total_outliers': len(all_outliers),
        'outlier_percentage': len(all_outliers) / len(data) * 100
    }

def detect_outliers_isolation_forest(data, contamination=0.1, random_state=42):
    """Detect outliers using Isolation Forest algorithm."""
    # Select numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        return {'error': 'No numeric columns found'}
    
    # Handle missing values
    numeric_data_clean = numeric_data.fillna(numeric_data.median())
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(numeric_data_clean)
    
    # Apply Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    outlier_labels = iso_forest.fit_predict(data_scaled)
    
    # Get outlier indices (-1 indicates outlier)
    outlier_indices = data.index[outlier_labels == -1]
    
    # Calculate anomaly scores
    anomaly_scores = iso_forest.decision_function(data_scaled)
    
    return {
        'outlier_indices': outlier_indices.tolist(),
        'outlier_count': len(outlier_indices),
        'outlier_percentage': len(outlier_indices) / len(data) * 100,
        'anomaly_scores': anomaly_scores,
        'contamination': contamination,
        'model': iso_forest,
        'scaler': scaler
    }

def detect_outliers_local_outlier_factor(data, n_neighbors=20, contamination=0.1):
    """Detect outliers using Local Outlier Factor (LOF)."""
    # Select numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        return {'error': 'No numeric columns found'}
    
    # Handle missing values
    numeric_data_clean = numeric_data.fillna(numeric_data.median())
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(numeric_data_clean)
    
    # Apply LOF
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    outlier_labels = lof.fit_predict(data_scaled)
    
    # Get outlier indices (-1 indicates outlier)
    outlier_indices = data.index[outlier_labels == -1]
    
    # Get LOF scores (negative_outlier_factor_)
    lof_scores = lof.negative_outlier_factor_
    
    return {
        'outlier_indices': outlier_indices.tolist(),
        'outlier_count': len(outlier_indices),
        'outlier_percentage': len(outlier_indices) / len(data) * 100,
        'lof_scores': lof_scores,
        'n_neighbors': n_neighbors,
        'contamination': contamination,
        'model': lof,
        'scaler': scaler
    }

def comprehensive_outlier_analysis(data, methods=['zscore', 'iqr', 'isolation_forest'], contamination=0.1):
    """Comprehensive outlier analysis using multiple methods."""
    results = {}
    
    for method in methods:
        if method == 'zscore':
            results[method] = detect_outliers_zscore(data)
        elif method == 'iqr':
            results[method] = detect_outliers_iqr(data)
        elif method == 'isolation_forest':
            results[method] = detect_outliers_isolation_forest(data, contamination=contamination)
        elif method == 'lof':
            results[method] = detect_outliers_local_outlier_factor(data, contamination=contamination)
    
    # Find consensus outliers (outliers detected by multiple methods)
    all_outlier_sets = []
    for method, result in results.items():
        if 'outlier_indices' in result:
            all_outlier_sets.append(set(result['outlier_indices']))
        elif 'all_outlier_indices' in result:
            all_outlier_sets.append(set(result['all_outlier_indices']))
    
    if all_outlier_sets:
        # Outliers detected by any method
        union_outliers = set.union(*all_outlier_sets)
        
        # Outliers detected by multiple methods
        consensus_outliers = set()
        for outlier in union_outliers:
            count = sum(1 for outlier_set in all_outlier_sets if outlier in outlier_set)
            if count > 1:
                consensus_outliers.add(outlier)
        
        consensus_analysis = {
            'union_outliers': list(union_outliers),
            'consensus_outliers': list(consensus_outliers),
            'union_count': len(union_outliers),
            'consensus_count': len(consensus_outliers),
            'union_percentage': len(union_outliers) / len(data) * 100,
            'consensus_percentage': len(consensus_outliers) / len(data) * 100
        }
    else:
        consensus_analysis = {}
    
    return {
        'method_results': results,
        'consensus_analysis': consensus_analysis,
        'methods_used': methods,
        'total_samples': len(data)
    }