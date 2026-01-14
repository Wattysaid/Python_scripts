"""
Anomaly Detection and Outlier Analysis Visualizations
-----------------------------------------------------
Functions for creating anomaly detection and outlier analysis visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def create_anomaly_detection_dashboard(data, contamination=0.1, title=None):
    """Create comprehensive anomaly detection visualization."""
    # Prepare data
    numeric_data = data.select_dtypes(include=[np.number])
    if len(numeric_data.columns) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'No numeric data available for anomaly detection',
               ha='center', va='center', transform=ax.transAxes)
        return fig

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_data)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_scores = iso_forest.fit_predict(X_scaled)
    iso_anomaly_scores = iso_forest.decision_function(X_scaled)

    # Plot anomaly scores
    ax1.scatter(range(len(iso_anomaly_scores)), iso_anomaly_scores,
               c=iso_scores, cmap='coolwarm', alpha=0.6)
    ax1.axhline(y=-0.5, color='red', linestyle='--', label='Anomaly threshold')
    ax1.set_xlabel('Observation Index')
    ax1.set_ylabel('Anomaly Score')
    ax1.set_title('Isolation Forest Anomaly Scores')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Local Outlier Factor
    lof = LocalOutlierFactor(contamination=contamination, n_neighbors=20)
    lof_scores = lof.fit_predict(X_scaled)
    lof_anomaly_scores = -lof.negative_outlier_factor_

    ax2.scatter(range(len(lof_anomaly_scores)), lof_anomaly_scores,
               c=lof_scores, cmap='coolwarm', alpha=0.6)
    ax2.axhline(y=np.percentile(lof_anomaly_scores, (1-contamination)*100),
               color='red', linestyle='--', label='Anomaly threshold')
    ax2.set_xlabel('Observation Index')
    ax2.set_ylabel('LOF Score')
    ax2.set_title('Local Outlier Factor Scores')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. One-Class SVM
    ocsvm = OneClassSVM(nu=contamination, kernel='rbf', gamma='auto')
    ocsvm_scores = ocsvm.fit_predict(X_scaled)
    ocsvm_decision_scores = ocsvm.decision_function(X_scaled)

    ax3.scatter(range(len(ocsvm_decision_scores)), ocsvm_decision_scores,
               c=ocsvm_scores, cmap='coolwarm', alpha=0.6)
    ax3.axhline(y=0, color='red', linestyle='--', label='Decision boundary')
    ax3.set_xlabel('Observation Index')
    ax3.set_ylabel('Decision Function')
    ax3.set_title('One-Class SVM Scores')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Anomaly detection summary
    ax4.axis('off')

    # Calculate consensus anomalies
    iso_anomalies = iso_scores == -1
    lof_anomalies = lof_scores == -1
    ocsvm_anomalies = ocsvm_scores == -1

    consensus_anomalies = iso_anomalies & lof_anomalies & ocsvm_anomalies
    any_anomalies = iso_anomalies | lof_anomalies | ocsvm_anomalies

    anomaly_summary = f"""
    Anomaly Detection Summary:

    Dataset Size: {len(data)} observations
    Contamination Rate: {contamination:.1%}

    Detected Anomalies:
    Isolation Forest: {np.sum(iso_anomalies)} ({np.sum(iso_anomalies)/len(data)*100:.1f}%)
    Local Outlier Factor: {np.sum(lof_anomalies)} ({np.sum(lof_anomalies)/len(data)*100:.1f}%)
    One-Class SVM: {np.sum(ocsvm_anomalies)} ({np.sum(ocsvm_anomalies)/len(data)*100:.1f}%)

    Consensus Anomalies (all methods): {np.sum(consensus_anomalies)} ({np.sum(consensus_anomalies)/len(data)*100:.1f}%)
    Any Method Anomalies: {np.sum(any_anomalies)} ({np.sum(any_anomalies)/len(data)*100:.1f}%)

    Method Agreement:
    """

    # Calculate agreement between methods
    iso_lof_agree = np.sum(iso_anomalies == lof_anomalies) / len(data)
    iso_ocsvm_agree = np.sum(iso_anomalies == ocsvm_anomalies) / len(data)
    lof_ocsvm_agree = np.sum(lof_anomalies == ocsvm_anomalies) / len(data)

    anomaly_summary += f"""
    Isolation Forest ↔ LOF: {iso_lof_agree:.1%}
    Isolation Forest ↔ OCSVM: {iso_ocsvm_agree:.1%}
    LOF ↔ OCSVM: {lof_ocsvm_agree:.1%}
    """

    ax4.text(0.1, 0.9, anomaly_summary, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle(title or 'Anomaly Detection Analysis Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig

def create_outlier_analysis_plot(data, method='iqr', title=None):
    """Create comprehensive outlier analysis visualization."""
    numeric_cols = data.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'No numeric data available for outlier analysis',
               ha='center', va='center', transform=ax.transAxes)
        return fig

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    outlier_counts = {}
    outlier_percentages = {}

    for i, col in enumerate(numeric_cols[:6]):  # Limit to 6 variables for display
        ax = axes[i]
        values = data[col].dropna()

        if method == 'iqr':
            # IQR method
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = (values < lower_bound) | (values > upper_bound)
        elif method == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(values))
            outliers = z_scores > 3
        elif method == 'modified_zscore':
            # Modified Z-score method
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            modified_z = 0.6745 * (values - median) / mad
            outliers = np.abs(modified_z) > 3.5
        else:
            outliers = pd.Series(False, index=values.index)

        # Plot box plot with outliers highlighted
        box_data = [values[~outliers], values[outliers]]
        bp = ax.boxplot(box_data, patch_artist=True, labels=['Normal', 'Outliers'])

        # Color the boxes
        colors = ['lightblue', 'red']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(f'{col} ({method.upper()} method)')
        ax.grid(True, alpha=0.3)

        outlier_counts[col] = outliers.sum()
        outlier_percentages[col] = (outliers.sum() / len(values)) * 100

    # Summary statistics
    if len(numeric_cols) > 6:
        ax_summary = axes[5]
    else:
        ax_summary = axes[len(numeric_cols) % 6]

    ax_summary.axis('off')

    summary_text = f"""
    Outlier Analysis Summary:

    Method Used: {method.upper()}
    Variables Analyzed: {len(numeric_cols)}

    Outlier Counts:
    """

    for col in numeric_cols[:10]:  # Show first 10
        if col in outlier_counts:
            summary_text += f"""
    {col}: {outlier_counts[col]} ({outlier_percentages[col]:.1f}%)"""

    if len(numeric_cols) > 10:
        summary_text += f"""
    ... and {len(numeric_cols) - 10} more variables"""

    # Overall statistics
    total_outliers = sum(outlier_counts.values())
    total_observations = sum(len(data[col].dropna()) for col in numeric_cols)
    overall_percentage = (total_outliers / total_observations) * 100 if total_observations > 0 else 0

    summary_text += f"""

    Overall Statistics:
    Total Outliers: {total_outliers}
    Total Observations: {total_observations}
    Overall Outlier Rate: {overall_percentage:.1f}%

    Detection Method Details:
    """

    if method == 'iqr':
        summary_text += """
    IQR Method: Values outside 1.5 * IQR from Q1/Q3"""
    elif method == 'zscore':
        summary_text += """
    Z-Score Method: Values with |z| > 3"""
    elif method == 'modified_zscore':
        summary_text += """
    Modified Z-Score: Values with |Mz| > 3.5"""

    ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen'))

    plt.suptitle(title or f'Outlier Analysis ({method.upper()} Method)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig

def create_multivariate_anomaly_plot(data, n_components=2, title=None):
    """Create multivariate anomaly detection visualization using PCA."""
    numeric_data = data.select_dtypes(include=[np.number])

    if len(numeric_data.columns) < 2:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'Need at least 2 numeric variables for multivariate analysis',
               ha='center', va='center', transform=ax.transAxes)
        return fig

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_data)

    # PCA for dimensionality reduction
    pca = PCA(n_components=min(n_components, len(numeric_data.columns)))
    X_pca = pca.fit_transform(X_scaled)

    # Anomaly detection on PCA space
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomaly_scores = iso_forest.fit_predict(X_pca)
    decision_scores = iso_forest.decision_function(X_pca)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. PCA scatter plot with anomalies
    if pca.n_components_ >= 2:
        scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=anomaly_scores,
                             cmap='coolwarm', alpha=0.7, s=50)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax1.set_title('PCA Space with Anomalies')
        ax1.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_ticks([-1, 1])
        cbar.set_ticklabels(['Anomaly', 'Normal'])

    # 2. Anomaly score distribution
    ax2.hist(decision_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(np.percentile(decision_scores, 10), color='red', linestyle='--',
               label='10th percentile (anomaly threshold)')
    ax2.set_xlabel('Anomaly Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Anomaly Score Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Variable contributions to anomalies
    ax3.axis('off')

    # Find most anomalous observations
    n_top_anomalies = min(5, np.sum(anomaly_scores == -1))
    if n_top_anomalies > 0:
        anomaly_indices = np.where(anomaly_scores == -1)[0]
        anomaly_scores_sorted = np.argsort(decision_scores[anomaly_indices])

        anomaly_analysis = """
        Top Anomalous Observations:

        """

        for i in range(n_top_anomalies):
            idx = anomaly_indices[anomaly_scores_sorted[i]]
            anomaly_analysis += f"""
        Observation {idx}:
        Anomaly Score: {decision_scores[idx]:.3f}
        """

            # Show extreme values for this observation
            obs_values = numeric_data.iloc[idx]
            extreme_features = []
            for col in numeric_data.columns:
                z_score = (obs_values[col] - numeric_data[col].mean()) / numeric_data[col].std()
                if abs(z_score) > 2:
                    extreme_features.append(f"{col}: {obs_values[col]:.2f} (z={z_score:.1f})")

            if extreme_features:
                anomaly_analysis += "        Extreme features: " + "; ".join(extreme_features[:3])
            anomaly_analysis += "\n"
    else:
        anomaly_analysis = "No anomalies detected with current parameters."

    ax3.text(0.1, 0.9, anomaly_analysis, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral'))

    # 4. Multivariate analysis summary
    ax4.axis('off')

    multivariate_summary = f"""
    Multivariate Anomaly Analysis:

    Dataset: {len(data)} observations, {len(numeric_data.columns)} variables
    PCA Components: {pca.n_components_}
    Explained Variance: {np.sum(pca.explained_variance_ratio_):.1%}

    Anomaly Detection:
    Method: Isolation Forest
    Contamination: 10%
    Anomalies Detected: {np.sum(anomaly_scores == -1)} ({np.sum(anomaly_scores == -1)/len(data)*100:.1f}%)

    PCA Loadings (Top 3 variables per component):
    """

    # Show top contributing variables for each PC
    for i in range(min(3, pca.n_components_)):
        loadings = pd.Series(pca.components_[i], index=numeric_data.columns)
        top_vars = loadings.abs().sort_values(ascending=False).head(3)

        multivariate_summary += f"""
    PC{i+1} ({pca.explained_variance_ratio_[i]:.1%}):
    """

        for var, loading in top_vars.items():
            multivariate_summary += f"    {var}: {loading:.3f}\n"

    ax4.text(0.1, 0.9, multivariate_summary, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue'))

    plt.suptitle(title or 'Multivariate Anomaly Detection Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig

def create_time_series_anomaly_plot(time_series_data, date_col, value_col, window_size=20, title=None):
    """Create time series anomaly detection visualization."""
    # Prepare data
    df = time_series_data.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

    values = df[value_col].values

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Time series with rolling statistics
    ax1.plot(df[date_col], values, alpha=0.7, label='Original')

    # Rolling mean and std
    rolling_mean = pd.Series(values).rolling(window=window_size).mean()
    rolling_std = pd.Series(values).rolling(window=window_size).std()

    ax1.plot(df[date_col], rolling_mean, 'r-', linewidth=2, label='Rolling Mean')
    ax1.fill_between(df[date_col], rolling_mean - 2*rolling_std,
                    rolling_mean + 2*rolling_std, alpha=0.3, color='red',
                    label='±2σ Confidence Band')
    ax1.set_title('Time Series with Rolling Statistics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Anomaly detection using rolling statistics
    z_scores = (values - rolling_mean) / rolling_std
    anomalies = np.abs(z_scores) > 3

    ax2.plot(df[date_col], z_scores, 'b-', alpha=0.7, label='Z-Score')
    ax2.axhline(y=3, color='red', linestyle='--', label='+3σ threshold')
    ax2.axhline(y=-3, color='red', linestyle='--', label='-3σ threshold')
    ax2.scatter(df[date_col][anomalies], z_scores[anomalies],
               color='red', s=50, label='Anomalies')
    ax2.set_title('Z-Score Based Anomaly Detection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Seasonal decomposition and anomalies
    ax3.axis('off')

    try:
        from statsmodels.tsa.seasonal import seasonal_decompose

        # Set frequency (assume daily data)
        ts = pd.Series(values, index=df[date_col])
        decomposition = seasonal_decompose(ts, model='additive', period=min(7, len(ts)//2))

        # Calculate residuals and detect anomalies
        residuals = decomposition.resid.dropna()
        resid_mean = residuals.mean()
        resid_std = residuals.std()
        resid_z_scores = (residuals - resid_mean) / resid_std
        resid_anomalies = np.abs(resid_z_scores) > 3

        seasonal_analysis = f"""
        Seasonal Decomposition Analysis:

        Trend Component: {'Detected' if decomposition.trend.std() > decomposition.resid.std() else 'Weak'}
        Seasonal Component: {'Strong' if decomposition.seasonal.std() > decomposition.resid.std() else 'Weak'}
        Residual Anomalies: {resid_anomalies.sum()} ({resid_anomalies.sum()/len(residuals)*100:.1f}%)

        Decomposition Statistics:
        Original Std: {ts.std():.3f}
        Trend Std: {decomposition.trend.std():.3f}
        Seasonal Std: {decomposition.seasonal.std():.3f}
        Residual Std: {decomposition.resid.std():.3f}
        """

    except:
        seasonal_analysis = "Seasonal decomposition not available (requires statsmodels)"

    ax3.text(0.1, 0.9, seasonal_analysis, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen'))

    # 4. Anomaly summary and patterns
    ax4.axis('off')

    anomaly_summary = f"""
    Time Series Anomaly Analysis:

    Total Observations: {len(values)}
    Window Size: {window_size}
    Anomalies Detected: {anomalies.sum()} ({anomalies.sum()/len(values)*100:.1f}%)

    Anomaly Patterns:
    """

    # Analyze anomaly patterns
    if anomalies.sum() > 0:
        anomaly_indices = np.where(anomalies)[0]

        # Check for clustering
        if len(anomaly_indices) > 1:
            gaps = np.diff(anomaly_indices)
            clustered = np.sum(gaps == 1)  # Consecutive anomalies
            anomaly_summary += f"""
    Clustered Anomalies: {clustered} consecutive pairs
    Isolated Anomalies: {anomalies.sum() - clustered}
    Average Gap: {np.mean(gaps):.1f} observations
    """
        else:
            anomaly_summary += """
    Single anomaly detected
    """

        # Magnitude analysis
        anomaly_values = values[anomalies]
        anomaly_summary += f"""
    Anomaly Magnitude:
    Mean: {np.mean(anomaly_values):.3f}
    Min: {np.min(anomaly_values):.3f}
    Max: {np.max(anomaly_values):.3f}
    Std: {np.std(anomaly_values):.3f}

    Direction:
    Positive Anomalies: {np.sum(anomaly_values > np.mean(values))}
    Negative Anomalies: {np.sum(anomaly_values < np.mean(values))}
    """
    else:
        anomaly_summary += """
    No anomalies detected with current threshold (3σ)
    """

    ax4.text(0.1, 0.9, anomaly_summary, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle(title or 'Time Series Anomaly Detection Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig