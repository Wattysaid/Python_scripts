"""
Statistical Analysis Visualizations
-----------------------------------
Functions for creating statistical analysis and hypothesis testing visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, t, chi2, f
import statsmodels.api as sm
from statsmodels.stats.power import TTestPower

def create_hypothesis_test_visualization(sample1, sample2, test_type='t-test', alpha=0.05):
    """Create comprehensive hypothesis testing visualization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Sample distributions
    sns.histplot(sample1, alpha=0.7, label='Sample 1', ax=ax1, color='blue')
    sns.histplot(sample2, alpha=0.7, label='Sample 2', ax=ax1, color='red')
    ax1.set_title('Sample Distributions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Box plots
    data = [sample1, sample2]
    ax2.boxplot(data, labels=['Sample 1', 'Sample 2'], patch_artist=True,
               boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax2.set_title('Box Plot Comparison')
    ax2.grid(True, alpha=0.3)

    # 3. Statistical test results
    ax3.axis('off')

    if test_type == 't-test':
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(sample1, sample2)

        # Calculate effect size (Cohen's d)
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        cohens_d = abs(mean1 - mean2) / pooled_std

        # Power analysis
        power_analysis = TTestPower()
        power = power_analysis.power(effect_size=cohens_d,
                                   nobs=len(sample1),
                                   alpha=alpha)

        test_results = f"""
        Hypothesis Test Results (Two-Sample t-test):

        Test Statistic: {t_stat:.4f}
        p-value: {p_value:.4f}
        Significance Level (α): {alpha}

        Effect Size (Cohen's d): {cohens_d:.3f}
        Power: {power:.3f}

        Sample 1: n={len(sample1)}, μ={mean1:.3f}, σ={std1:.3f}
        Sample 2: n={len(sample2)}, μ={mean2:.3f}, σ={std2:.3f}

        Conclusion: {'Reject H₀' if p_value < alpha else 'Fail to reject H₀'}
        """

    elif test_type == 'mann-whitney':
        # Non-parametric test
        u_stat, p_value = stats.mannwhitneyu(sample1, sample2, alternative='two-sided')

        test_results = f"""
        Hypothesis Test Results (Mann-Whitney U test):

        Test Statistic (U): {u_stat:.4f}
        p-value: {p_value:.4f}
        Significance Level (α): {alpha}

        Sample 1: n={len(sample1)}, median={np.median(sample1):.3f}
        Sample 2: n={len(sample2)}, median={np.median(sample2):.3f}

        Conclusion: {'Reject H₀' if p_value < alpha else 'Fail to reject H₀'}
        """

    ax3.text(0.1, 0.9, test_results, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # 4. Confidence intervals and effect size visualization
    means = [np.mean(sample1), np.mean(sample2)]
    stds = [np.std(sample1, ddof=1), np.std(sample2, ddof=1)]
    ns = [len(sample1), len(sample2)]

    # Calculate confidence intervals
    confidence_intervals = []
    for mean, std, n in zip(means, stds, ns):
        se = std / np.sqrt(n)
        ci_lower = mean - 1.96 * se
        ci_upper = mean + 1.96 * se
        confidence_intervals.append((ci_lower, ci_upper))

    # Plot means with confidence intervals
    ax4.errorbar([1, 2], means, yerr=[mean - ci[0] for mean, ci in zip(means, confidence_intervals)],
                fmt='o', capsize=5, markersize=8, color='darkblue', linewidth=2)
    ax4.set_xticks([1, 2])
    ax4.set_xticklabels(['Sample 1', 'Sample 2'])
    ax4.set_title('Means with 95% Confidence Intervals')
    ax4.grid(True, alpha=0.3)

    # Add effect size visualization
    if test_type == 't-test':
        # Show Cohen's d as a bar
        ax4.axhline(y=mean1, xmin=0.1, xmax=0.4, color='blue', linestyle='--', alpha=0.7)
        ax4.axhline(y=mean2, xmin=0.6, xmax=0.9, color='red', linestyle='--', alpha=0.7)
        ax4.fill_between([1.3, 1.7], mean1, mean2, alpha=0.3, color='gray',
                        label=f"Effect Size: {cohens_d:.2f}")
        ax4.legend()

    plt.suptitle(f'Hypothesis Testing Analysis ({test_type})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig

def create_distribution_analysis_plot(data, distribution_types=None):
    """Create comprehensive distribution analysis visualization."""
    if distribution_types is None:
        distribution_types = ['normal', 'lognormal', 'exponential', 'gamma']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # 1. Histogram with density
    sns.histplot(data, kde=True, ax=axes[0], color='skyblue', alpha=0.7)
    axes[0].set_title('Data Distribution')
    axes[0].grid(True, alpha=0.3)

    # 2. Q-Q plot
    sm.qqplot(data, line='45', ax=axes[1])
    axes[1].set_title('Q-Q Plot (Normal)')
    axes[1].grid(True, alpha=0.3)

    # 3. Box plot with outliers
    axes[2].boxplot(data, patch_artist=True,
                   boxprops=dict(facecolor='lightgreen', alpha=0.7))
    axes[2].set_title('Box Plot Analysis')
    axes[2].grid(True, alpha=0.3)

    # 4-6. Goodness of fit tests for different distributions
    data_sorted = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)

    for i, dist_name in enumerate(distribution_types[:3]):
        ax = axes[i + 3]

        if dist_name == 'normal':
            params = stats.norm.fit(data)
            fitted_dist = stats.norm(*params)
        elif dist_name == 'lognormal':
            params = stats.lognorm.fit(data)
            fitted_dist = stats.lognorm(*params)
        elif dist_name == 'exponential':
            params = stats.expon.fit(data)
            fitted_dist = stats.expon(*params)
        elif dist_name == 'gamma':
            params = stats.gamma.fit(data)
            fitted_dist = stats.gamma(*params)
        else:
            continue

        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(data, fitted_dist.cdf)

        # Plot empirical vs fitted CDF
        ax.plot(data_sorted, y, 'b-', label='Empirical', linewidth=2)
        ax.plot(data_sorted, fitted_dist.cdf(data_sorted), 'r--', label='Fitted', linewidth=2)
        ax.set_title(f'{dist_name.title()} Distribution Fit')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add test results as text
        ax.text(0.05, 0.95, f'KS test: p={ks_p:.4f}', transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray'))

    plt.suptitle('Distribution Analysis Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig

def create_regression_diagnostics_plot(X, y, model_results=None):
    """Create comprehensive regression diagnostics visualization."""
    if model_results is None:
        # Fit a simple linear regression if no model provided
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const)
        model_results = model.fit()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Get fitted values and residuals
    fitted_values = model_results.fittedvalues
    residuals = model_results.resid

    # 1. Residuals vs Fitted
    ax1.scatter(fitted_values, residuals, alpha=0.6, color='blue')
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted')
    ax1.grid(True, alpha=0.3)

    # 2. Normal Q-Q plot of residuals
    sm.qqplot(residuals, line='45', ax=ax2)
    ax2.set_title('Normal Q-Q Plot of Residuals')
    ax2.grid(True, alpha=0.3)

    # 3. Scale-Location plot
    sqrt_abs_residuals = np.sqrt(np.abs(residuals))
    ax3.scatter(fitted_values, sqrt_abs_residuals, alpha=0.6, color='green')
    ax3.set_xlabel('Fitted Values')
    ax3.set_ylabel('√|Residuals|')
    ax3.set_title('Scale-Location Plot')
    ax3.grid(True, alpha=0.3)

    # 4. Residuals vs Leverage
    leverage = model_results.get_influence().hat_matrix_diag
    ax4.scatter(leverage, residuals, alpha=0.6, color='purple')
    ax4.axhline(y=0, color='red', linestyle='--')
    ax4.set_xlabel('Leverage')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Residuals vs Leverage')
    ax4.grid(True, alpha=0.3)

    # Add Cook's distance contours if available
    try:
        cooks_d = model_results.get_influence().cooks_distance[0]
        # Highlight influential points
        influential = cooks_d > 4 / len(X)
        if any(influential):
            ax4.scatter(leverage[influential], residuals[influential],
                       color='red', s=50, alpha=0.8, label='Influential')
            ax4.legend()
    except:
        pass

    plt.suptitle('Regression Diagnostics Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig

def create_power_analysis_plot(sample_sizes, effect_sizes, alpha=0.05):
    """Create statistical power analysis visualization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Create meshgrid for power analysis
    n_grid, effect_grid = np.meshgrid(sample_sizes, effect_sizes)

    # Calculate power for different scenarios
    power_analysis = TTestPower()

    # 1. Power vs Sample Size (for different effect sizes)
    for effect_size in effect_sizes[:5]:  # Limit to 5 for clarity
        power_curve = [power_analysis.power(effect_size=effect_size, nobs=n, alpha=alpha)
                      for n in sample_sizes]
        ax1.plot(sample_sizes, power_curve, label=f'Effect size: {effect_size:.2f}', linewidth=2)

    ax1.axhline(y=0.8, color='red', linestyle='--', label='Target Power (0.8)')
    ax1.set_xlabel('Sample Size')
    ax1.set_ylabel('Statistical Power')
    ax1.set_title('Power vs Sample Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Power heatmap
    power_matrix = np.zeros_like(n_grid, dtype=float)
    for i, n in enumerate(sample_sizes):
        for j, effect in enumerate(effect_sizes):
            power_matrix[j, i] = power_analysis.power(effect_size=effect, nobs=n, alpha=alpha)

    im = ax2.imshow(power_matrix, aspect='auto', origin='lower', cmap='viridis',
                   extent=[sample_sizes.min(), sample_sizes.max(),
                          effect_sizes.min(), effect_sizes.max()])
    ax2.set_xlabel('Sample Size')
    ax2.set_ylabel('Effect Size')
    ax2.set_title('Statistical Power Heatmap')
    plt.colorbar(im, ax=ax2, label='Power')

    # 3. Required sample size for different power levels
    power_levels = [0.7, 0.8, 0.9]
    colors = ['blue', 'green', 'red']

    for power, color in zip(power_levels, colors):
        required_n = []
        for effect in effect_sizes:
            n = power_analysis.solve_power(effect_size=effect, power=power, alpha=alpha)
            required_n.append(n)

        ax3.plot(effect_sizes, required_n, color=color, linewidth=2,
                label=f'Power = {power}', marker='o')

    ax3.set_xlabel('Effect Size')
    ax3.set_ylabel('Required Sample Size')
    ax3.set_title('Sample Size Requirements')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Power analysis summary
    ax4.axis('off')

    # Calculate some example scenarios
    example_scenarios = [
        {'effect': 0.2, 'n': 100, 'name': 'Small effect, n=100'},
        {'effect': 0.5, 'n': 50, 'name': 'Medium effect, n=50'},
        {'effect': 0.8, 'n': 25, 'name': 'Large effect, n=25'}
    ]

    summary_text = """
    Statistical Power Analysis Summary:

    """

    for scenario in example_scenarios:
        power = power_analysis.power(effect_size=scenario['effect'],
                                   nobs=scenario['n'], alpha=alpha)
        summary_text += f"""
    {scenario['name']}:
    Power: {power:.3f}
    Effect Size: {scenario['effect']}
    Sample Size: {scenario['n']}
    """

    summary_text += f"""

    Significance Level (α): {alpha}
    Common Power Thresholds:
    • 0.8 (80%): Good power
    • 0.9 (90%): Excellent power

    Effect Size Guidelines (Cohen's d):
    • 0.2: Small effect
    • 0.5: Medium effect
    • 0.8: Large effect
    """

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan'))

    plt.suptitle('Statistical Power Analysis Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig