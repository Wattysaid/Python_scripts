"""
Regression Analysis
------------------
Advanced regression analysis functions.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def simple_regression_stats(x, y):
    """Perform simple linear regression with detailed statistics."""
    # Remove any NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    
    n = len(x_clean)
    y_pred = slope * x_clean + intercept
    residuals = y_clean - y_pred
    
    result = {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'r_squared': r_value**2,
        'p_value': p_value,
        'standard_error': std_err,
        'n_observations': n,
        'residuals': residuals,
        'predicted': y_pred,
        'mse': mean_squared_error(y_clean, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_clean, y_pred))
    }
    return result

def multiple_regression_stats(X, y):
    """Perform multiple regression analysis."""
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    n = X.shape[0]
    p = X.shape[1]
    
    # Calculate R-squared and adjusted R-squared
    r2 = r2_score(y, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    result = {
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'r_squared': r2,
        'adjusted_r_squared': adj_r2,
        'n_observations': n,
        'n_features': p,
        'residuals': residuals,
        'predicted': y_pred,
        'mse': mean_squared_error(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred))
    }
    return result

def analyze_residuals(residuals):
    """Analyze regression residuals."""
    result = {
        'mean_residuals': np.mean(residuals),
        'std_residuals': np.std(residuals),
        'normality_test': stats.shapiro(residuals),
        'durbin_watson': np.sum(np.diff(residuals)**2) / np.sum(residuals**2),
        'residual_plots_data': {
            'residuals': residuals,
            'abs_residuals': np.abs(residuals),
            'standardized_residuals': residuals / np.std(residuals)
        }
    }
    return result