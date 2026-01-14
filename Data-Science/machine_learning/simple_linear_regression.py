"""
Simple Linear Regression
-------------------------
Simple functions to perform linear regression analysis.
"""

import pandas as pd
import numpy as np


def simple_linear_regression(X, y):
    """Perform simple linear regression using normal equation."""
    # Add bias term (intercept)
    X_with_bias = np.column_stack([np.ones(len(X)), X])
    
    # Normal equation: theta = (X^T * X)^-1 * X^T * y
    try:
        theta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        return theta
    except np.linalg.LinAlgError:
        # Use pseudoinverse if matrix is singular
        theta = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        return theta


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R-squared': r_squared
    }