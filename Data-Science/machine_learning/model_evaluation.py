"""
Model Evaluation
---------------
Functions for comprehensive model evaluation and validation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

def comprehensive_classification_evaluation(model, X, y, cv_folds=5, test_size=0.2, random_state=42):
    """Comprehensive evaluation for classification models."""
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Cross-validation scores
    cv_accuracy = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
    cv_precision = cross_val_score(model, X, y, cv=cv_folds, scoring='precision_weighted')
    cv_recall = cross_val_score(model, X, y, cv=cv_folds, scoring='recall_weighted')
    cv_f1 = cross_val_score(model, X, y, cv=cv_folds, scoring='f1_weighted')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # ROC AUC (for binary classification)
    roc_auc = None
    if len(np.unique(y)) == 2 and hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    evaluation_results = {
        'test_metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        },
        'cv_metrics': {
            'accuracy': {'mean': cv_accuracy.mean(), 'std': cv_accuracy.std(), 'scores': cv_accuracy},
            'precision': {'mean': cv_precision.mean(), 'std': cv_precision.std(), 'scores': cv_precision},
            'recall': {'mean': cv_recall.mean(), 'std': cv_recall.std(), 'scores': cv_recall},
            'f1_score': {'mean': cv_f1.mean(), 'std': cv_f1.std(), 'scores': cv_f1}
        },
        'confusion_matrix': cm,
        'classification_report': class_report,
        'predictions': y_pred,
        'actual': y_test
    }
    
    return evaluation_results

def comprehensive_regression_evaluation(model, X, y, cv_folds=5, test_size=0.2, random_state=42):
    """Comprehensive evaluation for regression models."""
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Basic metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation scores
    cv_mse = -cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
    cv_mae = -cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_absolute_error')
    cv_r2 = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
    
    # Residual analysis
    residuals = y_test - y_pred
    residual_stats = {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'min': np.min(residuals),
        'max': np.max(residuals),
        'median': np.median(residuals)
    }
    
    evaluation_results = {
        'test_metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2
        },
        'cv_metrics': {
            'mse': {'mean': cv_mse.mean(), 'std': cv_mse.std(), 'scores': cv_mse},
            'mae': {'mean': cv_mae.mean(), 'std': cv_mae.std(), 'scores': cv_mae},
            'r2': {'mean': cv_r2.mean(), 'std': cv_r2.std(), 'scores': cv_r2}
        },
        'residual_analysis': residual_stats,
        'residuals': residuals,
        'predictions': y_pred,
        'actual': y_test
    }
    
    return evaluation_results

def learning_curve_analysis(model, X, y, cv_folds=5, train_sizes=None, scoring='accuracy'):
    """Generate learning curve analysis."""
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv_folds, train_sizes=train_sizes, 
        scoring=scoring, n_jobs=-1, random_state=42
    )
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    return {
        'train_sizes': train_sizes_abs,
        'train_scores': {
            'mean': train_mean,
            'std': train_std,
            'raw': train_scores
        },
        'validation_scores': {
            'mean': val_mean,
            'std': val_std,
            'raw': val_scores
        },
        'scoring_metric': scoring
    }

def validation_curve_analysis(model, X, y, param_name, param_range, cv_folds=5, scoring='accuracy'):
    """Generate validation curve analysis for hyperparameter tuning."""
    train_scores, val_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        cv=cv_folds, scoring=scoring, n_jobs=-1
    )
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Find best parameter
    best_param_idx = np.argmax(val_mean)
    best_param = param_range[best_param_idx]
    best_score = val_mean[best_param_idx]
    
    return {
        'param_range': param_range,
        'train_scores': {
            'mean': train_mean,
            'std': train_std,
            'raw': train_scores
        },
        'validation_scores': {
            'mean': val_mean,
            'std': val_std,
            'raw': val_scores
        },
        'best_param': best_param,
        'best_score': best_score,
        'param_name': param_name,
        'scoring_metric': scoring
    }

def cross_validation_detailed(model, X, y, cv_folds=5, scoring_metrics=None):
    """Detailed cross-validation with multiple metrics."""
    if scoring_metrics is None:
        # Default metrics based on problem type
        if len(np.unique(y)) < 10:  # Likely classification
            scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        else:  # Likely regression
            scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
    
    cv_results = {}
    for metric in scoring_metrics:
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring=metric)
        cv_results[metric] = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max()
        }
    
    return {
        'cv_results': cv_results,
        'cv_folds': cv_folds,
        'total_samples': len(y)
    }