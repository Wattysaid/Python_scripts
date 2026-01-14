"""
Ensemble Methods
---------------
Functions for ensemble machine learning techniques.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor, BaggingClassifier, BaggingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def random_forest_analysis(X, y, task_type='classification', n_estimators=100, test_size=0.2, random_state=42):
    """Perform Random Forest analysis with feature importance."""
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    if task_type == 'classification':
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        performance = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    else:  # regression
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Performance metrics
        mse = mean_squared_error(y_test, y_pred)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        
        performance = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'cv_mean': -cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model': model,
        'performance': performance,
        'feature_importance': feature_importance,
        'predictions': y_pred,
        'test_actual': y_test
    }

def gradient_boosting_analysis(X, y, task_type='classification', n_estimators=100, learning_rate=0.1, test_size=0.2, random_state=42):
    """Perform Gradient Boosting analysis."""
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    if task_type == 'classification':
        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        performance = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        additional_results = {'prediction_probabilities': y_pred_proba}
    else:  # regression
        model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        
        performance = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'cv_mean': -cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        additional_results = {}
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Training progress
    training_scores = model.train_score_
    
    result = {
        'model': model,
        'performance': performance,
        'feature_importance': feature_importance,
        'predictions': y_pred,
        'test_actual': y_test,
        'training_scores': training_scores
    }
    result.update(additional_results)
    
    return result

def voting_ensemble(X, y, task_type='classification', test_size=0.2, random_state=42):
    """Create and evaluate voting ensemble."""
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    if task_type == 'classification':
        # Define base classifiers
        clf1 = LogisticRegression(random_state=random_state, max_iter=1000)
        clf2 = RandomForestClassifier(n_estimators=50, random_state=random_state)
        clf3 = SVC(probability=True, random_state=random_state)
        
        # Hard voting
        hard_voting = VotingClassifier(
            estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)],
            voting='hard'
        )
        
        # Soft voting
        soft_voting = VotingClassifier(
            estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)],
            voting='soft'
        )
        
        # Train and evaluate
        models = {
            'logistic_regression': clf1,
            'random_forest': clf2,
            'svm': clf3,
            'hard_voting': hard_voting,
            'soft_voting': soft_voting
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X, y, cv=5)
            
            results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
        
    else:  # regression
        # Define base regressors
        reg1 = RandomForestRegressor(n_estimators=50, random_state=random_state)
        reg2 = GradientBoostingRegressor(random_state=random_state)
        reg3 = SVR()
        
        # Voting regressor
        voting_reg = VotingRegressor(
            estimators=[('rf', reg1), ('gb', reg2), ('svr', reg3)]
        )
        
        models = {
            'random_forest': reg1,
            'gradient_boosting': reg2,
            'svr': reg3,
            'voting_ensemble': voting_reg
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            
            results[name] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'cv_mean': -cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
    
    # Find best model
    if task_type == 'classification':
        best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
        metric_name = 'accuracy'
    else:
        best_model = min(results.keys(), key=lambda x: results[x]['mse'])
        metric_name = 'mse'
    
    return {
        'results': results,
        'best_model': best_model,
        'best_metric': results[best_model][metric_name],
        'test_actual': y_test,
        'task_type': task_type
    }