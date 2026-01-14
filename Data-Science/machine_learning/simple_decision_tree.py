# Simple Decision Tree
import pandas as pd
import numpy as np
import sys

def gini_impurity(y):
    if len(y) == 0:
        return 0
    p = np.bincount(y) / len(y)
    return 1 - np.sum(p**2)

def split_data(X, y, feature, threshold):
    left_mask = X[:, feature] <= threshold
    return X[left_mask], y[left_mask], X[~left_mask], y[~left_mask]

def find_best_split(X, y):
    best_gini = float('inf')
    best_split = None
    
    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            X_left, y_left, X_right, y_right = split_data(X, y, feature, threshold)
            
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            
            gini_left = gini_impurity(y_left)
            gini_right = gini_impurity(y_right)
            weighted_gini = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)
            
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_split = (feature, threshold)
    
    return best_split