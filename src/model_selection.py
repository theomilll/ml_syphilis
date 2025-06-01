# model_selection.py
# Defines parameter search spaces, cross-validation, and model training routines.

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb # XGBoost

# Seed for reproducibility as per project guidelines
import random, os
random.seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"

# --- Parameter Grids for Models (as per Section 3.7.1 of roadmap) ---

LOGISTIC_REGRESSION_PARAMS = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear'] # 'liblinear' is good for l1 penalty and smaller datasets
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'random_state': [42] # Ensure reproducibility within RF
}

XGBOOST_PARAMS = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'use_label_encoder': [False], # Suppress warning for newer XGBoost versions
    'eval_metric': ['logloss'],    # Suppress warning for newer XGBoost versions
    'random_state': [42] # Ensure reproducibility within XGBoost
}

# --- Model Training Function ---

def train_model_with_grid_search(estimator, param_grid, X_train, y_train, 
                                 scoring='recall', cv_folds=5):
    """
    Trains a model using GridSearchCV with StratifiedKFold.

    Args:
        estimator: The scikit-learn compatible estimator.
        param_grid: The dictionary of parameters to search over.
        X_train: Training features.
        y_train: Training target.
        scoring: The scoring metric for GridSearchCV (default: 'recall').
        cv_folds: Number of folds for StratifiedKFold (default: 5).

    Returns:
        best_estimator: The best model found by GridSearchCV.
        best_score: The best score achieved by the best_estimator.
    """
    print(f"\nTraining {estimator.__class__.__name__}...")
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=skf,
        scoring=scoring,
        n_jobs=-1, # Use all available cores
        verbose=1 # Add verbosity to see progress
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters for {estimator.__class__.__name__}: {grid_search.best_params_}")
    print(f"Best {scoring} score for {estimator.__class__.__name__}: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_score_

print("model_selection.py loaded with model training functions and parameter grids.")
