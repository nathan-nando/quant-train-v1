import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import logging
import copy

logger = logging.getLogger(__name__)

class AnchoredWalkForwardCV:
    """
    Anchored Walk-Forward Optimization for Time Series.
    Uses an expanding window (anchored at start).
    Tracks hyperparameter stability across folds.
    """
    def __init__(self, n_splits=5, gap=0):
        self.n_splits = n_splits
        self.gap = gap
        self.tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        self.fold_results = []
        
    def split(self, X, y=None, groups=None):
        return self.tscv.split(X, y, groups)
        
    def evaluate_stability(self, model_class, param_grid, X, y, metric_fn):
        """
        Runs a grid search or similar over the expanding windows,
        and tracks which hyperparameters are selected in each fold.
        This detects if the model is overfitting to specific time regimes.
        """
        # simplified for demonstration; in reality, uses GridSearchCV per fold
        fold_idx = 1
        for train_idx, test_idx in self.split(X):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            
            # Here one would run hyperopt or grid search.
            # We mock the return of the best params for structural demonstration.
            logger.info(f"Fold {fold_idx}: Train={len(train_idx)}, Test={len(test_idx)}")
            
            fold_idx += 1
            
        return self.fold_results
