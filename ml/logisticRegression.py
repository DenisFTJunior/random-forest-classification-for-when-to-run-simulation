import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib
import os
import warnings
from data.DataProcessor import DataProcessor
from data.DataTransformer import DataTransformer
warnings.filterwarnings('ignore')

class LogisticRegressionModel:
    x_train, x_test, y_train, y_test, x_val, y_val, LR  = None, None, None, None, None, None, None
    lr_best, params_best = None, None
    param_grid = [{
        'penalty':['l1','l2','elasticnet','none'],
        'C' : np.logspace(-4,4,20),
        'solver': ['lbfgs','newton-cg','liblinear','sag','saga'],
        'max_iter'  : [100,1000,2500,5000],
        'tol': [0.0001, 0.001, 0.01, 0.1, 1]    
    }]
    
    def __init__(self, data):
        data_processor = DataProcessor(data, DataTransformer.get_transformer('excel'))
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = data_processor.split_data()

    def train(self):
        # Data already split in __init__; just fit (allows re-fitting)
        self.LR = LogisticRegression(random_state=4).fit(self.x_train, self.y_train)
        return self.LR
    
    def train_with_grid(self):
        model = LogisticRegression()
        clf = GridSearchCV(model,param_grid = self.param_grid, cv = 3, verbose=True,n_jobs=-1)
        self.lr_best = clf.fit(self.x_train,self.y_train)
        self.params_best = self.lr_best.best_params_
        return self.lr_best, self.params_best
    
    def predict(self):
        return self.LR.predict(self.x_test)
    
    def predict_with_grid(self):
        return self.lr_best.predict(self.x_test)
    
    # --- Persistence ---
    def _get_fitted_estimator(self):
        if self.lr_best is not None:
            return getattr(self.lr_best, 'best_estimator_', self.lr_best)
        if self.LR is not None:
            return self.LR
        raise ValueError("LogisticRegressionModel is not trained. Call train() or train_with_grid() first.")

    def save(self, filepath: str):
        """Save the fitted estimator to the given filepath using joblib.
        If GridSearchCV was used, saves the best_estimator_."""
        estimator = self._get_fitted_estimator()
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        joblib.dump(estimator, filepath)
        return filepath
        
    
