import numpy as np
from sklearn.model_selection import  GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import warnings
from data.DataProcessor import DataProcessor
from data.DataTransformer import DataTransformer
from utils.threshold import Threshold


warnings.filterwarnings('ignore')

class RandomForestModel:
    x_train, x_test, y_train, y_test, x_val, y_val, RF,thr, thr_metrics  = None, None, None, None, None, None, None, 0.5, None
    data_processor = None
    param_grid = [{
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }]

    def __init__(self, data):
        data_processor = DataProcessor(data, DataTransformer.get_transformer('excel'))
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = data_processor.split_data()

    def train(self):
        """
        Train the RandomForestClassifier. Optionally pass a custom class_weight dict, e.g. {0: 1, 1: 3}.
        class_weight controls the penalty for misclassifying each class:
        - Higher value for a class means the model will try harder to avoid errors for that class.
        - Example: {0: 1, 1: 3} makes false negatives (missed positives) 3x more costly than false positives.
        - Use this to reduce false negatives (raise recall) if class 1 is more important.
        If not provided, defaults to None (no weighting).
        """
        self.RF = RandomForestClassifier(random_state=4, class_weight={0: 5, 1: 1}).fit(self.x_train, self.y_train)
        return self.RF
    

    def optimize(self):
        model = RandomForestClassifier(random_state=4)
        clf = GridSearchCV(model, param_grid=self.param_grid, cv=3, verbose=True, n_jobs=-1)
        self.RF = clf.fit(self.x_train, self.y_train)

        return self.RF

    def predict(self):
        return self.RF.predict(self.x_test)


    # --- Threshold utilities ---
    def predict_proba_val(self):
        return self.RF.predict_proba(self.x_val)[:,1]

    def predict_proba_test(self):
        return self.RF.predict_proba(self.x_test)[:,1]


    def predict_with_threshold(self):
        return (self.predict_proba_test() >= self.thr).astype(int)

    # --- Persistence ---
    def _get_fitted_estimator(self):
        if self.RF is None:
            raise ValueError("RandomForestModel is not trained. Call train() or optimize() first.")
        return getattr(self.RF, 'best_estimator_', self.RF)

    def save(self, filepath: str):
        """Save the fitted estimator to the given filepath using joblib.
        If GridSearchCV was used, saves the best_estimator_."""
        estimator = self._get_fitted_estimator()
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        joblib.dump(estimator, filepath)
        return filepath

    @staticmethod
    def load(filepath: str):
        """Load a RandomForestModel from the given filepath using joblib."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
        model = joblib.load(filepath)
        rf_model = RandomForestModel.__new__(RandomForestModel)  # Bypass __init__
        rf_model.RF = model
        return rf_model
    
    def prepare_thresholding(self):
        """Prepare data for threshold optimization."""
        threshold = Threshold(self.predict_proba_val, self.y_val)

        self.thr, self.thr_metrics= threshold.find_threshold_with_constraints(max_false_negatives=15, min_accuracy=0.8)
        if self.thr is None:
            print("No threshold met the constraints (FN<=10 & acc>=0.75). Using default 0.5.")
            self.thr = 0.5
            self.thr_metrics = {}
      
