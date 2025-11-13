import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import  GridSearchCV
import joblib
import os
import warnings
from data.DataProcessor import DataProcessor
from data.DataTransformer import DataTransformer

warnings.filterwarnings('ignore')

class XGBoostModel:
	xgb_best, params_best, XGB = None, None, None
	param_grid = [{
		'n_estimators': [100, 200, 300],
		'max_depth': [3, 5, 7],
		'learning_rate': [0.01, 0.05, 0.1],
		'subsample': [0.8, 1.0],
		'colsample_bytree': [0.8, 1.0],
		'gamma': [0, 1, 5]
	}]

	def __init__(self, data):
		data_processor = DataProcessor(data, DataTransformer.get_transformer('excel'))
		self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = data_processor.split_data()

	def train(self):

		self.XGB = XGBClassifier(random_state=4, use_label_encoder=False, eval_metric='logloss')
		self.XGB.fit(self.x_train, self.y_train)
		return self.XGB

	def train_with_grid(self):
		model = XGBClassifier(random_state=4, use_label_encoder=False, eval_metric='logloss')
		clf = GridSearchCV(model, param_grid=self.param_grid, cv=3, verbose=True, n_jobs=-1)
		self.xgb_best = clf.fit(self.x_train, self.y_train)
		self.params_best = self.xgb_best.best_params_
		return self.xgb_best, self.params_best

	def predict(self):
		return self.XGB.predict(self.x_test)

	def predict_with_grid(self):
		return self.xgb_best.predict(self.x_test)

	# --- Persistence ---
	def _get_fitted_estimator(self):
		if self.xgb_best is not None:
			return getattr(self.xgb_best, 'best_estimator_', self.xgb_best)
		if self.XGB is not None:
			return self.XGB
		raise ValueError("XGBoostModel is not trained. Call train() or train_with_grid() first.")

	def save(self, filepath: str):
		"""Save the fitted estimator to the given filepath using joblib.
		If GridSearchCV was used, saves the best_estimator_."""
		estimator = self._get_fitted_estimator()
		directory = os.path.dirname(filepath)
		if directory:
			os.makedirs(directory, exist_ok=True)
		joblib.dump(estimator, filepath)
		return filepath
