import os
from ml.evaluate import EvaluateModel
import numpy as np

# Prepare data
data = os.path.join('data', 'Resultados_Patricia-Rodada-03.xlsx')

# Logistic Regression Model
# from ml.logisticRegression import LogisticRegressionModel
# model = LogisticRegressionModel(data)
# model.train()
# y_pred = model.predict()

# garantir que não mate o que está + Infinito
# model.train_with_grid()
# y_pred_best = model.predict_with_grid()

# metrics = EvaluateModel.classification_metrics(model.y_test,y_pred)
# metrics_best = EvaluateModel.classification_metrics(model.y_test,y_pred_best)
# print("Metrics for Logistic Regression", metrics)
# print("Metrics for Best Logistic Regression", metrics_best)

# Random Forest Model
from ml.randomForest import RandomForestModel
rf_model = RandomForestModel(data)
rf_model.train()
rf_model.prepare_thresholding()
# rf_model.optimize()

rf_model.save('ml/cache/random_forest_model.joblib')

y_pred_rf = rf_model.predict_with_threshold()
metrics_rf = EvaluateModel.classification_metrics(rf_model.y_test, y_pred_rf)
val_preds = (rf_model.predict_proba_val() >= rf_model.thr).astype(int)
metrics_rf_val = EvaluateModel.classification_metrics(rf_model.y_val, val_preds)
print(f"Chosen threshold: {rf_model.thr:.3f}")
if rf_model.thr_metrics:
    print("Validation constraint metrics:", rf_model.thr_metrics)
print("Metrics for Random Forest (Test)", metrics_rf)
print("Metrics for Random Forest (Validation)", metrics_rf_val)

# rf_model.train_with_grid()
# y_pred_rf_best = rf_model.predict_with_grid()
# metrics_rf_best = EvaluateModel.classification_metrics(rf_model.y_test, y_pred_rf_best)
# print("Metrics for Best Random Forest", metrics_rf_best)

# XGBoost Model
# from ml.xgboosting import XGBoostModel
# xgb_model = XGBoostModel(data)
# xgb_model.train()
# y_pred_xgb = xgb_model.predict()

# metrics_xgb = EvaluateModel.classification_metrics(xgb_model.y_test, y_pred_xgb)
# metrics_xgb_val = EvaluateModel.classification_metrics(xgb_model.y_val, xgb_model.XGB.predict(xgb_model.x_val))
# print("Metrics for XGBoost on Validation Set", metrics_xgb_val)
# print("Metrics for XGBoost", metrics_xgb)

# xgb_model.train_with_grid()
# y_pred_xgb_best = xgb_model.predict_with_grid()
# metrics_xgb_best = EvaluateModel.classification_metrics(xgb_model.y_test, y_pred_xgb_best)
# print("Metrics for Best XGBoost", metrics_xgb_best)

        