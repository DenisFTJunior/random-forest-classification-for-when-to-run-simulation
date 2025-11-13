# IC – Constraint‑Aware ML Classification

End‑to‑end classification pipeline on tabular (Excel) data with:

- Models: Random Forest, Logistic Regression, XGBoost
- Constraint‑based threshold search (accuracy and false‑negative bounds)
- Persistence: save/load models via joblib, save predictions to CSV
- Simple prediction pipeline for single‑row inference

## Project structure

- data/: input spreadsheets
- ml/
  - randomForest.py, logisticRegression.py, xgboosting.py
  - cache/: saved models and prediction outputs
- utils/
  - threshold.py – searches a decision threshold under constraints
- pipeline/
  - run_predition.py – load/create model and write predictions
- main.py – training/evaluation entry point

## Setup (Windows + VS Code)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -U scikit-learn xgboost pandas numpy joblib
```

## Train and evaluate

```powershell
python .\main.py
```

Outputs:

- Model: ml\cache\random_forest_model.joblib
- Metrics printed to console
- Chosen threshold reported (from utils\threshold.py)

## Load a model and predict

```python
import joblib
import numpy as np

rf = joblib.load("ml/cache/random_forest_model.joblib")
X = ...  # shape (n_samples, n_features)
probs = rf.predict_proba(X)[:, 1]
thr = 0.5  # or the saved/printed threshold
preds = (probs >= thr).astype(int)
```

## Pipeline: single‑row prediction + CSV

The pipeline script anchors paths to the project root to avoid CWD issues and appends predictions to:

- ml\cache\predictions.csv (columns: id, probability, prediction, threshold, saved_at)

Run:

```powershell
python .\pipeline\run_predition.py
```

## Threshold search (HOC)

utils\threshold.py scans thresholds in [0,1] and selects the one that:

- meets accuracy ≥ min_accuracy and FN ≤ max_false_negatives
- then optimizes by lowest FN, highest accuracy, highest recall

## Notes

- Ensure inference uses the same feature order as training.
- For production, consider wrapping preprocessing + model in a Pipeline and persisting it together.
