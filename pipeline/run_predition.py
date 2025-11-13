import os, sys, csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.randomForest import RandomForestModel
from sklearn.preprocessing import StandardScaler
from utils.data import read_row_by_id, _read_table
import numpy as np

# Anchor all paths to project root to avoid CWD issues
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join(ROOT_DIR, 'data', 'Resultados_Patricia-Rodada-03.xlsx')

def create_or_load_rf(): 
    file = "random_forest_model.joblib"
    model_dir = os.path.join(ROOT_DIR, 'ml', 'cache')
    model_path = os.path.join(model_dir, file)
    if os.path.exists(model_path):
        print(f"Loading Random Forest model from {model_path} ...")
        return RandomForestModel.load(model_path)
    # Train a new model and save with consistent filename
    print("Training new Random Forest model...")
    rf = RandomForestModel(data_path)
    rf.train()
    rf.prepare_thresholding()
    os.makedirs(model_dir, exist_ok=True)
    rf.save(model_path)
    return rf
    
def prepare_row(id_value):
    data = _read_table(data_path)
    delta_col = 'Delta yaw neutro'
    base_cols = [c for c in data.columns if (c.startswith('R') or c.startswith('P')) and c != 'Peso do sistema']
    row = read_row_by_id(data_path, id_value)
    if row is None:
        raise ValueError(f"ID {id_value} not found.")
    # Build feature vector in training order
    if row[delta_col] > 3: 
        print(f"Warning: The delta value for ID {id_value} is less than 3, not predicting.")
        return None
    feature_cols = base_cols + [delta_col]              # matches DataProcessor (base + delta)

    try:
        input_values = [row[c] for c in feature_cols]
    except KeyError as e:
        raise KeyError(f"Missing column in row: {e}")

    x_raw = np.array(input_values, dtype=float).reshape(1, -1)

    try:
        data = data[data[delta_col] < 3]
        x_all = data[feature_cols].copy()
    except KeyError as e:
        raise KeyError(f"While fitting scaler missing column: {e}")
    scaler = StandardScaler().fit(x_all)

    x_scaled = scaler.transform(x_raw)
    return x_scaled

def _ensure_dir(path: str):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

def save_prediction_row(output_csv: str, row: dict):
    _ensure_dir(output_csv)
    write_header = not os.path.exists(output_csv)
    with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def predict_with_rf(id_value=6869): 
    input_data = prepare_row(id_value)
    print(input_data)
    rf = create_or_load_rf()
    # input_data is already scaled 2D array

    y_pred_rf = rf.predict_with_threshold()

    
    return y_pred_rf.ravel() if hasattr(y_pred_rf, 'ravel') else y_pred_rf

predict_with_rf()