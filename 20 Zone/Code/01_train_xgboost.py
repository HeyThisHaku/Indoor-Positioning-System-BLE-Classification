import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
model_dir = os.path.join(parent_dir, "Model_Prediction")
eval_dir = os.path.join(parent_dir, "Evaluasi_Visual")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

candidate_paths = [
    os.path.join(parent_dir, "..", "Main_Data", "ips_dataset_features_v2.csv"),
    os.path.join(parent_dir, "..", "..", "Code-20261012T064619Z-1-001", "Code", "Dataset", "ips_dataset_features_v2.csv"),
    r"d:\penelitian_ipos\Code-20261012T064619Z-1-001\Code\Dataset\ips_dataset_features_v2.csv",
    os.path.join(parent_dir, "..", "Data_Utama", "ips_dataset_features_v2.csv"),
    r"C:\Users\nakar\Downloads\Buku\Even-2526-MR-SK-WE\new_penelitian\Data_Utama\ips_dataset_features_v2.csv"
]
feat_file = next((p for p in candidate_paths if os.path.exists(p)), None)
if not feat_file:
    raise FileNotFoundError("ips_dataset_features_v2.csv could not be located in standard paths.")

df = pd.read_csv(feat_file)
def get_label(table):
    if table <= 20:
        z = table if table <= 10 else table - 10
    else:
        z = (table - 10) if table <= 30 else (table - 20)
    return f"Zona {z}"
df['target'] = df['Table'].apply(get_label)

feature_cols = [c for c in df.columns if c not in ('Table', 'Side', 'Sample', 'Zone', 'target')]
df['Row'] = df.groupby(['Table', 'Side']).cumcount()
df['Total'] = df.groupby(['Table', 'Side'])['Sample'].transform('count')

train_mask = (df['Row'] < df['Total'] * 0.8)

classes = [f"Zona {i}" for i in range(1, 21)]
class_to_idx = {c: i for i, c in enumerate(classes)}

scaler = StandardScaler()
X_train = scaler.fit_transform(df.loc[train_mask, feature_cols])
y_train_str = df.loc[train_mask, 'target']
y_train_idx = np.array([class_to_idx[c] for c in y_train_str])

xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
    random_state=42, tree_method='hist'
)
xgb.fit(X_train, y_train_idx)

joblib.dump(xgb, os.path.join(model_dir, 'model_xgboost.joblib'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler_xgboost.joblib'))
print("XGBoost Model trained and saved successfully in:", model_dir)
