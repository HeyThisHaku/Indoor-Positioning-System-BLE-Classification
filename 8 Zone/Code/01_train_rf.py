import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

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
    return f"Zona {(table - 1) // 5 + 1}"
df['target'] = df['Table'].apply(get_label)

feature_cols = [c for c in df.columns if c not in ('Table', 'Side', 'Sample', 'Zone', 'target')]
df['Row'] = df.groupby(['Table', 'Side']).cumcount()
df['Total'] = df.groupby(['Table', 'Side'])['Sample'].transform('count')

train_mask = (df['Row'] < df['Total'] * 0.8)

scaler = StandardScaler()
X_train = scaler.fit_transform(df.loc[train_mask, feature_cols])
y_train = df.loc[train_mask, 'target']

rf = RandomForestClassifier(
    n_estimators=300, min_samples_split=5, min_samples_leaf=4,
    max_features='sqrt', max_depth=30, class_weight='balanced',
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

joblib.dump(rf, os.path.join(model_dir, 'model_rf.joblib'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
print("RF Model trained and saved successfully in:", model_dir)
