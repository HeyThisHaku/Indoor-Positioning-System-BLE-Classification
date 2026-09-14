import os, pandas as pd, numpy as np, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

feat_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Main_Data', 'ips_dataset_features_v2.csv'))
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Model_Prediction'))

df = pd.read_csv(feat_file)
def get_label(table):
    return f"Table {table}"
df['target'] = df['Table'].apply(get_label)

feature_cols = [c for c in df.columns if c not in ('Table', 'Side', 'Sample', 'Zone', 'target')]
df['Row'] = df.groupby(['Table', 'Side']).cumcount()
df['Total'] = df.groupby(['Table', 'Side'])['Sample'].transform('count')
train_mask = (df['Row'] < df['Total'] * 0.8)

scaler = StandardScaler()
X_train = scaler.fit_transform(df.loc[train_mask, feature_cols])
y_train = df.loc[train_mask, 'target']

rf = RandomForestClassifier(n_estimators=300, min_samples_split=5, min_samples_leaf=4, max_features='sqrt', max_depth=30, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

os.makedirs(model_dir, exist_ok=True)
joblib.dump(rf, os.path.join(model_dir, 'model_rf.joblib'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
print("RF Model trained and saved!")
