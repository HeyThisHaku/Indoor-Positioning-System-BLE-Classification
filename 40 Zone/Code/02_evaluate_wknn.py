import os, pandas as pd, numpy as np, joblib
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

feat_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Main_Data', 'ips_dataset_features_v2.csv'))
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Model_Prediction'))
eval_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Visual_Evaluation'))
os.makedirs(eval_dir, exist_ok=True)

df = pd.read_csv(feat_file)
def get_label(table):
    return f"Table {table}"

df['target'] = df['Table'].apply(get_label)
feature_cols = [c for c in df.columns if c not in ('Table', 'Side', 'Sample', 'Zone', 'target')]
df['Row'] = df.groupby(['Table', 'Side']).cumcount()
df['Total'] = df.groupby(['Table', 'Side'])['Sample'].transform('count')
test_mask = (df['Row'] >= df['Total'] * 0.8)

scaler = joblib.load(os.path.join(model_dir, 'scaler_wknn.joblib'))
wknn = joblib.load(os.path.join(model_dir, 'model_wknn.joblib'))

X_test = scaler.transform(df.loc[test_mask, feature_cols])
y_test = df.loc[test_mask, 'target']
y_pred = wknn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy WKNN: {acc*100:.2f}%")
