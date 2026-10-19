import os
import shutil
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

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
    return f"Meja {table}"
df['target'] = df['Table'].apply(get_label)

feature_cols = [c for c in df.columns if c not in ('Table', 'Side', 'Sample', 'Zone', 'target')]
df['Row'] = df.groupby(['Table', 'Side']).cumcount()
df['Total'] = df.groupby(['Table', 'Side'])['Sample'].transform('count')

test_mask = (df['Row'] >= df['Total'] * 0.8)

classes = [f"Meja {i}" for i in range(1, 41)]
tick_labels = [str(i) for i in range(1, 41)]
class_to_idx = {c: i for i, c in enumerate(classes)}

scaler = joblib.load(os.path.join(model_dir, 'scaler_wknn.joblib'))
model = joblib.load(os.path.join(model_dir, 'model_wknn.joblib'))

X_test = scaler.transform(df.loc[test_mask, feature_cols])
y_test = df.loc[test_mask, 'target']

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred) * 100
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
prec = prec * 100
rec = rec * 100
f1 = f1 * 100

print(f"==================================================")
print(f"  WKNN Evaluation - 40 Label")
print(f"==================================================")
print(f"Accuracy WKNN: {acc:.2f}%")
print(f"Macro Precision: {prec:.2f}%")
print(f"Macro Recall:    {rec:.2f}%")
print(f"Macro F1-Score:  {f1:.2f}%")
print(f"==================================================")

# 1. Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred, labels=classes)
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=tick_labels, yticklabels=tick_labels,
            annot_kws={'size': 8}, cbar=True)
plt.title(f'Confusion Matrix WKNN - 40 Label\nAccuracy: {acc:.2f}%', fontsize=14, pad=15)
plt.xlabel('Predicted Zone', fontsize=11, labelpad=8)
plt.ylabel('True Zone', fontsize=11, labelpad=8)
plt.tight_layout()

cm_path = os.path.join(eval_dir, 'confusion_matrix_wknn.png')
plt.savefig(cm_path, dpi=300)
plt.close('all')

# 2. Overall Metrics Bar Plot
plt.figure(figsize=(8, 6))
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_vals = [acc, prec, rec, f1]
bars = plt.bar(metrics_names, metrics_vals, color='#1f77b4', width=0.6)
plt.title(f'Overall Performance Metrics WKNN - 40 Label', fontsize=14, pad=15)
plt.ylabel('Percentage (%)', fontsize=12)
plt.ylim(0, 100)

for b in bars:
    h = b.get_height()
    plt.text(b.get_x() + b.get_width()/2., h + 1.5, f'{h:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
metrics_path = os.path.join(eval_dir, 'overall_metrics_wknn.png')
plt.savefig(metrics_path, dpi=300)
plt.close('all')

# Sync to latex/images/TrainRes/40
latex_sub = os.path.join(r"d:\penelitian_ipos\latex\images\TrainRes", "40")
if os.path.exists(latex_sub):
    shutil.copy2(cm_path, os.path.join(latex_sub, 'confusion_matrix_wknn.png'))
    shutil.copy2(metrics_path, os.path.join(latex_sub, 'overall_metrics_wknn.png'))

print("Visualizations saved successfully in:", eval_dir)
