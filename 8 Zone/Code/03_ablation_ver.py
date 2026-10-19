import os
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
eval_dir = os.path.join(parent_dir, "Evaluasi_Visual")
os.makedirs(eval_dir, exist_ok=True)

candidate_paths = [
    os.path.join(parent_dir, "..", "Main_Data", "ips_dataset_features_v2.csv"),
    os.path.join(parent_dir, "..", "..", "Code-20261012T064619Z-1-001", "Code", "Dataset", "ips_dataset_features_v2.csv"),
    r"d:\penelitian_ipos\Code-20261012T064619Z-1-001\Code\Dataset\ips_dataset_features_v2.csv",
    os.path.join(parent_dir, "..", "Data_Utama", "ips_dataset_features_v2.csv"),
    r"C:\Users\nakar\Downloads\Buku\Even-2526-MR-SK-WE\new_penelitian\Data_Utama\ips_dataset_features_v2.csv"
]
feat_file = next((p for p in candidate_paths if os.path.exists(p)), None)

candidate_raw = [
    os.path.join(parent_dir, "..", "..", "Code-20261012T064619Z-1-001", "Code", "Dataset", "ips_dataset_wide_windowed_v2.csv"),
    r"d:\penelitian_ipos\Code-20261012T064619Z-1-001\Code\Dataset\ips_dataset_wide_windowed_v2.csv",
    os.path.join(parent_dir, "..", "Data_Utama", "ips_dataset_wide_windowed_v2.csv"),
    r"C:\Users\nakar\Downloads\Buku\Even-2526-MR-SK-WE\new_penelitian\Data_Utama\ips_dataset_wide_windowed_v2.csv"
]
raw_file = next((p for p in candidate_raw if os.path.exists(p)), None)

df_feat = pd.read_csv(feat_file)
df_raw = pd.read_csv(raw_file)

def get_label(table):
    return f"Zona {(table - 1) // 5 + 1}"
df_feat['target'] = df_feat['Table'].apply(get_label)
df_raw['target'] = df_raw['Table'].apply(get_label)

df_feat['Row'] = df_feat.groupby(['Table', 'Side']).cumcount()
df_feat['Total'] = df_feat.groupby(['Table', 'Side'])['Sample'].transform('count')
train_mask = (df_feat['Row'] < df_feat['Total'] * 0.8)
test_mask = ~train_mask

y_train = df_feat.loc[train_mask, 'target']
y_test = df_feat.loc[test_mask, 'target']

classes = [f"Zona {i}" for i in range(1, 9)]
class_to_idx = {c: i for i, c in enumerate(classes)}
y_train_idx = np.array([class_to_idx[c] for c in y_train])
y_test_idx = np.array([class_to_idx[c] for c in y_test])

raw_cols = ['RSSI_3', 'RSSI_8', 'RSSI_33', 'RSSI_38']
kf_cols = ['RSSI_3_KF', 'RSSI_8_KF', 'RSSI_33_KF', 'RSSI_38_KF']
rank_cols = ['Rank_3', 'Rank_8', 'Rank_33', 'Rank_38']
diff_cols = ['Diff_3_8', 'Diff_3_33', 'Diff_3_38', 'Diff_8_33', 'Diff_8_38', 'Diff_33_38']

def get_model(m_key):
    if m_key == 'rf':
        return RandomForestClassifier(n_estimators=300, min_samples_split=5, min_samples_leaf=4, max_features='sqrt', max_depth=30, class_weight='balanced', random_state=42, n_jobs=-1)
    elif m_key == 'wknn':
        return KNeighborsClassifier(n_neighbors=59, weights='distance', p=1)
    elif m_key == 'xgboost':
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, tree_method='hist')
    elif m_key == 'svm':
        return SVC(kernel='rbf', C=10, gamma='scale', random_state=42)

def eval_subset(m_key, df_tr, df_te, cols):
    sc = StandardScaler()
    xtr = sc.fit_transform(df_tr[cols])
    xte = sc.transform(df_te[cols])
    m = get_model(m_key)
    if m_key == 'xgboost':
        m.fit(xtr, y_train_idx)
        return accuracy_score(y_test_idx, m.predict(xte)) * 100
    else:
        m.fit(xtr, y_train)
        return accuracy_score(y_test, m.predict(xte)) * 100

models_to_eval = [
    ('rf', 'RF', ['#cccccc', '#88ccee', '#44aa99', '#332288', '#2ca02c']),
    ('wknn', 'WKNN', ['#cccccc', '#88ccee', '#44aa99', '#332288', '#1f77b4']),
    ('xgboost', 'XGBoost', ['#cccccc', '#88ccee', '#44aa99', '#332288', '#ff7f0e']),
    ('svm', 'SVM-RBF', ['#cccccc', '#88ccee', '#44aa99', '#332288', '#9467bd']),
]

for m_key, title_name, colors in models_to_eval:
    print(f"Running Ablation Study for {title_name} (8 Label)...")
    acc_raw = eval_subset(m_key, df_raw.loc[train_mask], df_raw.loc[test_mask], raw_cols)
    acc_kf = eval_subset(m_key, df_feat.loc[train_mask], df_feat.loc[test_mask], kf_cols)
    acc_kf_rank = eval_subset(m_key, df_feat.loc[train_mask], df_feat.loc[test_mask], kf_cols + rank_cols)
    acc_kf_diff = eval_subset(m_key, df_feat.loc[train_mask], df_feat.loc[test_mask], kf_cols + diff_cols)
    acc_all = eval_subset(m_key, df_feat.loc[train_mask], df_feat.loc[test_mask], kf_cols + rank_cols + diff_cols)
    
    abl_labels = ['Raw RSSI', 'Kalman Filter\n(KF) Only', 'KF + Rank', 'KF + Diff', 'All Features\n(KF+Rank+Diff)']
    abl_accuracies = [acc_raw, acc_kf, acc_kf_rank, acc_kf_diff, acc_all]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(abl_labels, abl_accuracies, color=colors)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Ablation Study {title_name} (8 Label): Feature Contribution', fontsize=14, pad=20)
    ax.set_ylim(0, 100)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    out_img = os.path.join(eval_dir, f'ablation_study_{m_key}.png')
    plt.savefig(out_img, dpi=300)
    plt.close('all')

    latex_sub = os.path.join(r"d:\penelitian_ipos\latex\images\TrainRes", "8")
    if os.path.exists(latex_sub):
        shutil.copy2(out_img, os.path.join(latex_sub, f'ablation_study_{m_key}.png'))

print("All ablation study graphics successfully generated!")
