import os, pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
feat_file = os.path.join(base_dir, "Main_Data", "ips_dataset_features_v2.csv")
raw_file = os.path.join(base_dir, "Main_Data", "ips_dataset_wide_windowed_v2.csv")
eval_dir = os.path.join(base_dir, "4 Zone", "Visual_Evaluation")
os.makedirs(eval_dir, exist_ok=True)

df_feat = pd.read_csv(feat_file)
df_raw = pd.read_csv(raw_file)

def get_label(table):
    rem = table % 10
    is_right = (1 <= rem <= 5)
    is_front = (table <= 20)
    if is_right and is_front: return "1"
    elif not is_right and is_front: return "2"
    elif is_right and not is_front: return "3"
    else: return "4"
        
df_feat['target'] = df_feat['Table'].apply(get_label)
df_raw['target'] = df_raw['Table'].apply(get_label)

df_feat['Row'] = df_feat.groupby(['Table', 'Side']).cumcount()
df_feat['Total'] = df_feat.groupby(['Table', 'Side'])['Sample'].transform('count')
train_mask = (df_feat['Row'] < df_feat['Total'] * 0.8)
test_mask = ~train_mask

y_train = df_feat.loc[train_mask, 'target']
y_test = df_feat.loc[test_mask, 'target']

raw_cols = ['RSSI_3', 'RSSI_8', 'RSSI_33', 'RSSI_38']
kf_cols = ['RSSI_3_KF', 'RSSI_8_KF', 'RSSI_33_KF', 'RSSI_38_KF']
rank_cols = ['Rank_3', 'Rank_8', 'Rank_33', 'Rank_38']
diff_cols = ['Diff_3_8', 'Diff_3_33', 'Diff_3_38', 'Diff_8_33', 'Diff_8_38', 'Diff_33_38']

def get_model(model_name):
    if model_name == 'RF':
        return RandomForestClassifier(n_estimators=300, min_samples_split=5, min_samples_leaf=4, max_features='sqrt', max_depth=30, class_weight='balanced', random_state=42, n_jobs=-1)
    else:
        # WKNN for 4 Zone best config
        return KNeighborsClassifier(n_neighbors=93, weights='distance', p=2)

def eval_subset(model_name, df_tr, df_te, cols):
    sc = StandardScaler()
    xtr = sc.fit_transform(df_tr[cols])
    xte = sc.transform(df_te[cols])
    m = get_model(model_name)
    m.fit(xtr, y_train)
    return accuracy_score(y_test, m.predict(xte)) * 100

for model_name, title_name, colors in [('RF', 'RF', ['#cccccc', '#88ccee', '#44aa99', '#332288', '#2ca02c']), 
                                       ('WKNN', 'WKNN', ['#cccccc', '#88ccee', '#44aa99', '#332288', '#1f77b4'])]:
    print(f"Running Ablation Study for {model_name}...")
    
    acc_raw = eval_subset(model_name, df_raw.loc[train_mask], df_raw.loc[test_mask], raw_cols)
    acc_kf = eval_subset(model_name, df_feat.loc[train_mask], df_feat.loc[test_mask], kf_cols)
    acc_kf_rank = eval_subset(model_name, df_feat.loc[train_mask], df_feat.loc[test_mask], kf_cols + rank_cols)
    acc_kf_diff = eval_subset(model_name, df_feat.loc[train_mask], df_feat.loc[test_mask], kf_cols + diff_cols)
    acc_all = eval_subset(model_name, df_feat.loc[train_mask], df_feat.loc[test_mask], kf_cols + rank_cols + diff_cols)
    
    abl_labels = ['Raw RSSI', 'Kalman Filter\n(KF) Only', 'KF + Rank', 'KF + Diff', 'All Features\n(KF+Rank+Diff)']
    abl_accuracies = [acc_raw, acc_kf, acc_kf_rank, acc_kf_diff, acc_all]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(abl_labels, abl_accuracies, color=colors)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Ablation Study {title_name} (4 Label): Feature Contribution', fontsize=14, pad=20)
    ax.set_ylim(0, 100)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, f'ablation_study_{model_name.lower()}.png'), dpi=300)
    plt.close('all')


