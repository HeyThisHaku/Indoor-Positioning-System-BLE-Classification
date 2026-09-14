import os, pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

feat_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Main_Data', 'ips_dataset_features_v2.csv'))
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

df = pd.read_csv(feat_file)
def get_label(table):
    return str((table - 1) // 5 + 1)

df['target'] = df['Table'].apply(get_label)

feature_cols = [c for c in df.columns if c not in ('Table', 'Side', 'Sample', 'Zone', 'target')]
df['Row'] = df.groupby(['Table', 'Side']).cumcount()
df['Total'] = df.groupby(['Table', 'Side'])['Sample'].transform('count')
train_mask = (df['Row'] < df['Total'] * 0.8)
test_mask = ~train_mask

scaler = StandardScaler()
X_train = scaler.fit_transform(df.loc[train_mask, feature_cols])
y_train = df.loc[train_mask, 'target']

X_test = scaler.transform(df.loc[test_mask, feature_cols])
y_test = df.loc[test_mask, 'target']

k_values = list(range(1, 202, 2)) 
acc_manhattan = []
acc_euclidean = []

print("Mulai pengujian Hyperparameter K dari 1 hingga 200 (8 Zone)...")
for k in k_values:
    wknn_m = KNeighborsClassifier(n_neighbors=k, weights='distance', p=1)
    wknn_m.fit(X_train, y_train)
    acc_m = accuracy_score(y_test, wknn_m.predict(X_test))
    acc_manhattan.append(acc_m)
    
    wknn_e = KNeighborsClassifier(n_neighbors=k, weights='distance', p=2)
    wknn_e.fit(X_train, y_train)
    acc_e = accuracy_score(y_test, wknn_e.predict(X_test))
    acc_euclidean.append(acc_e)
    print(f"K={k} | Manhattan: {acc_m*100:.2f}% | Euclidean: {acc_e*100:.2f}%")

plt.figure(figsize=(12,6))
plt.plot(k_values, [a*100 for a in acc_manhattan], label='Manhattan (p=1)', color='blue')
plt.plot(k_values, [a*100 for a in acc_euclidean], label='Euclidean (p=2)', color='red')
plt.axvline(x=160, color='gray', linestyle='--', label='Rule of Thumb: K=160 (sqrt N)')

plt.title('WKNN Accuracy vs K in 8 Label Scenario (K=1 to 200)')
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "Visual_Evaluation", "wknn_k_comparison_200.png"))
print(f"Graph saved at {os.path.join(output_dir, 'Visual_Evaluation', 'wknn_k_comparison_200.png')}")

best_m_acc = max(acc_manhattan)
best_m_k = k_values[acc_manhattan.index(best_m_acc)]
best_e_acc = max(acc_euclidean)
best_e_k = k_values[acc_euclidean.index(best_e_acc)]

print(f"\n--- BEST RESULT ---")
print(f"Euclidean (p=2) Best : K={best_e_k} with accuracy {best_e_acc*100:.2f}%")
print(f"Manhattan (p=1) Best : K={best_m_k} with accuracy {best_m_acc*100:.2f}%")
