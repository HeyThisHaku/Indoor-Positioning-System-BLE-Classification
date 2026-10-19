import os, sys, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Lokasi file dataset
feat_file = r"C:\Users\nakar\Downloads\Buku\Even-2526-MR-SK-WE\new_penelitian\Data_Utama\ips_dataset_features_v2.csv"
if not os.path.exists(feat_file):
    alt_file = r"d:\penelitian_ipos\Code-20261012T064619Z-1-001\Code\Dataset\ips_dataset_features_v1.csv"
    if os.path.exists(alt_file):
        feat_file = alt_file

# Lokasi output folder (relatif terhadap folder skenario ini)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
eval_dir = os.path.join(parent_dir, "Evaluasi_Visual")
os.makedirs(eval_dir, exist_ok=True)

print(f"=== RE-TUNING WKNN BERSIH (INNER-VALIDATION): 8 Zona (Row-Level Zoning) ===")
print(f"Loading data: {{feat_file}}")
df = pd.read_csv(feat_file)

# 1. Label target untuk skenario ini
def get_label(table):
    return f"Zona {(table - 1) // 5 + 1}"

df['target'] = df['Table'].apply(get_label)

# 2. 14 Fitur model (tidak menyertakan metadata Table, Side, Sample)
feature_cols = [c for c in df.columns if c not in ('Table', 'Side', 'Sample', 'Zone', 'target')]
df['Row'] = df.groupby(['Table', 'Side']).cumcount()
df['Total'] = df.groupby(['Table', 'Side'])['Sample'].transform('count')

# 3. Time-based Split: 80% Train dan 20% Held-Out Test
train_mask = (df['Row'] < df['Total'] * 0.8)
test_mask = ~train_mask

df_tr = df[train_mask].copy()
# Inner Validation Split di dalam Train (80% Fit, 20% Validation)
val_mask_tr = (df_tr['Row'] >= df_tr['Total'] * 0.8 * 0.8)
fit_mask_tr = ~val_mask_tr

print(f"Total Sampel      : {{len(df)}}")
print(f"Train Set (80%)   : {{train_mask.sum()}} baris")
print(f"Test Set (20%)    : {{test_mask.sum()}} baris (DIKUNCI / TIDAK DISENTUH SAAT TUNING)")
print(f"  -> Inner Fit    : {{fit_mask_tr.sum()}} baris")
print(f"  -> Inner Val    : {{val_mask_tr.sum()}} baris (Murni untuk memilih K dan p)\n")

# 4. Standardisasi HANYA pada Fit subset
scaler_inner = StandardScaler()
X_fit = scaler_inner.fit_transform(df.loc[train_mask].loc[fit_mask_tr, feature_cols])
y_fit = df.loc[train_mask].loc[fit_mask_tr, 'target'].values

X_val = scaler_inner.transform(df.loc[train_mask].loc[val_mask_tr, feature_cols])
y_val = df.loc[train_mask].loc[val_mask_tr, 'target'].values

# 5. Grid Search K (ganjil 1..201) dan p (1 = Manhattan, 2 = Euclidean) pada Validation subset
k_values = list(range(1, 202, 2))
acc_manhattan = []
acc_euclidean = []

print("Mulai Grid Search K=1 s.d. 201 pada Inner Validation Set...")
t0 = time.time()
best_val_acc = -1.0
best_k = None
best_p = None

for k in k_values:
    # Manhattan (p=1)
    knn_m = KNeighborsClassifier(n_neighbors=k, weights='distance', p=1, n_jobs=-1)
    knn_m.fit(X_fit, y_fit)
    acc_m = accuracy_score(y_val, knn_m.predict(X_val)) * 100
    acc_manhattan.append(acc_m)
    if acc_m > best_val_acc:
        best_val_acc = acc_m
        best_k = k
        best_p = 1

    # Euclidean (p=2)
    knn_e = KNeighborsClassifier(n_neighbors=k, weights='distance', p=2, n_jobs=-1)
    knn_e.fit(X_fit, y_fit)
    acc_e = accuracy_score(y_val, knn_e.predict(X_val)) * 100
    acc_euclidean.append(acc_e)
    if acc_e > best_val_acc:
        best_val_acc = acc_e
        best_k = k
        best_p = 2

    if k in [1, 25, 51, 75, 101, 151, 201] or k == best_k:
        print(f"  K={{k:3d}} | Val Manhattan: {{acc_m:5.2f}}% | Val Euclidean: {{acc_e:5.2f}}%")

best_metric_str = "Manhattan (p=1)" if best_p == 1 else "Euclidean (p=2)"
print(f"\nSelesai Grid Search ({{time.time()-t0:.1f}}s):")
print(f"  >>> Parameter Terbaik dari Validasi: K={{best_k}}, Metrik={{best_metric_str}}")
print(f"  >>> Akurasi Validasi Terbaik       : {{best_val_acc:.2f}}%\n")

# 6. Simpan Grafik Evaluasi Validasi (Pengganti Figure 9 di Paper)
plt.figure(figsize=(10, 5))
plt.plot(k_values, acc_manhattan, label='Manhattan (p=1)', color='blue')
plt.plot(k_values, acc_euclidean, label='Euclidean (p=2)', color='red')
plt.axvline(x=best_k, color='green', linestyle='--', label=f'Best K={{best_k}} ({{best_metric_str}})')
plt.title(f'WKNN Validation Accuracy vs K (8 Zona (Row-Level Zoning))')
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('Validation Accuracy (%)')
plt.legend()
plt.grid(True)
plot_path = os.path.join(eval_dir, "wknn_k_validation_tuning.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(eval_dir, '8wknn_k_comparison_200.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(eval_dir, 'wknn_k_comparison_200.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Grafik tuning tersimpan di: {{plot_path}}")

# 7. Refit Model ke SELURUH Train Set (25,657 baris) dengan K & p Terpilih
print("\nMelatih ulang model final pada seluruh Data Train...")
scaler_full = StandardScaler()
X_train_full = scaler_full.fit_transform(df.loc[train_mask, feature_cols])
y_train_full = df.loc[train_mask, 'target'].values

final_model = KNeighborsClassifier(n_neighbors=best_k, weights='distance', p=best_p, n_jobs=-1)
final_model.fit(X_train_full, y_train_full)

# 8. Uji SATU KALI pada Held-Out Test Set (6,315 baris)
X_test_scaled = scaler_full.transform(df.loc[test_mask, feature_cols])
y_test = df.loc[test_mask, 'target'].values
y_pred_test = final_model.predict(X_test_scaled)

test_acc = accuracy_score(y_test, y_pred_test) * 100
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average='macro')

print("="*60)
print(f"HASIL AKHIR PADA HELD-OUT TEST SET (8 Zona (Row-Level Zoning)):")
print(f"  - Test Accuracy : {{test_acc:.2f}}%")
print(f"  - Macro Precision: {{prec*100:.2f}}%")
print(f"  - Macro Recall   : {{rec*100:.2f}}%")
print(f"  - Macro F1-Score : {{f1*100:.2f}}%")
print("="*60)
