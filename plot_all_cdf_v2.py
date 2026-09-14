import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = os.path.dirname(os.path.abspath(__file__))
feat_file = os.path.join(base_dir, "Main_Data", "ips_dataset_features_v2.csv")


print("Loading dataset...")
df_master = pd.read_csv(feat_file)
feature_cols = [c for c in df_master.columns if c not in ('Table', 'Side', 'Sample', 'Zone', 'target')]
df_master['Row'] = df_master.groupby(['Table', 'Side']).cumcount()
df_master['Total'] = df_master.groupby(['Table', 'Side'])['Sample'].transform('count')
test_mask = (df_master['Row'] >= df_master['Total'] * 0.8)

scenarios = [
    {
        "name": "40 Labels", "folder": "40 Zone", "img_dir": "40",
        "map_grid": [
            [10, 9, 8, 7, 6, None, None, 5, 4, 3, 2, 1],
            [20, 19, 18, 17, 16, None, None, 15, 14, 13, 12, 11],
            [30, 29, 28, 27, 26, None, None, 25, 24, 23, 22, 21],
            [40, 39, 38, 37, 36, None, None, 35, 34, 33, 32, 31]
        ],
        "cell_width": 0.8,
        "cell_height": 0.72,
        "format_label": lambda x: f"Table {x}"
    },
    {
        "name": "20 Labels", "folder": "20 Zone", "img_dir": "20",
        "map_grid": [
            [10, 9, 8, 7, 6, None, None, 5, 4, 3, 2, 1],
            [20, 19, 18, 17, 16, None, None, 15, 14, 13, 12, 11]
        ],
        "cell_width": 0.8,
        "cell_height": 1.44,
        "format_label": lambda x: f"Zone {x}"
    },
    {
        "name": "8 Labels", "folder": "8 Zone", "img_dir": "8",
        "map_grid": [
            [2, None, None, 1],
            [4, None, None, 3],
            [6, None, None, 5],
            [8, None, None, 7]
        ],
        "cell_width": 1.6 / 3.0,
        "cell_height": 0.72,
        "format_label": lambda x: f"Zone {x}"
    },
    {
        "name": "4 Labels", "folder": "4 Zone", "img_dir": "4",
        "map_grid": [
            [2, None, None, 1],
            [4, None, None, 3]
        ],
        "cell_width": 1.6 / 3.0,
        "cell_height": 1.44,
        "format_label": lambda x: {1: "Right Front", 2: "Left Front", 3: "Right Back", 4: "Left Back"}[x]
    }
]

# Note: The mapping function for data generation (to evaluate model against real targets)
# Still needs to map original tables (1-40) to their zones.
def get_label_40(table): return f"Table {table}"
def get_label_20(table):
    if table <= 20: z = table if table <= 10 else table - 10
    else: z = (table - 10) if table <= 30 else (table - 20)
    return f"Zone {z}"
def get_label_8(table): return f"Zone {(table - 1) // 5 + 1}"
def get_label_4(table):
    rem = table % 10
    is_right = (1 <= rem <= 5)
    is_front = (table <= 20)
    if is_right and is_front: return "Right Front"
    elif not is_right and is_front: return "Left Front"
    elif is_right and not is_front: return "Right Back"
    else: return "Left Back"

target_fns = {
    "40 Labels": get_label_40,
    "20 Labels": get_label_20,
    "8 Labels": get_label_8,
    "4 Labels": get_label_4
}

for sc in scenarios:
    print(f"\n--- Processing {sc['name']} ---")
    model_dir = os.path.join(base_dir, sc['folder'], "Model_Prediction")
    # Change to Evaluasi_Visual inside each Skenario folder
    img_out_dir = os.path.join(base_dir, sc['folder'], "Visual_Evaluation")
    os.makedirs(img_out_dir, exist_ok=True)
    
    # Calculate Explicit Zone Coordinates based on the explicit map_grid
    zone_coords = {}
    for r, row in enumerate(sc['map_grid']):
        for c, val in enumerate(row):
            if val is not None:
                lbl = sc['format_label'](val)
                zone_coords[lbl] = np.array([c * sc['cell_width'], r * sc['cell_height']])
    
    # Data target mappings
    df = df_master.copy()
    df['target'] = df['Table'].apply(target_fns[sc['name']])
    y_test = df.loc[test_mask, 'target']
    
    # Load Models
    try:
        scaler_rf = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
        rf = joblib.load(os.path.join(model_dir, 'model_rf.joblib'))
    except Exception as e:
        print(f"Skipping RF for {sc['name']}: {e}")
        continue
        
    try:
        scaler_wknn_path = os.path.join(model_dir, 'scaler_wknn.joblib')
        if not os.path.exists(scaler_wknn_path):
            scaler_wknn_path = os.path.join(model_dir, 'scaler.joblib')
        scaler_wknn = joblib.load(scaler_wknn_path)
        wknn = joblib.load(os.path.join(model_dir, 'model_wknn.joblib'))
    except Exception as e:
        print(f"Skipping WKNN for {sc['name']}: {e}")
        continue

    X_test_rf = scaler_rf.transform(df.loc[test_mask, feature_cols])
    X_test_wknn = scaler_wknn.transform(df.loc[test_mask, feature_cols])
    
    # ===== PLOT CDF CONFIDENCE =====
    print("Plotting CDF Confidence...")
    rf_probs = rf.predict_proba(X_test_rf)
    wknn_probs = wknn.predict_proba(X_test_wknn)

    rf_classes = list(rf.classes_)
    wknn_classes = list(wknn.classes_)

    rf_true_probs = [rf_probs[i, rf_classes.index(y)] for i, y in enumerate(y_test)]
    wknn_true_probs = [wknn_probs[i, wknn_classes.index(y)] for i, y in enumerate(y_test)]

    plt.figure(figsize=(8, 6))
    sns.ecdfplot(data=rf_true_probs, label='Random Forest', color='#2ca02c')
    sns.ecdfplot(data=wknn_true_probs, label='WKNN', color='#1f77b4')
    plt.title(f'CDF of True Class Prediction Probability ({sc["name"]})', fontsize=14)
    plt.xlabel('Prediction Probability (Confidence)', fontsize=12)
    plt.ylabel('Cumulative Distribution (CDF)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(img_out_dir, 'cdf_confidence.png'), dpi=300)
    plt.close()
    
    # ===== PLOT CDF LOCALIZATION ERROR =====
    print("Plotting CDF Localization Error...")
    rf_preds = rf.predict(X_test_rf)
    wknn_preds = wknn.predict(X_test_wknn)

    def get_distance_error(y_true, y_pred, coords):
        errors = []
        for t, p in zip(y_true, y_pred):
            if t in coords and p in coords:
                errors.append(np.linalg.norm(coords[t] - coords[p]))
            else:
                errors.append(0.0)
        return errors

    rf_errors = get_distance_error(y_test, rf_preds, zone_coords)
    wknn_errors = get_distance_error(y_test, wknn_preds, zone_coords)

    plt.figure(figsize=(8, 6))
    sns.ecdfplot(data=rf_errors, label='Random Forest', color='#2ca02c')
    sns.ecdfplot(data=wknn_errors, label='WKNN', color='#1f77b4')
    plt.title(f'CDF of Localization Error ({sc["name"]})', fontsize=14)
    plt.xlabel('Distance Error (meters)', fontsize=12)
    plt.ylabel('Cumulative Probability (CDF)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(img_out_dir, 'cdf_localization_error.png'), dpi=300)
    plt.close()
    
print("\nAll scenarios generated successfully.")
