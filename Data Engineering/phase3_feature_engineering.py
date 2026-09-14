import os
import pandas as pd
import numpy as np
from itertools import combinations

print("=== Phase 3: Feature Engineering ===")
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_file = os.path.join(base_dir, 'Main_Data', 'ips_dataset_wide_windowed_v2.csv')
output_file = os.path.join(base_dir, 'Main_Data', 'ips_dataset_features_v2.csv')

df = pd.read_csv(input_file)
BEACONS = [3, 8, 33, 38]
beacon_cols = [f'RSSI_{b}' for b in BEACONS]

def kalman_filter_1d(series, Q=0.05, R=4.0):
    z = series.to_numpy(dtype=float)
    n = len(z)
    x, P = np.zeros(n), np.zeros(n)
    x[0], P[0] = z[0], 1.0
    for t in range(1, n):
        if z[t] <= -99.0:
            x[t], P[t] = -100.0, 1.0
            continue
        x_pred = z[t] if x[t-1] <= -99.0 else x[t-1]
        P_pred = 1.0 if x[t-1] <= -99.0 else P[t-1] + Q
        K = P_pred / (P_pred + R)
        x[t] = x_pred + K * (z[t] - x_pred)
        P[t] = (1 - K) * P_pred
    return x

kf_cols = [f'RSSI_{b}_KF' for b in BEACONS]
for col, kf_col in zip(beacon_cols, kf_cols):
    df[kf_col] = df.groupby(['Table', 'Side'])[col].transform(kalman_filter_1d)

rank_cols = [f'Rank_{b}' for b in BEACONS]
df[rank_cols] = df[kf_cols].rank(axis=1, ascending=False, method='min').astype(int)

pair_cols = []
for a, b in combinations(BEACONS, 2):
    col = f'Diff_{a}_{b}'
    df[col] = df[f'RSSI_{a}_KF'] - df[f'RSSI_{b}_KF']
    pair_cols.append(col)

feature_cols = kf_cols + rank_cols + pair_cols
df_final = df[['Table', 'Side', 'Sample', *feature_cols]]
df_final.to_csv(output_file, index=False)
print("Saved features to:", output_file)
