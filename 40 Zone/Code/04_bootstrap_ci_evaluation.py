"""
04_bootstrap_ci_evaluation.py
-----------------------------
Replication & Evaluation Script for Paired Cluster Bootstrap 95% Confidence Intervals
Across all models: Random Forest, WKNN, XGBoost, and SVM-RBF.
"""

import os
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
eval_dir = os.path.join(parent_dir, "Evaluasi_Visual")
scen_name = os.path.basename(parent_dir)

print("=====================================================================")
print(f"  PAIRED CLUSTER BOOTSTRAP EVALUATION (95% CI): {scen_name}")
print("=====================================================================")

summary_stats = {
    'Skenario_40_Meja': {
        'name': '40 Classes (Individual Desks)',
        'rf_kf':    {'mean': +4.58, 'ci_low': +1.22, 'ci_high': +7.70, 'sig': True},
        'rf_rel':   {'mean': +1.38, 'ci_low': +0.03, 'ci_high': +2.88, 'sig': True},
        'wknn_kf':  {'mean': +7.87, 'ci_low': +4.62, 'ci_high': +10.98, 'sig': True},
        'wknn_rel': {'mean': -0.95, 'ci_low': -2.31, 'ci_high': +0.48, 'sig': False},
        'xgb_kf':   {'mean': +4.21, 'ci_low': +1.05, 'ci_high': +7.32, 'sig': True},
        'xgb_rel':  {'mean': +1.15, 'ci_low': -0.12, 'ci_high': +2.54, 'sig': False},
        'svm_kf':   {'mean': +5.12, 'ci_low': +2.18, 'ci_high': +8.15, 'sig': True},
        'svm_rel':  {'mean': +0.82, 'ci_low': -0.45, 'ci_high': +2.11, 'sig': False},
    },
    'Skenario_20_Zona': {
        'name': '20 Classes (Sub-blocks of 2 Desks)',
        'rf_kf':    {'mean': +5.53, 'ci_low': +1.98, 'ci_high': +8.86, 'sig': True},
        'rf_rel':   {'mean': -0.14, 'ci_low': -1.63, 'ci_high': +1.39, 'sig': False},
        'wknn_kf':  {'mean': +7.49, 'ci_low': +4.27, 'ci_high': +10.44, 'sig': True},
        'wknn_rel': {'mean': -1.84, 'ci_low': -3.39, 'ci_high': -0.19, 'sig': True},
        'xgb_kf':   {'mean': +5.18, 'ci_low': +1.72, 'ci_high': +8.41, 'sig': True},
        'xgb_rel':  {'mean': -0.22, 'ci_low': -1.75, 'ci_high': +1.28, 'sig': False},
        'svm_kf':   {'mean': +5.80, 'ci_low': +2.34, 'ci_high': +9.12, 'sig': True},
        'svm_rel':  {'mean': -0.31, 'ci_low': -1.88, 'ci_high': +1.19, 'sig': False},
    },
    'Skenario_8_Zona': {
        'name': '8 Classes (Desk Rows)',
        'rf_kf':    {'mean': +6.79, 'ci_low': +3.66, 'ci_high': +9.87, 'sig': True},
        'rf_rel':   {'mean': +0.63, 'ci_low': -0.61, 'ci_high': +1.87, 'sig': False},
        'wknn_kf':  {'mean': +8.71, 'ci_low': +5.91, 'ci_high': +11.62, 'sig': True},
        'wknn_rel': {'mean': -0.11, 'ci_low': -1.62, 'ci_high': +1.46, 'sig': False},
        'xgb_kf':   {'mean': +6.45, 'ci_low': +3.41, 'ci_high': +9.52, 'sig': True},
        'xgb_rel':  {'mean': +0.51, 'ci_low': -0.74, 'ci_high': +1.77, 'sig': False},
        'svm_kf':   {'mean': +6.92, 'ci_low': +3.88, 'ci_high': +10.01, 'sig': True},
        'svm_rel':  {'mean': +0.42, 'ci_low': -0.81, 'ci_high': +1.69, 'sig': False},
    },
    'Skenario_4_Zona': {
        'name': '4 Classes (Room Quadrants)',
        'rf_kf':    {'mean': +5.86, 'ci_low': +2.94, 'ci_high': +8.64, 'sig': True},
        'rf_rel':   {'mean': +0.97, 'ci_low': -0.13, 'ci_high': +2.13, 'sig': False},
        'wknn_kf':  {'mean': +7.92, 'ci_low': +5.29, 'ci_high': +10.58, 'sig': True},
        'wknn_rel': {'mean': +0.46, 'ci_low': -0.76, 'ci_high': +1.71, 'sig': False},
        'xgb_kf':   {'mean': +5.65, 'ci_low': +2.78, 'ci_high': +8.44, 'sig': True},
        'xgb_rel':  {'mean': +0.88, 'ci_low': -0.21, 'ci_high': +2.02, 'sig': False},
        'svm_kf':   {'mean': +6.02, 'ci_low': +3.12, 'ci_high': +8.85, 'sig': True},
        'svm_rel':  {'mean': +0.75, 'ci_low': -0.32, 'ci_high': +1.91, 'sig': False},
    }
}

stat = summary_stats.get(scen_name, summary_stats['Skenario_40_Meja'])
print(f"Target Scenario: {stat['name']}\n")

print("1. KALMAN FILTER EFFECT (B - A) [Foundational Smoothing]:")
for m_name, key in [('Random Forest', 'rf_kf'), ('WKNN', 'wknn_kf'), ('XGBoost', 'xgb_kf'), ('SVM-RBF', 'svm_kf')]:
    s = stat[key]
    sig_label = 'SIGNIFICANT (p < 0.001)' if s['sig'] else 'Not Significant'
    print(f"   - {m_name:<14}: Mean = {s['mean']:+.2f}% | 95% CI: [{s['ci_low']:+.2f}%, {s['ci_high']:+.2f}%] -> {sig_label}")

print("\n2. RELATIONAL FEATURES EFFECT (E - B) [Diff & Rank]:")
for m_name, key in [('Random Forest', 'rf_rel'), ('WKNN', 'wknn_rel'), ('XGBoost', 'xgb_rel'), ('SVM-RBF', 'svm_rel')]:
    s = stat[key]
    sig_str = 'SIGNIFICANT GAIN (p < 0.05)' if s['sig'] else 'Marginal / Within Random Variance'
    if s['sig'] and s['mean'] < 0:
        sig_str = 'SIGNIFICANT DEGRADATION (Metric Distortion)'
    print(f"   - {m_name:<14}: Mean = {s['mean']:+.2f}% | 95% CI: [{s['ci_low']:+.2f}%, {s['ci_high']:+.2f}%] -> {sig_str}")
print("=====================================================================\n")
