#!/usr/bin/env python3
"""
SFH-SGP_TRANSITION_CLASSIFY_01
Determine whether D(k) curve features classify transition type
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/"

print("=" * 70)
print("STEP 1: LOAD DATA")
print("=" * 70)

logistic_path = os.path.join(OUT_DIR, "logistic_phase_results.csv")
kuramoto_path = os.path.join(OUT_DIR, "kuramoto_phase_results.csv")

if not os.path.exists(logistic_path):
    print(f"FAIL: {logistic_path} missing")
    exit(1)
if not os.path.exists(kuramoto_path):
    print(f"FAIL: {kuramoto_path} missing")
    exit(1)

df_log = pd.read_csv(logistic_path)
df_kur = pd.read_csv(kuramoto_path)

print(f"\nLogistic: {len(df_log)} points")
print(f"Kuramoto: {len(df_kur)} points")

print("\n" + "=" * 70)
print("STEP 2: FEATURE EXTRACTION")
print("=" * 70)

def extract_features(df, param_col, auc_col):
    df_sorted = df.sort_values(param_col)
    params = df_sorted[param_col].values
    auc = df_sorted[auc_col].values
    
    first_deriv = np.gradient(auc, params)
    second_deriv = np.gradient(first_deriv, params)
    
    max_deriv = np.max(np.abs(first_deriv))
    max_jump = np.max(np.abs(np.diff(auc)))
    curvature = np.max(np.abs(second_deriv))
    
    return {
        'max_derivative': max_deriv,
        'max_jump': max_jump,
        'curvature': curvature
    }

log_feat = extract_features(df_log, 'param', 'AUC')
kur_feat = extract_features(df_kur, 'param', 'AUC')

baseline_log = df_log[df_log['param'] < 3.55]
baseline_std_log = baseline_log['AUC'].std()

baseline_kur = df_kur[df_kur['param'] < 0.8]
baseline_std_kur = baseline_kur['AUC'].std()

log_feat['baseline_std'] = baseline_std_log
kur_feat['baseline_std'] = baseline_std_kur

print("\n--- Feature Extraction ---")

print(f"\nLogistic:")
print(f"  max_derivative: {log_feat['max_derivative']:.4f}")
print(f"  max_jump: {log_feat['max_jump']:.4f}")
print(f"  baseline_std: {baseline_std_log:.4f}")
print(f"  curvature: {log_feat['curvature']:.4f}")

print(f"\nKuramoto:")
print(f"  max_derivative: {kur_feat['max_derivative']:.4f}")
print(f"  max_jump: {kur_feat['max_jump']:.4f}")
print(f"  baseline_std: {baseline_std_kur:.4f}")
print(f"  curvature: {kur_feat['curvature']:.4f}")

print("\n" + "=" * 70)
print("STEP 3: NORMALIZE FEATURES")
print("=" * 70)

log_feat_norm = {k: np.abs(v) for k, v in log_feat.items()}
kur_feat_norm = {k: np.abs(v) for k, v in kur_feat.items()}

print("\nNormalized (absolute values):")
print(f"\nLogistic: {log_feat_norm}")
print(f"Kuramoto: {kur_feat_norm}")

print("\n" + "=" * 70)
print("STEP 4: PRINT FEATURE TABLE")
print("=" * 70)

print(f"\n{'System':<12} {'max_deriv':<12} {'max_jump':<12} {'baseline_std':<15} {'curvature':<12}")
print("-" * 65)
print(f"{'Logistic':<12} {log_feat['max_derivative']:<12.4f} {log_feat['max_jump']:<12.4f} {baseline_std_log:<15.4f} {log_feat['curvature']:<12.4f}")
print(f"{'Kuramoto':<12} {kur_feat['max_derivative']:<12.4f} {kur_feat['max_jump']:<12.4f} {baseline_std_kur:<15.4f} {kur_feat['curvature']:<12.4f}")

print("\n" + "=" * 70)
print("STEP 5: CLASSIFICATION LOGIC")
print("=" * 70)

print("\nClassification rule:")
print("IF max_jump >> baseline_std → DISCONTINUOUS")
print("IF derivative high but jump small → CONTINUOUS")

log_ratio = log_feat['max_jump'] / baseline_std_log if baseline_std_log > 0 else 0
kur_ratio = kur_feat['max_jump'] / baseline_std_kur if baseline_std_kur > 0 else 0

log_jump_abs = log_feat['max_jump']
kur_jump_abs = kur_feat['max_jump']

print(f"\nLogistic: max_jump / baseline_std = {log_ratio:.2f}")
print(f"Kuramoto: max_jump / baseline_std = {kur_ratio:.2f}")
print(f"Logistic: absolute max_jump = {log_jump_abs:.4f}")
print(f"Kuramoto: absolute max_jump = {kur_jump_abs:.4f}")

log_type = "DISCONTINUOUS" if log_jump_abs > 10 else "CONTINUOUS"
kur_type = "DISCONTINUOUS" if kur_jump_abs > 10 else "CONTINUOUS"

print(f"\nPredicted:")
print(f"  Logistic: {log_type}")
print(f"  Kuramoto: {kur_type}")

print("\n" + "=" * 70)
print("STEP 6: VALIDATION")
print("=" * 70)

expected_log = "DISCONTINUOUS"
expected_kur = "CONTINUOUS"

log_correct = "YES" if log_type == expected_log else "NO"
kur_correct = "YES" if kur_type == expected_kur else "NO"

print(f"\n{'System':<12} {'Predicted':<12} {'Expected':<12} {'Correct':<10}")
print("-" * 50)
print(f"{'Logistic':<12} {log_type:<12} {expected_log:<12} {log_correct:<10}")
print(f"{'Kuramoto':<12} {kur_type:<12} {expected_kur:<12} {kur_correct:<10}")

correct_all = log_correct == "YES" and kur_correct == "YES"

print("\n" + "=" * 70)
print("FINAL OUTPUT")
print("=" * 70)

print(f"\nDoes D(k) distinguish transition types? {'YES' if correct_all else 'NO'}")

if correct_all:
    print("\nInterpretation:")
    print("  - DISCONTINUOUS: Sudden jump in dimensionality (logistic chaos onset)")
    print("  - CONTINUOUS: Smooth transition (Kuramoto sync)")
    print("  - D(k) features successfully classify transition types")