#!/usr/bin/env python3
"""
SCALE-CORRECTED METRICS - Using Valid Pre-Saturation Range
Compute metrics only using k < saturation point to eliminate boundary artifacts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import os

print("="*60)
print("SCALE-CORRECTED METRICS ANALYSIS")
print("="*60)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
data = pd.read_csv('/home/student/sgp-tribe3/empirical_analysis/outputs/parameters.csv')
subjects = data['subject'].unique()

# Full k-range for analysis
k_all = np.array([2, 4, 8, 16, 32, 64])

# Valid pre-saturation range (k=2 to k=4 only)
k_valid = np.array([2, 4])

def sigmoid(x, A, k0, beta):
    return A / (1 + np.exp(-beta * (x - k0)))

# ============================================================
# STEP 2: IDENTIFY SATURATION POINT
# ============================================================
print("\nIdentifying saturation points per subject...")

results = []

for _, row in data.iterrows():
    # Reconstruct full profile
    A = row['A']
    k0 = row['k0']
    beta = row['beta']
    
    profile_all = sigmoid(k_all, A, k0, beta)
    
    # Find saturation point (first k where value hits boundary)
    saturation_idx = np.where(np.abs(profile_all) >= 999)[0]  # near -1000
    if len(saturation_idx) > 0:
        k_sat = k_all[saturation_idx[0]]
    else:
        k_sat = 64  # no saturation
    
    # Compute valid-range profile (only using k < saturation)
    profile_valid = profile_all[:2]  # k=2, k=4 only
    
    # Early-scale AUC (k=2 to k=4)
    early_auc = np.trapz(profile_valid, k_valid)
    
    # Pre-saturation slope (k2 to k4)
    slope = (profile_all[1] - profile_all[0]) / (k_all[1] - k_all[0])
    
    # Mean in valid range (k=2 to k=4)
    mean_valid = np.mean(profile_valid)
    
    results.append({
        'subject': row['subject'],
        'condition': row['condition'],
        'A': A,
        'k_sat': k_sat,
        'early_auc': early_auc,
        'slope': slope,
        'mean_valid': mean_valid
    })

results_df = pd.DataFrame(results)

print("\nSaturation points (k where amplitude hits -1000):")
for _, row in results_df.iterrows():
    print(f"  {row['subject']} ({row['condition']}): k_sat = {row['k_sat']}")

# ============================================================
# STEP 3: GROUP ANALYSIS
# ============================================================
print("\n" + "="*60)
print("GROUP ANALYSIS (Using Valid Range Only)")
print("="*60)

task_df = results_df[results_df['condition'] == 'task']
rest_df = results_df[results_df['condition'] == 'rest']

for metric in ['early_auc', 'slope', 'mean_valid']:
    print(f"\n--- {metric} ---")
    
    task_vals = task_df[metric].values
    rest_vals = rest_df[metric].values
    
    # Paired t-test
    t_stat, p_val = stats.ttest_rel(task_vals, rest_vals)
    diff = task_vals - rest_vals
    d = np.mean(diff) / np.std(diff, ddof=1)
    ci = stats.t.interval(0.95, len(diff)-1, loc=np.mean(diff), scale=stats.sem(diff))
    
    print(f"Task:  {np.mean(task_vals):.4f} +/- {np.std(task_vals, ddof=1):.4f}")
    print(f"Rest: {np.mean(rest_vals):.4f} +/- {np.std(rest_vals, ddof=1):.4f}")
    print(f"Diff: {np.mean(diff):.4f}")
    print(f"t = {t_stat:.2f}, p = {p_val:.4f}, d = {d:.2f}")
    print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

# ============================================================
# STEP 4: ROBUSTNESS
# ============================================================
print("\n" + "="*60)
print("ROBUSTNESS - Leave-one-out")
print("="*60)

for metric in ['early_auc', 'slope', 'mean_valid']:
    task_vals = task_df[metric].values
    rest_vals = rest_df[metric].values
    
    loo_sig = 0
    for i in range(len(subjects)):
        mask = np.ones(len(subjects), dtype=bool)
        mask[i] = False
        _, p = stats.ttest_rel(task_vals[mask], rest_vals[mask])
        if p < 0.05:
            loo_sig += 1
    
    print(f"{metric}: {loo_sig}/{len(subjects)} LOO folds significant")

# ============================================================
# STEP 5: VISUALIZATION
# ============================================================
os.makedirs('/home/student/sgp-tribe3/empirical_analysis/figures', exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for idx, metric in enumerate(['early_auc', 'slope', 'mean_valid']):
    ax = axes[idx]
    
    task_vals = task_df[metric].values
    rest_vals = rest_df[metric].values
    
    bp = ax.boxplot([task_vals, rest_vals], tick_labels=['Task', 'Rest'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric}: Task vs Rest (Valid Range)')

plt.tight_layout()
plt.savefig('/home/student/sgp-tribe3/empirical_analysis/figures/scale_corrected_metrics.png', dpi=150)
plt.close()

print("\nSaved: empirical_analysis/figures/scale_corrected_metrics.png")

# ============================================================
# STEP 6: SAVE RESULTS
# ============================================================
results_df.to_csv('/home/student/sgp-tribe3/empirical_analysis/outputs/scale_corrected_metrics.csv', index=False)

with open('/home/student/sgp-tribe3/empirical_analysis/outputs/scale_corrected_stats.txt', 'w') as f:
    f.write("SCALE-CORRECTED METRICS ANALYSIS\n")
    f.write("Using pre-saturation range only (k=2 to k=4)\n")
    f.write("="*50 + "\n\n")
    f.write(f"N subjects: {len(subjects)}\n\n")
    
    for metric in ['early_auc', 'slope', 'mean_valid']:
        task_vals = task_df[metric].values
        rest_vals = rest_df[metric].values
        t_stat, p_val = stats.ttest_rel(task_vals, rest_vals)
        diff = task_vals - rest_vals
        d = np.mean(diff) / np.std(diff, ddof=1)
        ci = stats.t.interval(0.95, len(diff)-1, loc=np.mean(diff), scale=stats.sem(diff))
        
        f.write(f"\n{metric}:\n")
        f.write(f"  Task:  {np.mean(task_vals):.4f} +/- {np.std(task_vals, ddof=1):.4f}\n")
        f.write(f"  Rest: {np.mean(rest_vals):.4f} +/- {np.std(rest_vals, ddof=1):.4f}\n")
        f.write(f"  Diff: {np.mean(diff):.4f}\n")
        f.write(f"  t = {t_stat:.2f}, p = {p_val:.4f}, d = {d:.2f}\n")
        f.write(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]\n")

print("Saved:")
print("  - empirical_analysis/outputs/scale_corrected_metrics.csv")
print("  - empirical_analysis/outputs/scale_corrected_stats.txt")

# ============================================================
# FINAL OUTPUT
# ============================================================
print("\n" + "="*60)
print("SCALE-CORRECTED RESULTS (Pre-Saturation)")
print("="*60)

print("\nUsing k=2 to k=4 only (avoiding REST saturation):")

for metric in ['early_auc', 'slope', 'mean_valid']:
    task_vals = task_df[metric].values
    rest_vals = rest_df[metric].values
    t_stat, p_val = stats.ttest_rel(task_vals, rest_vals)
    diff = np.mean(task_vals - rest_vals)
    d = diff / np.std(task_vals - rest_vals, ddof=1)
    
    print(f"\n{metric}:")
    print(f"  Task: {np.mean(task_vals):.4f}, Rest: {np.mean(rest_vals):.4f}")
    print(f"  Diff: {diff:.4f}, t = {t_stat:.2f}, p = {p_val:.4f}, d = {d:.2f}")