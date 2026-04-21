#!/usr/bin/env python3
"""
STABILIZED METRICS - REMOVE AMPLITUDE BOUNDARY ARTIFACTS
Compute robust metrics that don't saturate at boundaries
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

print("="*60)
print("STABILIZED METRICS ANALYSIS")
print("="*60)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
data = pd.read_csv('/home/student/sgp-tribe3/empirical_analysis/outputs/full_cohort_results.csv')

subjects = data['subject'].unique()
k_values = np.array([2, 4, 8, 16, 32, 64])

print(f"N subjects: {len(subjects)}")

# ============================================================
# STEP 2: RECONSTRUCT PROFILES
# ============================================================
print("\nReconstructing profiles...")

def sigmoid(x, A, k0, beta):
    return A / (1 + np.exp(-beta * (x - k0)))

results = []
for _, row in data.iterrows():
    # Reconstruct full profile
    profile = sigmoid(k_values, row['amplitude'], row['midpoint'], row['steepness'])
    
    # Normalize profile (per subject)
    profile_norm = profile / np.max(np.abs(profile))
    
    results.append({
        'subject': row['subject'],
        'condition': row['condition'],
        'profile_raw': profile,
        'profile_norm': profile_norm
    })

results_df = pd.DataFrame(results)

# ============================================================
# STEP 3: COMPUTE STABILIZED METRICS
# ============================================================
print("Computing stabilized metrics...")

metrics = []

for _, row in results_df.iterrows():
    profile = row['profile_raw']
    
    # Metric 1: Effective Amplitude (range)
    A_eff = np.max(profile) - np.min(profile)
    
    # Metric 2: AUC (normalized)
    profile_norm = profile / np.max(np.abs(profile))
    auc = trapezoid(profile_norm, k_values)
    auc_norm = auc / len(k_values)
    
    # Metric 3: Effective amplitude from mid-range (avoid boundaries)
    # Use k=8 to k=32 range where sigmoid is typically transitioning
    k_mid_mask = (k_values >= 8) & (k_values <= 32)
    profile_mid = profile[k_mid_mask]
    
    # A_trunc = mean of mid-range profile (more stable than saturation value)
    A_trunc = np.mean(profile_mid)
    
    metrics.append({
        'subject': row['subject'],
        'condition': row['condition'],
        'A_eff': A_eff,
        'auc_norm': auc_norm,
        'A_trunc': A_trunc
    })

metrics_df = pd.DataFrame(metrics)

# ============================================================
# STEP 4: GROUP ANALYSIS
# ============================================================
print("\n" + "="*60)
print("GROUP ANALYSIS")
print("="*60)

for metric_name in ['A_eff', 'auc_norm', 'A_trunc']:
    print(f"\n--- {metric_name} ---")
    
    task_vals = metrics_df[metrics_df['condition'] == 'task'][metric_name].values
    rest_vals = metrics_df[metrics_df['condition'] == 'rest'][metric_name].values
    
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
# STEP 5: ROBUSTNESS
# ============================================================
print("\n" + "="*60)
print("ROBUSTNESS - Leave-one-out")
print("="*60)

for metric_name in ['A_eff', 'auc_norm', 'A_trunc']:
    task_vals = metrics_df[metrics_df['condition'] == 'task'][metric_name].values
    rest_vals = metrics_df[metrics_df['condition'] == 'rest'][metric_name].values
    
    loo_sig = 0
    for i in range(len(subjects)):
        mask = np.ones(len(subjects), dtype=bool)
        mask[i] = False
        _, p = stats.ttest_rel(task_vals[mask], rest_vals[mask])
        if p < 0.05:
            loo_sig += 1
    
    print(f"{metric_name}: {loo_sig}/{len(subjects)} LOO folds significant")

# ============================================================
# STEP 6: VISUALIZATION
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for idx, metric_name in enumerate(['A_eff', 'auc_norm', 'A_trunc']):
    ax = axes[idx]
    
    task_vals = metrics_df[metrics_df['condition'] == 'task'][metric_name].values
    rest_vals = metrics_df[metrics_df['condition'] == 'rest'][metric_name].values
    
    bp = ax.boxplot([task_vals, rest_vals], tick_labels=['Task', 'Rest'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name}: Task vs Rest')

plt.tight_layout()
plt.savefig('/home/student/sgp-tribe3/empirical_analysis/figures/stabilized_metrics.png', dpi=150)
plt.close()

print("\nSaved: empirical_analysis/figures/stabilized_metrics.png")

# ============================================================
# STEP 7: SAVE RESULTS
# ============================================================
metrics_df.to_csv('/home/student/sgp-tribe3/empirical_analysis/outputs/stabilized_metrics.csv', index=False)

# Save summary statistics
with open('/home/student/sgp-tribe3/empirical_analysis/outputs/stabilized_stats.txt', 'w') as f:
    f.write("STABILIZED METRICS ANALYSIS\n")
    f.write("="*50 + "\n\n")
    f.write(f"N subjects: {len(subjects)}\n\n")
    
    for metric_name in ['A_eff', 'auc_norm', 'A_trunc']:
        task_vals = metrics_df[metrics_df['condition'] == 'task'][metric_name].values
        rest_vals = metrics_df[metrics_df['condition'] == 'rest'][metric_name].values
        t_stat, p_val = stats.ttest_rel(task_vals, rest_vals)
        diff = task_vals - rest_vals
        d = np.mean(diff) / np.std(diff, ddof=1)
        ci = stats.t.interval(0.95, len(diff)-1, loc=np.mean(diff), scale=stats.sem(diff))
        
        f.write(f"\n{metric_name}:\n")
        f.write(f"  Task:  {np.mean(task_vals):.4f} +/- {np.std(task_vals, ddof=1):.4f}\n")
        f.write(f"  Rest: {np.mean(rest_vals):.4f} +/- {np.std(rest_vals, ddof=1):.4f}\n")
        f.write(f"  Diff: {np.mean(diff):.4f}\n")
        f.write(f"  t = {t_stat:.2f}, p = {p_val:.4f}, d = {d:.2f}\n")
        f.write(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]\n")

print("Saved:")
print("  - empirical_analysis/outputs/stabilized_metrics.csv")
print("  - empirical_analysis/outputs/stabilized_stats.txt")

print("\n" + "="*60)
print("COMPLETE")
print("="*60)