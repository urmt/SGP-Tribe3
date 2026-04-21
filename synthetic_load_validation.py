#!/usr/bin/env python3
"""
SFH-SGP_SYNTHETIC_LOAD_VALIDATION_01
Test dimensionality vs load scaling using simplified approach
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("SFH-SGP_SYNTHETIC_LOAD_VALIDATION_01")
print("="*60)

# ============================================================
# STEP 1: DATA GENERATION
# ============================================================
print("\nGenerating data...")

embedding_dim = 100
samples = 200
replicates = 15

load_levels = {'LOW': 5, 'MEDIUM': 15, 'HIGH': 30}

# Store dimensionality profiles
all_profiles = []

for load_name, int_dim in load_levels.items():
    for rep in range(replicates):
        Z = np.random.randn(samples, int_dim)
        W = np.random.randn(int_dim, embedding_dim)
        X = Z @ W + np.random.randn(samples, embedding_dim) * 0.1
        
        # Compute simple dimensionality proxy: mean distance to k nearest neighbors
        from sklearn.neighbors import NearestNeighbors
        k = 4
        
        # Early-scale dimensionality estimate (using k=4)
        nn = NearestNeighbors(n_neighbors=k+1)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        d_early = np.mean(distances[:, 1:])
        
        # Late-scale (using k=32)
        nn32 = NearestNeighbors(n_neighbors=33)
        nn32.fit(X)
        distances32, _ = nn32.kneighbors(X)
        d_late = np.mean(distances32[:, 1:])
        
        # Null model (random data)
        X_null = np.random.randn(samples, embedding_dim)
        nn_null = NearestNeighbors(n_neighbors=k+1)
        nn_null.fit(X_null)
        dist_null, _ = nn_null.kneighbors(X_null)
        d_null_early = np.mean(dist_null[:, 1:])
        
        # Residual (early scale)
        residual = d_early - d_null_early
        
        all_profiles.append({
            'load': load_name,
            'replicate': rep,
            'd_early': d_early,
            'd_late': d_late,
            'residual': residual
        })

results_df = pd.DataFrame(all_profiles)
print(f"Generated {len(results_df)} observations")

# ============================================================
# STEP 2: METRICS
# ============================================================
print("\nExtracting metrics...")

# Early AUC proxy (just use residual at early scale)
results_df['early_auc'] = results_df['residual']

# ============================================================
# STEP 3: STATISTICS
# ============================================================
print("\n" + "="*60)
print("STATISTICS")
print("="*60)

# Map load to numeric
load_map = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
results_df['load_num'] = results_df['load'].map(load_map)

# For each metric
for metric in ['early_auc', 'd_early']:
    print(f"\n--- {metric} ---")
    
    means = results_df.groupby('load')[metric].mean()
    stds = results_df.groupby('load')[metric].std()
    
    for load_name in ['LOW', 'MEDIUM', 'HIGH']:
        print(f"  {load_name}: {means[load_name]:.4f} ± {stds[load_name]:.4f}")
    
    # Pairwise
    for load1, load2 in [('LOW', 'MEDIUM'), ('MEDIUM', 'HIGH'), ('LOW', 'HIGH')]:
        v1 = results_df[results_df['load'] == load1][metric].values
        v2 = results_df[results_df['load'] == load2][metric].values
        t_stat, p_val = stats.ttest_ind(v1, v2)
        d = (np.mean(v1) - np.mean(v2)) / np.sqrt((np.var(v1) + np.var(v2))/2)
        print(f"  {load1} vs {load2}: diff = {np.mean(v1)-np.mean(v2):.4f}, t = {t_stat:.2f}, p = {p_val:.4f}, d = {d:.2f}")
    
    # Trend
    slope, intercept, r, p, se = stats.linregress(results_df['load_num'], results_df[metric])
    print(f"  Trend: slope = {slope:.4f}, r = {r:.2f}, p = {p:.4f}")

# ============================================================
# STEP 4: ROBUSTNESS (simplified)
# ============================================================
print("\n" + "="*60)
print("ROBUSTNESS")
print("="*60)

metric = 'early_auc'
# Bootstrap
np.random.seed(42)
boot_slopes = []
for _ in range(500):
    sample = results_df.sample(frac=1, replace=True)
    slope, _, r, p, _ = stats.linregress(sample['load_num'], sample[metric])
    boot_slopes.append(slope)

boot_slopes = np.array(boot_slopes)
print(f"Bootstrap: slope = {np.mean(boot_slopes):.4f} ± {np.std(boot_slopes):.4f}")
print(f"  95% CI: [{np.percentile(boot_slopes, 2.5):.4f}, {np.percentile(boot_slopes, 97.5):.4f}]")

# ============================================================
# STEP 5: SAVE & PLOT
# ============================================================
print("\nSaving results...")

os.makedirs('/home/student/sgp-tribe3/synthetic_analysis/load_validation', exist_ok=True)
results_df.to_csv('/home/student/sgp-tribe3/synthetic_analysis/load_validation/parameters.csv', index=False)

with open('/home/student/sgp-tribe3/synthetic_analysis/load_validation/statistics_summary.txt', 'w') as f:
    f.write("SFH-SGP_SYNTHETIC_LOAD_VALIDATION_01\n")
    f.write("="*50 + "\n\n")
    f.write(f"Replicates: {replicates}\n")
    f.write(f"Levels: LOW (dim=5), MEDIUM (dim=15), HIGH (dim=30)\n\n")
    for metric in ['early_auc', 'd_early']:
        f.write(f"\n{metric}:\n")
        means = results_df.groupby('load')[metric].mean()
        stds = results_df.groupby('load')[metric].std()
        for l in ['LOW', 'MEDIUM', 'HIGH']:
            f.write(f"  {l}: {means[l]:.4f} +/- {stds[l]:.4f}\n")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for idx, metric in enumerate(['early_auc', 'd_early']):
    ax = axes[idx]
    means = results_df.groupby('load')[metric].mean()
    stds = results_df.groupby('load')[metric].std()
    x = [1, 2, 3]
    y = [means['LOW'], means['MEDIUM'], means['HIGH']]
    yerr = [stds['LOW'], stds['MEDIUM'], stds['HIGH']]
    ax.errorbar(x, y, yerr=yerr, fmt='o-', lw=2, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(['Low', 'Medium', 'High'])
    ax.set_xlabel('Load Level')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} vs Load')

plt.tight_layout()
plt.savefig('/home/student/sgp-tribe3/synthetic_analysis/load_validation/load_gradient_plot.png', dpi=150)
plt.close()

print("\nSaved:")
print("  - parameters.csv")
print("  - statistics_summary.txt")
print("  - load_gradient_plot.png")

print("\n" + "="*60)
print("COMPLETE")
print("="*60)