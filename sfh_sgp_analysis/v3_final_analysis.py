"""
SGP-Tribe3 V3 - Final Analysis
================================
Addressing all peer review concerns with honest reporting
"""

import numpy as np
import json
from scipy import stats
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
warnings.filterwarnings('ignore')

OUTPUT_DIR = "/home/student/sgp-tribe3/manuscript/v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)

print("=" * 80)
print("SGP-Tribe3 V3 - FINAL ANALYSIS")
print("=" * 80)

# Load data
with open('/home/student/sgp-tribe3/results/phase2_combined.json') as f:
    data = json.load(f)

results = data['results']
categories = [r['category'] for r in results]
unique_cats = sorted(set(categories))
NODE_NAMES = ['G1_broca', 'G2_wernicke', 'G3_tpj', 'G4_pfc', 'G5_dmn', 
              'G6_limbic', 'G7_sensory', 'G8_atl', 'G9_premotor']

X = np.array([[r['sgp_nodes'][node] for node in NODE_NAMES] for r in results])

# Compute chi
grand_mean = X.mean(axis=0)
delta = X - grand_mean
cov_matrix = np.cov(delta.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
leading_eigenvec = eigenvectors[:, np.argmax(eigenvalues)].real
C = delta @ leading_eigenvec
F = delta[:, 6]
chi = np.abs(C) + np.abs(F)
chi_norm = (chi - chi.min()) / (chi.max() - chi.min())

# Category-level chi
cat_chi = {}
for cat in unique_cats:
    mask = np.array(categories) == cat
    cat_chi[cat] = chi_norm[mask].mean()

cat_vals = np.array([cat_chi[c] for c in unique_cats])
median = np.median(cat_vals)

# Basins
high_chi = [c for c, v in cat_chi.items() if v > median]
low_chi = [c for c, v in cat_chi.items() if v <= median]

# Original silhouette
basin_labels = (cat_vals > median).astype(int)
orig_sil = silhouette_score(cat_vals.reshape(-1, 1), basin_labels)

# Proper permutation test
n_perms = 10000
perm_sils = []

for i in range(n_perms):
    shuffled_cats = np.random.permutation(categories)
    shuffled_chi = chi_norm.copy()
    
    perm_cat_chi = {}
    for cat in unique_cats:
        mask = np.array(shuffled_cats) == cat
        perm_cat_chi[cat] = shuffled_chi[mask].mean()
    
    perm_vals = np.array([perm_cat_chi[c] for c in unique_cats])
    perm_median = np.median(perm_vals)
    perm_labels = (perm_vals > perm_median).astype(int)
    
    if len(np.unique(perm_labels)) > 1:
        perm_sil = silhouette_score(perm_vals.reshape(-1, 1), perm_labels)
        perm_sils.append(perm_sil)

perm_sils = np.array(perm_sils)
p_perm = np.mean(perm_sils >= orig_sil)

print(f"\nResults:")
print(f"  Silhouette: {orig_sil:.4f}")
print(f"  Permutation p-value: {p_perm:.4f}")
print(f"  Basins: {len(high_chi)} high, {len(low_chi)} low")

# t-test between basins
high_vals = [cat_chi[c] for c in high_chi]
low_vals = [cat_chi[c] for c in low_chi]
t_stat, p_ttest = stats.ttest_ind(high_vals, low_vals)
cohens_d = (np.mean(high_vals) - np.mean(low_vals)) / np.sqrt((np.std(high_vals)**2 + np.std(low_vals)**2)/2)

print(f"  t-test: t={t_stat:.2f}, p={p_ttest:.2e}")
print(f"  Cohen's d: {cohens_d:.2f}")

# Sensitivity analysis
alpha_range = np.linspace(0.1, 2.0, 20)
beta_range = np.linspace(0.1, 2.0, 20)
sil_grid = np.zeros((len(alpha_range), len(beta_range)))

for i, a in enumerate(alpha_range):
    for j, b in enumerate(beta_range):
        chi_grid = a * np.abs(C) + b * np.abs(F)
        chi_grid_norm = (chi_grid - chi_grid.min()) / (chi_grid.max() - chi_grid.min())
        
        grid_cat_chi = {}
        for cat in unique_cats:
            mask = np.array(categories) == cat
            grid_cat_chi[cat] = chi_grid_norm[mask].mean()
        
        grid_vals = np.array([grid_cat_chi[c] for c in unique_cats])
        grid_median = np.median(grid_vals)
        grid_labels = (grid_vals > grid_median).astype(int)
        
        if len(np.unique(grid_labels)) > 1:
            sil_grid[i, j] = silhouette_score(grid_vals.reshape(-1, 1), grid_labels)

opt_idx = np.unravel_index(np.argmax(sil_grid), sil_grid.shape)
opt_a, opt_b = alpha_range[opt_idx[0]], beta_range[opt_idx[1]]
max_sil = sil_grid.max()

idx_1 = np.argmin(np.abs(alpha_range - 1.0))
idx_2 = np.argmin(np.abs(beta_range - 1.0))
sil_1_1 = sil_grid[idx_1, idx_2]
pct_optimal = (sil_1_1 / max_sil) * 100

print(f"\nSensitivity:")
print(f"  Optimal: alpha={opt_a:.2f}, beta={opt_b:.2f}")
print(f"  Silhouette at (1,1): {sil_1_1:.4f}")
print(f"  % of optimal: {pct_optimal:.1f}%")

# Generate figure
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1: Rankings
ax = axes[0, 0]
sorted_cats = sorted(cat_chi.items(), key=lambda x: x[1], reverse=True)
cats_sorted = [c[0] for c in sorted_cats]
chi_sorted = [c[1] for c in sorted_cats]
colors = ['#e74c3c' if c in high_chi else '#3498db' for c in cats_sorted]
ax.barh(cats_sorted[::-1], chi_sorted[::-1], color=colors[::-1])
ax.axvline(x=median, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('chi (Torsion)')
ax.set_title('Category Rankings by chi\nRed=High chi, Blue=Low chi')

# 2: Sensitivity
ax = axes[0, 1]
im = ax.imshow(sil_grid, extent=[beta_range[0], beta_range[-1], alpha_range[-1], alpha_range[0]], 
             aspect='auto', cmap='viridis', origin='upper')
ax.plot(1, 1, 'rx', markersize=15)
ax.set_xlabel('beta')
ax.set_ylabel('alpha')
ax.set_title(f'Sensitivity: {pct_optimal:.1f}% of optimal at (1,1)')
plt.colorbar(im, ax=ax, label='Silhouette')

# 3: Permutation
ax = axes[1, 0]
ax.hist(perm_sils, bins=50, alpha=0.7, color='gray')
ax.axvline(x=orig_sil, color='red', linewidth=2, label=f'Observed ({orig_sil:.3f})')
ax.set_xlabel('Silhouette')
ax.set_ylabel('Frequency')
ax.set_title(f'Permutation Test (p={p_perm:.3f})')
ax.legend()

# 4: Basin comparison
ax = axes[1, 1]
bp = ax.boxplot([high_vals, low_vals], labels=['High chi', 'Low chi'], patch_artist=True)
bp['boxes'][0].set_facecolor('#e74c3c')
bp['boxes'][1].set_facecolor('#3498db')
ax.set_ylabel('chi')
ax.set_title(f't={t_stat:.2f}, p={p_ttest:.2e}, d={cohens_d:.2f}')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figures/v3_results.png", dpi=150)
plt.close()

# Save results
v3_results = {
    'silhouette': float(orig_sil),
    'permutation_p': float(p_perm),
    't_statistic': float(t_stat),
    't_pvalue': float(p_ttest),
    'cohens_d': float(cohens_d),
    'optimal_alpha': float(opt_a),
    'optimal_beta': float(opt_b),
    'pct_optimal': float(pct_optimal),
    'high_chi_cats': sorted(high_chi),
    'low_chi_cats': sorted(low_chi),
    'threshold': float(median)
}

with open(f"{OUTPUT_DIR}/v3_results.json", 'w') as f:
    json.dump(v3_results, f, indent=2)

print(f"\nSaved to {OUTPUT_DIR}/")
print("\nV3 COMPLETE")
