#!/usr/bin/env python3
"""
SFH-SGP_VALIDATION_STUDY_01 - NON-PARAMETRIC AUC ANALYSIS
Validates residual structure without sigmoid fitting
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

# ============================================================
# STEP 1 — LOAD DATA
# ============================================================
print("="*60)
print("STEP 1: LOAD DATA")
print("="*60)

data = pd.read_csv('/home/student/sgp-tribe3/empirical_analysis/outputs/parameters.csv')

subjects = data['subject'].unique()
k_values = np.array([2, 4, 8, 16, 32, 64])

print(f"N subjects: {len(subjects)}")
print(f"k-values: {k_values}")

# ============================================================
# STEP 2 — RECONSTRUCT RESIDUAL CURVES
# ============================================================
print("\n" + "="*60)
print("STEP 2: RECONSTRUCT RESIDUAL CURVES")
print("="*60)

# Reconstruct curves from sigmoid parameters
# D(k) = A / (1 + exp(-beta*(k - k0)))

def reconstruct_profile(A, k0, beta, k_values):
    """Reconstruct dimensionality profile from sigmoid parameters"""
    return A / (1 + np.exp(-beta * (k_values - k0)))

all_curves = []

for _, row in data.iterrows():
    subject = row['subject']
    condition = row['condition']
    A = row['A']
    k0 = row['k0']
    beta = row['beta']
    
    # Reconstruct profile
    profile = reconstruct_profile(A, k0, beta, k_values)
    
    all_curves.append({
        'subject': subject,
        'condition': condition,
        **{f'k{k}': v for k, v in zip(k_values, profile)}
    })

curves_df = pd.DataFrame(all_curves)

print("\nSample residual curves:")
print(curves_df.head())

# ============================================================
# STEP 3 — NORMALIZE PER SUBJECT
# ============================================================
print("\n" + "="*60)
print("STEP 3: NORMALIZE PER SUBJECT")
print("="*60)

# Option A: Scale to [-1, 1] range per subject
k_cols = [f'k{k}' for k in k_values]

for idx, row in curves_df.iterrows():
    values = row[k_cols].values
    max_abs = np.max(np.abs(values))
    if max_abs > 0:
        curves_df.loc[idx, k_cols] = values / max_abs

print("Normalized curves (sample):")
print(curves_df[k_cols].head())

# ============================================================
# STEP 4 — COMPUTE AUC
# ============================================================
print("\n" + "="*60)
print("STEP 4: COMPUTE AUC")
print("="*60)

# Compute AUC for each subject and condition
auc_values = []

for idx, row in curves_df.iterrows():
    values = row[k_cols].values
    # Trapezoidal integration
    auc = trapezoid(values, k_values)
    auc_values.append(auc)

curves_df['AUC'] = auc_values

print("\nAUC values:")
print(curves_df[['subject', 'condition', 'AUC']])

# ============================================================
# STEP 5 — STATISTICS
# ============================================================
print("\n" + "="*60)
print("STEP 5: STATISTICS")
print("="*60)

task_auc = curves_df[curves_df['condition'] == 'task']['AUC'].values
rest_auc = curves_df[curves_df['condition'] == 'rest']['AUC'].values

# Paired t-test
t_stat, p_val = stats.ttest_rel(task_auc, rest_auc)
diff = task_auc - rest_auc
mean_diff = np.mean(diff)
se_diff = np.std(diff, ddof=1) / np.sqrt(len(diff))
cohens_d = mean_diff / np.std(diff, ddof=1)
ci_low = mean_diff - 1.96 * se_diff
ci_high = mean_diff + 1.96 * se_diff

print(f"\nAUC Results:")
print(f"  Task:  {np.mean(task_auc):.4f} ± {np.std(task_auc, ddof=1):.4f}")
print(f"  Rest: {np.mean(rest_auc):.4f} ± {np.std(rest_auc, ddof=1):.4f}")
print(f"  Diff: {mean_diff:.4f}")
print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"  t = {t_stat:.2f}, p = {p_val:.6f}")
print(f"  Cohen's d = {cohens_d:.2f}")

# ============================================================
# STEP 6 — ROBUSTNESS
# ============================================================
print("\n" + "="*60)
print("STEP 6: ROBUSTNESS")
print("="*60)

# Bootstrap
np.random.seed(42)
n_bootstrap = 1000

bootstrap_diffs = []
for _ in range(n_bootstrap):
    idx = np.random.randint(0, len(task_auc), len(task_auc))
    boot_diff = np.mean(task_auc[idx] - rest_auc[idx])
    bootstrap_diffs.append(boot_diff)

bootstrap_diffs = np.array(bootstrap_diffs)
boot_mean = np.mean(bootstrap_diffs)
boot_se = np.std(bootstrap_diffs)
boot_ci = [np.percentile(bootstrap_diffs, 2.5), np.percentile(bootstrap_diffs, 97.5)]

print(f"\nBootstrap:")
print(f"  Mean: {boot_mean:.4f}")
print(f"  SE: {boot_se:.4f}")
print(f"  95% CI: [{boot_ci[0]:.4f}, {boot_ci[1]:.4f}]")

# Leave-one-out
print("\nLeave-one-out:")
for i in range(len(subjects)):
    mask = np.ones(len(subjects), dtype=bool)
    mask[i] = False
    loo_t, loo_p = stats.ttest_rel(task_auc[mask], rest_auc[mask])
    print(f"  Exclude {subjects[i]}: t = {loo_t:.2f}, p = {loo_p:.4f}")

# ============================================================
# STEP 7 — VISUALIZATION
# ============================================================
print("\n" + "="*60)
print("STEP 7: VISUALIZATION")
print("="*60)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Boxplot
ax1 = axes[0]
bp = ax1.boxplot([task_auc, rest_auc], labels=['Task', 'Rest'], patch_artist=True)
bp['boxes'][0].set_facecolor('#2ecc71')
bp['boxes'][1].set_facecolor('#e74c3c')
ax1.set_ylabel('AUC (normalized)')
ax1.set_title('AUC: Task vs Rest')

# Paired lines
ax2 = axes[1]
for i, subj in enumerate(subjects):
    task_val = curves_df[(curves_df['subject'] == subj) & (curves_df['condition'] == 'task')]['AUC'].values[0]
    rest_val = curves_df[(curves_df['subject'] == subj) & (curves_df['condition'] == 'rest')]['AUC'].values[0]
    ax2.plot([0, 1], [task_val, rest_val], 'o-', color='gray', alpha=0.5)
ax2.scatter([0]*len(subjects), task_auc, c='#2ecc71', s=50, zorder=5)
ax2.scatter([1]*len(subjects), rest_auc, c='#e74c3c', s=50, zorder=5)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['Task', 'Rest'])
ax2.set_ylabel('AUC (normalized)')
ax2.set_title('Paired Subject Changes')

# Histogram of differences
ax3 = axes[2]
ax3.hist(diff, bins=10, edgecolor='black', alpha=0.7)
ax3.axvline(x=0, color='black', linestyle='--')
ax3.axvline(x=mean_diff, color='red', linestyle='-', label=f'Mean = {mean_diff:.3f}')
ax3.set_xlabel('Task - Rest (AUC)')
ax3.set_ylabel('Count')
ax3.set_title('Distribution of Differences')
ax3.legend()

plt.tight_layout()
plt.savefig('/home/student/sgp-tribe3/figures/SFH_SGP_AUC_plots.png', dpi=150, bbox_inches='tight')
plt.close()

print("Saved: /home/student/sgp-tribe3/figures/SFH_SGP_AUC_plots.png")

# ============================================================
# STEP 8 — OUTPUT
# ============================================================
print("\n" + "="*60)
print("STEP 8: OUTPUT")
print("="*60)

# Save table
curves_df.to_csv('/home/student/sgp-tribe3/results/SFH_SGP_AUC_table.csv', index=False)

# Save stats
with open('/home/student/sgp-tribe3/results/SFH_SGP_AUC_stats.txt', 'w') as f:
    f.write("SFH-SGP AUC ANALYSIS (NON-PARAMETRIC)\n")
    f.write("="*50 + "\n\n")
    f.write(f"N subjects: {len(subjects)}\n")
    f.write(f"k-values: {list(k_values)}\n\n")
    f.write("AUC Results:\n")
    f.write("-"*30 + "\n")
    f.write(f"Task:  {np.mean(task_auc):.4f} ± {np.std(task_auc, ddof=1):.4f}\n")
    f.write(f"Rest: {np.mean(rest_auc):.4f} ± {np.std(rest_auc, ddof=1):.4f}\n")
    f.write(f"Difference: {mean_diff:.4f}\n")
    f.write(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]\n")
    f.write(f"t = {t_stat:.2f}, p = {p_val:.6f}\n")
    f.write(f"Cohen's d = {cohens_d:.2f}\n\n")
    f.write("Bootstrap:\n")
    f.write("-"*30 + "\n")
    f.write(f"Mean: {boot_mean:.4f}, SE: {boot_se:.4f}\n")
    f.write(f"95% CI: [{boot_ci[0]:.4f}, {boot_ci[1]:.4f}]\n")

print("\nSaved:")
print("  - /home/student/sgp-tribe3/results/SFH_SGP_AUC_table.csv")
print("  - /home/student/sgp-tribe3/results/SFH_SGP_AUC_stats.txt")
print("  - /home/student/sgp-tribe3/figures/SFH_SGP_AUC_plots.png")