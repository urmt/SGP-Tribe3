#!/usr/bin/env python3
"""
AUC-ONLY ANALYSIS (NORMALIZED) - NO SIGMOID FITTING
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

# STEP 1 — LOAD DATA
print("Loading data...")

data = pd.read_csv('/home/student/sgp-tribe3/empirical_analysis/outputs/parameters.csv')
k_values = np.array([2, 4, 8, 16, 32, 64])

print(f"N subjects: {len(data['subject'].unique())}")
print(f"Conditions: {list(data['condition'].unique())}")
print(f"k-values: {list(k_values)}")

# STEP 2 — COMPUTE NORMALIZED AUC
print("\nComputing normalized AUC...")

# Reconstruct profiles and normalize per subject to remove scale differences
def sigmoid(x, A, k0, beta):
    return A / (1 + np.exp(-beta * (x - k0)))

results = []
for _, row in data.iterrows():
    # Reconstruct at each k-value
    profile = sigmoid(k_values, row['A'], row['k0'], row['beta'])
    
    # Normalize: scale to [-1, 1] range
    profile_norm = profile / np.max(np.abs(profile))
    
    # AUC on normalized profile
    auc = trapezoid(profile_norm, k_values)
    auc_norm = auc / len(k_values)
    
    results.append({
        'subject': row['subject'],
        'condition': row['condition'],
        'AUC': auc,
        'AUC_norm': auc_norm
    })

df = pd.DataFrame(results)

# STEP 3 — STATISTICS
print("\n" + "="*60)
print("AUC RESULTS:")
print("="*60)

task = df[df['condition'] == 'task']['AUC_norm'].values
rest = df[df['condition'] == 'rest']['AUC_norm'].values

# Paired t-test
t_stat, p_val = stats.ttest_rel(task, rest)
diff = task - rest
mean_diff = np.mean(diff)
sd_diff = np.std(diff, ddof=1)
se = sd_diff / np.sqrt(len(diff))
cohens_d = mean_diff / sd_diff
ci_low = mean_diff - 1.96 * se
ci_high = mean_diff + 1.96 * se

# STEP 4 — ROBUSTNESS
print("\nBootstrap (1000 iterations)...")
np.random.seed(42)
boot_diffs = []
for _ in range(1000):
    idx = np.random.randint(0, len(task), len(task))
    boot_diffs.append(np.mean(task[idx] - rest[idx]))
boot_diffs = np.array(boot_diffs)

boot_mean = np.mean(boot_diffs)
boot_se = np.std(boot_diffs)
boot_ci = [np.percentile(boot_diffs, 2.5), np.percentile(boot_diffs, 97.5)]

print("\nLeave-one-out validation...")
loo_sig = 0
for i in range(len(task)):
    mask = np.ones(len(task), dtype=bool)
    mask[i] = False
    _, p = stats.ttest_rel(task[mask], rest[mask])
    if p < 0.05:
        loo_sig += 1

# STEP 5 — VISUALS
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax1 = axes[0]
bp = ax1.boxplot([task, rest], tick_labels=['Task', 'Rest'], patch_artist=True)
bp['boxes'][0].set_facecolor('#2ecc71')
bp['boxes'][1].set_facecolor('#e74c3c')
ax1.set_ylabel('AUC (normalized)')
ax1.set_title('AUC: Task vs Rest')

ax2 = axes[1]
subjects = df['subject'].unique()
for i, subj in enumerate(subjects):
    t_val = df[(df['subject'] == subj) & (df['condition'] == 'task')]['AUC_norm'].values[0]
    r_val = df[(df['subject'] == subj) & (df['condition'] == 'rest')]['AUC_norm'].values[0]
    ax2.plot([0, 1], [t_val, r_val], 'o-', color='gray', alpha=0.5)
ax2.scatter([0]*len(subjects), task, c='#2ecc71', s=50, zorder=5)
ax2.scatter([1]*len(subjects), rest, c='#e74c3c', s=50, zorder=5)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['Task', 'Rest'])
ax2.set_ylabel('AUC (normalized)')
ax2.set_title('Paired Changes')

plt.tight_layout()
plt.savefig('/home/student/sgp-tribe3/figures/SFH_SGP_AUC_final.png', dpi=150)
plt.close()

# STEP 6 — MANDATORY OUTPUT FORMAT
print(f"\nTask: {np.mean(task):.4f} +/- {np.std(task, ddof=1):.4f}")
print(f"Rest: {np.mean(rest):.4f} +/- {np.std(rest, ddof=1):.4f}")
print(f"Difference: {mean_diff:.4f}")
print(f"t: {t_stat:.4f}")
print(f"p: {p_val:.6f}")
print(f"Cohen_d: {cohens_d:.4f}")
print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

# Robustness
print(f"\nBootstrap: mean={boot_mean:.4f} +/- {boot_se:.4f}, 95% CI=[{boot_ci[0]:.4f}, {boot_ci[1]:.4f}]")
print(f"Leave-one-out: {loo_sig}/10 significant at p<0.05")

# Save
df.to_csv('/home/student/sgp-tribe3/results/SFH_SGP_AUC_final.csv', index=False)

with open('/home/student/sgp-tribe3/results/SFH_SGP_AUC_final.txt', 'w') as f:
    f.write("AUC RESULTS:\n")
    f.write(f"Task: {np.mean(task):.4f} +/- {np.std(task, ddof=1):.4f}\n")
    f.write(f"Rest: {np.mean(rest):.4f} +/- {np.std(rest, ddof=1):.4f}\n")
    f.write(f"Difference: {mean_diff:.4f}\n")
    f.write(f"t: {t_stat:.4f}\n")
    f.write(f"p: {p_val:.6f}\n")
    f.write(f"Cohen_d: {cohens_d:.4f}\n")
    f.write(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]\n")

print("\nDone.")