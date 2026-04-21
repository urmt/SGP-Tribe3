#!/usr/bin/env python3
"""
EXPANDED EMPIRICAL VALIDATION - FULL COHORT ANALYSIS
Analyze all available subjects in OpenNeuro ds000114
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

# ============================================================
# STEP 1: DATA DISCOVERY
# ============================================================
print("="*60)
print("STEP 1: DATA DISCOVERY")
print("="*60)

# Load existing parameters data
data = pd.read_csv('/home/student/sgp-tribe3/empirical_analysis/outputs/parameters.csv')

subjects = data['subject'].unique()
n_subjects = len(subjects)
conditions = data['condition'].unique()

print(f"Detected subjects: {n_subjects}")
print(f"Available conditions: {list(conditions)}")

if n_subjects < 15:
    print(f"\nWARNING: Only {n_subjects} subjects available (minimum 25 recommended)")

# ============================================================
# STEP 2: PROCESSING
# ============================================================
print("\n" + "="*60)
print("STEP 2: PROCESSING")
print("="*60)

k_values = np.array([2, 4, 8, 16, 32, 64])

def sigmoid(x, A, k0, beta):
    return A / (1 + np.exp(-beta * (x - k0)))

# Process each subject
results = []
failed_fits = []

for subject in subjects:
    for condition in conditions:
        row = data[(data['subject'] == subject) & (data['condition'] == condition)]
        
        if len(row) == 0:
            failed_fits.append(f"{subject}_{condition}")
            continue
            
        A = row['A'].values[0]
        k0 = row['k0'].values[0]
        beta = row['beta'].values[0]
        
        # Reconstruct residual profile
        profile = sigmoid(k_values, A, k0, beta)
        
        # Normalize
        profile_norm = profile / np.max(np.abs(profile))
        
        # Compute AUC
        auc = trapezoid(profile_norm, k_values)
        auc_norm = auc / len(k_values)
        
        results.append({
            'subject': subject,
            'condition': condition,
            'A': A,
            'k0': k0,
            'beta': beta,
            'profile': profile,
            'profile_norm': profile_norm,
            'AUC': auc,
            'AUC_norm': auc_norm
        })

if failed_fits:
    print(f"Failed fits: {failed_fits}")
else:
    print("All subjects processed successfully")

results_df = pd.DataFrame(results)

# ============================================================
# STEP 3 & 4: GROUP ANALYSIS
# ============================================================
print("\n" + "="*60)
print("STEP 3 & 4: GROUP ANALYSIS")
print("="*60)

# Separate conditions
task_data = results_df[results_df['condition'] == 'task']
rest_data = results_df[results_df['condition'] == 'rest']

# Amplitude analysis
A_task = task_data['A'].values
A_rest = rest_data['A'].values

t_A, p_A = stats.ttest_rel(A_task, A_rest)
dA = np.mean(A_task - A_rest) / np.std(A_task - A_rest, ddof=1)
ci_A = stats.t.interval(0.95, len(A_task)-1, 
                     loc=np.mean(A_task - A_rest),
                     scale=stats.sem(A_task - A_rest))

print(f"\n--- AMPLITUDE ANALYSIS ---")
print(f"Task:   {np.mean(A_task):.2f} ± {np.std(A_task, ddof=1):.2f}")
print(f"Rest:  {np.mean(A_rest):.2f} ± {np.std(A_rest, ddof=1):.2f}")
print(f"Diff:  {np.mean(A_task - A_rest):.2f}")
print(f"t = {t_A:.2f}, p = {p_A:.6f}, d = {dA:.2f}")
print(f"95% CI: [{ci_A[0]:.2f}, {ci_A[1]:.2f}]")

# AUC analysis
auc_task = task_data['AUC_norm'].values
auc_rest = rest_data['AUC_norm'].values

t_AUC, p_AUC = stats.ttest_rel(auc_task, auc_rest)
dAUC = np.mean(auc_task - auc_rest) / np.std(auc_task - auc_rest, ddof=1)
ci_AUC = stats.t.interval(0.95, len(auc_task)-1,
                      loc=np.mean(auc_task - auc_rest),
                      scale=stats.sem(auc_task - auc_rest))

print(f"\n--- AUC ANALYSIS ---")
print(f"Task:   {np.mean(auc_task):.4f} ± {np.std(auc_task, ddof=1):.4f}")
print(f"Rest:  {np.mean(auc_rest):.4f} ± {np.std(auc_rest, ddof=1):.4f}")
print(f"Diff:  {np.mean(auc_task - auc_rest):.4f}")
print(f"t = {t_AUC:.2f}, p = {p_AUC:.6f}, d = {dAUC:.2f}")
print(f"95% CI: [{ci_AUC[0]:.4f}, {ci_AUC[1]:.4f}]")

# ============================================================
# STEP 5: ROBUSTNESS
# ============================================================
print("\n" + "="*60)
print("STEP 5: ROBUSTNESS")
print("="*60)

# Bootstrap
np.random.seed(42)
n_bootstrap = 1000

# Amplitude bootstrap
boot_diffs_A = []
for _ in range(n_bootstrap):
    idx = np.random.randint(0, len(A_task), len(A_task))
    boot_diffs_A.append(np.mean(A_task[idx] - A_rest[idx]))
boot_diffs_A = np.array(boot_diffs_A)

# AUC bootstrap
boot_diffs_auc = []
for _ in range(n_bootstrap):
    idx = np.random.randint(0, len(auc_task), len(auc_task))
    boot_diffs_auc.append(np.mean(auc_task[idx] - auc_rest[idx]))
boot_diffs_auc = np.array(boot_diffs_auc)

print(f"\n--- AMPLITUDE BOOTSTRAP ---")
print(f"Mean: {np.mean(boot_diffs_A):.2f} ± {np.std(boot_diffs_A):.2f}")
print(f"95% CI: [{np.percentile(boot_diffs_A, 2.5):.2f}, {np.percentile(boot_diffs_A, 97.5):.2f}]")

print(f"\n--- AUC BOOTSTRAP ---")
print(f"Mean: {np.mean(boot_diffs_auc):.4f} ± {np.std(boot_diffs_auc):.4f}")
print(f"95% CI: [{np.percentile(boot_diffs_auc, 2.5):.4f}, {np.percentile(boot_diffs_auc, 97.5):.4f}]")

# Leave-one-out
print(f"\n--- LEAVE-ONE-OUT ---")
loo_sig_A = 0
loo_sig_auc = 0
for i in range(len(subjects)):
    mask = np.ones(len(subjects), dtype=bool)
    mask[i] = False
    _, p = stats.ttest_rel(A_task[mask], A_rest[mask])
    if p < 0.05:
        loo_sig_A += 1
    _, p = stats.ttest_rel(auc_task[mask], auc_rest[mask])
    if p < 0.05:
        loo_sig_auc += 1

print(f"Amplitude significant in {loo_sig_A}/{len(subjects)} LOO folds")
print(f"AUC significant in {loo_sig_auc}/{len(subjects)} LOO folds")

# ============================================================
# STEP 6: VISUALIZATION
# ============================================================
print("\n" + "="*60)
print("STEP 6: VISUALIZATION")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Amplitude boxplot
ax1 = axes[0, 0]
bp = ax1.boxplot([A_task, A_rest], tick_labels=['Task', 'Rest'], patch_artist=True)
bp['boxes'][0].set_facecolor('#2ecc71')
bp['boxes'][1].set_facecolor('#e74c3c')
ax1.set_ylabel('Amplitude')
ax1.set_title('Amplitude: Task vs Rest')

# AUC boxplot
ax2 = axes[0, 1]
bp = ax2.boxplot([auc_task, auc_rest], tick_labels=['Task', 'Rest'], patch_artist=True)
bp['boxes'][0].set_facecolor('#2ecc71')
bp['boxes'][1].set_facecolor('#e74c3c')
ax2.set_ylabel('AUC (normalized)')
ax2.set_title('AUC: Task vs Rest')

# Paired amplitude
ax3 = axes[1, 0]
for i, subj in enumerate(subjects):
    ax3.plot([0, 1], [A_task[i], A_rest[i]], 'o-', color='gray', alpha=0.5)
ax3.scatter([0]*len(subjects), A_task, c='#2ecc71', s=50, zorder=5)
ax3.scatter([1]*len(subjects), A_rest, c='#e74c3c', s=50, zorder=5)
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Task', 'Rest'])
ax3.set_ylabel('Amplitude')
ax3.set_title('Paired Subject Changes (Amplitude)')

# Paired AUC
ax4 = axes[1, 1]
for i, subj in enumerate(subjects):
    ax4.plot([0, 1], [auc_task[i], auc_rest[i]], 'o-', color='gray', alpha=0.5)
ax4.scatter([0]*len(subjects), auc_task, c='#2ecc71', s=50, zorder=5)
ax4.scatter([1]*len(subjects), auc_rest, c='#e74c3c', s=50, zorder=5)
ax4.set_xticks([0, 1])
ax4.set_xticklabels(['Task', 'Rest'])
ax4.set_ylabel('AUC (normalized)')
ax4.set_title('Paired Subject Changes (AUC)')

plt.tight_layout()
plt.savefig('/home/student/sgp-tribe3/empirical_analysis/figures/full_cohort_analysis.png', dpi=150)
plt.close()

print("Saved: empirical_analysis/figures/full_cohort_analysis.png")

# ============================================================
# STEP 7: SAVE RESULTS
# ============================================================
print("\n" + "="*60)
print("STEP 7: SAVE RESULTS")
print("="*60)

# Save per-subject results
output_df = results_df[['subject', 'condition', 'A', 'k0', 'beta', 'AUC_norm']].copy()
output_df.columns = ['subject', 'condition', 'amplitude', 'midpoint', 'steepness', 'AUC']
output_df.to_csv('/home/student/sgp-tribe3/empirical_analysis/outputs/full_cohort_results.csv', index=False)

# Save summary statistics
with open('/home/student/sgp-tribe3/empirical_analysis/outputs/statistics_summary.txt', 'w') as f:
    f.write("FULL COHORT ANALYSIS - STATISTICS SUMMARY\n")
    f.write("="*50 + "\n\n")
    f.write(f"N subjects: {n_subjects}\n")
    f.write(f"Conditions: {list(conditions)}\n\n")
    
    f.write("AMPLITUDE ANALYSIS:\n")
    f.write("-"*30 + "\n")
    f.write(f"Task:   {np.mean(A_task):.2f} +/- {np.std(A_task, ddof=1):.2f}\n")
    f.write(f"Rest:  {np.mean(A_rest):.2f} +/- {np.std(A_rest, ddof=1):.2f}\n")
    f.write(f"Diff:  {np.mean(A_task - A_rest):.2f}\n")
    f.write(f"t = {t_A:.2f}, p = {p_A:.6f}\n")
    f.write(f"Cohen's d = {dA:.2f}\n")
    f.write(f"95% CI: [{ci_A[0]:.2f}, {ci_A[1]:.2f}]\n\n")
    
    f.write("AUC ANALYSIS:\n")
    f.write("-"*30 + "\n")
    f.write(f"Task:   {np.mean(auc_task):.4f} +/- {np.std(auc_task, ddof=1):.4f}\n")
    f.write(f"Rest:  {np.mean(auc_rest):.4f} +/- {np.std(auc_rest, ddof=1):.4f}\n")
    f.write(f"Diff:  {np.mean(auc_task - auc_rest):.4f}\n")
    f.write(f"t = {t_AUC:.2f}, p = {p_AUC:.6f}\n")
    f.write(f"Cohen's d = {dAUC:.2f}\n")
    f.write(f"95% CI: [{ci_AUC[0]:.4f}, {ci_AUC[1]:.4f}]\n\n")
    
    f.write("ROBUSTNESS:\n")
    f.write("-"*30 + "\n")
    f.write(f"Amplitude LOO significant: {loo_sig_A}/{len(subjects)}\n")
    f.write(f"AUC LOO significant: {loo_sig_auc}/{len(subjects)}\n")

print("Saved:")
print("  - empirical_analysis/outputs/full_cohort_results.csv")
print("  - empirical_analysis/outputs/statistics_summary.txt")

# ============================================================
# FINAL OUTPUT
# ============================================================
print("\n" + "="*60)
print("FULL COHORT ANALYSIS COMPLETE")
print("="*60)

print(f"\nN subjects: {n_subjects}")
print(f"\nAMPLITUDE:")
print(f"Task:   {np.mean(A_task):.2f} +/- {np.std(A_task, ddof=1):.2f}")
print(f"Rest:  {np.mean(A_rest):.2f} +/- {np.std(A_rest, ddof=1):.2f}")
print(f"Diff:  {np.mean(A_task - A_rest):.2f}")
print(f"t = {t_A:.2f}, p = {p_A:.6f}, d = {dA:.2f}")

print(f"\nAUC:")
print(f"Task:   {np.mean(auc_task):.4f} +/- {np.std(auc_task, ddof=1):.4f}")
print(f"Rest:  {np.mean(auc_rest):.4f} +/- {np.std(auc_rest, ddof=1):.4f}")
print(f"Diff:  {np.mean(auc_task - auc_rest):.4f}")
print(f"t = {t_AUC:.2f}, p = {p_AUC:.6f}, d = {dAUC:.2f}")