#!/usr/bin/env python3
"""
SFH-SGP_VALIDATION_STUDY_01
Validates interpretive mapping of sigmoid parameters in empirical neural data
"""

import numpy as np
import pandas as pd
from scipy import stats

# Load parameters data
data = pd.read_csv('/home/student/sgp-tribe3/empirical_analysis/outputs/parameters.csv')

# Organize data: separate task and rest conditions
task_data = data[data['condition'] == 'task'].copy()
rest_data = data[data['condition'] == 'rest'].copy()

# Merge on subject to create paired data
paired = pd.merge(task_data, rest_data, on='subject', suffixes=('_task', '_rest'))

N = len(paired)
print(f"N subjects: {N}")

# Extract parameters
A_task = paired['A_task'].values
A_rest = paired['A_rest'].values
k0_task = paired['k0_task'].values
k0_rest = paired['k0_rest'].values
beta_task = paired['beta_task'].values
beta_rest = paired['beta_rest'].values

# Calculate differences
dA = A_task - A_rest
dk0 = k0_task - k0_rest
dbeta = beta_task - beta_rest

# ============================================================
# STATISTICAL TESTING
# ============================================================

print("\n" + "="*60)
print("STATISTICAL RESULTS")
print("="*60)

# Paired t-tests
results = {}

for param, task_vals, rest_vals, diffs in [('A', A_task, A_rest, dA), 
                                        ('k0', k0_task, k0_rest, dk0),
                                        ('beta', beta_task, beta_rest, dbeta)]:
    # Paired t-test
    t_stat, p_val = stats.ttest_rel(task_vals, rest_vals)
    
    # Cohen's d (using difference distribution)
    cohens_d = np.mean(diffs) / np.std(diffs, ddof=1)
    
    # 95% CI of difference
    mean_diff = np.mean(diffs)
    se_diff = np.std(diffs, ddof=1) / np.sqrt(N)
    ci_low = mean_diff - 1.96 * se_diff
    ci_high = mean_diff + 1.96 * se_diff
    
    # Mean ± SD
    mean_task = np.mean(task_vals)
    sd_task = np.std(task_vals, ddof=1)
    mean_rest = np.mean(rest_vals)
    sd_rest = np.std(rest_vals, ddof=1)
    
    results[param] = {
        'mean_task': mean_task,
        'sd_task': sd_task,
        'mean_rest': mean_rest,
        'sd_rest': sd_rest,
        'mean_diff': mean_diff,
        'se_diff': se_diff,
        't_stat': t_stat,
        'p_val': p_val,
        'cohens_d': cohens_d,
        'ci_low': ci_low,
        'ci_high': ci_high
    }
    
    print(f"\n{param}:")
    print(f"  Task:   {mean_task:.2f} ± {sd_task:.2f}")
    print(f"  Rest:  {mean_rest:.2f} ± {sd_rest:.2f}")
    print(f"  Diff:  {mean_diff:.2f} [{ci_low:.2f}, {ci_high:.2f}]")
    print(f"  t = {t_stat:.2f}, p = {p_val:.4f}")
    print(f"  Cohen's d = {cohens_d:.2f}")

# ============================================================
# BOOTSTRAP RESAMPLING
# ============================================================

print("\n" + "="*60)
print("BOOTSTRAP RESAMPLING (1000 iterations)")
print("="*60)

np.random.seed(42)
n_bootstrap = 1000

bootstrap_results = {}

for param, task_vals, rest_vals in [('A', A_task, A_rest), 
                             ('k0', k0_task, k0_rest),
                             ('beta', beta_task, beta_rest)]:
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        # Resample indices
        idx = np.random.randint(0, N, N)
        boot_diff = np.mean(task_vals[idx] - rest_vals[idx])
        bootstrap_diffs.append(boot_diff)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    boot_mean = np.mean(bootstrap_diffs)
    boot_se = np.std(bootstrap_diffs)
    ci_low = np.percentile(bootstrap_diffs, 2.5)
    ci_high = np.percentile(bootstrap_diffs, 97.5)
    
    bootstrap_results[param] = {
        'mean': boot_mean,
        'se': boot_se,
        'ci_low': ci_low,
        'ci_high': ci_high
    }
    
    print(f"\n{param}:")
    print(f"  Bootstrap mean: {boot_mean:.2f}")
    print(f"  Bootstrap SE: {boot_se:.2f}")
    print(f"  95% CI: [{ci_low:.2f}, {ci_high:.2f}]")

# ============================================================
# LEAVE-ONE-OUT ANALYSIS
# ============================================================

print("\n" + "="*60)
print("LEAVE-ONE-OUT ANALYSIS")
print("="*60)

for param, task_vals, rest_vals in [('A', A_task, A_rest), 
                             ('k0', k0_task, k0_rest),
                             ('beta', beta_task, beta_rest)]:
    loo_results = []
    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        t_stat, p_val = stats.ttest_rel(task_vals[mask], rest_vals[mask])
        loo_results.append(p_val)
    
    print(f"\n{param}:")
    print(f"  Min p-value (leave-one-out): {min(loo_results):.4f}")
    print(f"  Max p-value (leave-one-out): {max(loo_results):.4f}")
    print(f"  Subjects influencing significance: {sum([p < 0.05 for p in loo_results])}/{N}")

# ============================================================
# SAVE RESULTS
# ============================================================

# Save table
table_data = []
for param in ['A', 'k0', 'beta']:
    r = results[param]
    table_data.append({
        'parameter': param,
        'task_mean': r['mean_task'],
        'task_sd': r['sd_task'],
        'rest_mean': r['mean_rest'],
        'rest_sd': r['sd_rest'],
        'difference': r['mean_diff'],
        'ci_95': f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]",
        't_statistic': r['t_stat'],
        'p_value': r['p_val'],
        'cohens_d': r['cohens_d']
    })

table_df = pd.DataFrame(table_data)
table_df.to_csv('/home/student/sgp-tribe3/results/SFH_SGP_validation_table.csv', index=False)

# Save statistics text
with open('/home/student/sgp-tribe3/results/SFH_SGP_stats.txt', 'w') as f:
    f.write("SFH-SGP_VALIDATION_STUDY_01\n")
    f.write("="*60 + "\n\n")
    f.write(f"N subjects: {N}\n\n")
    
    f.write("PARAMETER RESULTS:\n")
    f.write("-"*40 + "\n")
    for param in ['A', 'k0', 'beta']:
        r = results[param]
        f.write(f"\n{param}:\n")
        f.write(f"  Task:   {r['mean_task']:.2f} ± {r['sd_task']:.2f}\n")
        f.write(f"  Rest:  {r['mean_rest']:.2f} ± {r['sd_rest']:.2f}\n")
        f.write(f"  Difference: {r['mean_diff']:.2f}\n")
        f.write(f"  95% CI: [{r['ci_low']:.2f}, {r['ci_high']:.2f}]\n")
        f.write(f"  t = {r['t_stat']:.2f}, p = {r['p_val']:.4f}\n")
        f.write(f"  Cohen's d = {r['cohens_d']:.2f}\n")
    
    f.write("\n\nBOOTSTRAP RESULTS:\n")
    f.write("-"*40 + "\n")
    for param in ['A', 'k0', 'beta']:
        b = bootstrap_results[param]
        f.write(f"\n{param}:\n")
        f.write(f"  Mean: {b['mean']:.2f}, SE: {b['se']:.2f}\n")
        f.write(f"  95% CI: [{b['ci_low']:.2f}, {b['ci_high']:.2f}]\n")

print("\n" + "="*60)
print("FILES SAVED")
print("="*60)
print("Results table: /home/student/sgp-tribe3/results/SFH_SGP_validation_table.csv")
print("Stats text: /home/student/sgp-tribe3/results/SFH_SGP_stats.txt")