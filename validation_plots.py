#!/usr/bin/env python3
"""
SFH-SGP_VALIDATION_STUDY_01 - Visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Load data
data = pd.read_csv('/home/student/sgp-tribe3/empirical_analysis/outputs/parameters.csv')
task_data = data[data['condition'] == 'task'].copy()
rest_data = data[data['condition'] == 'rest'].copy()

# Merge on subject
paired = pd.merge(task_data, rest_data, on='subject', suffixes=('_task', '_rest'))
subjects = paired['subject'].values
N = len(subjects)

A_task = paired['A_task'].values
A_rest = paired['A_rest'].values
k0_task = paired['k0_task'].values
k0_rest = paired['k0_rest'].values
beta_task = paired['beta_task'].values
beta_rest = paired['beta_rest'].values

# Create figure
fig = plt.figure(figsize=(12, 10))

# ============================================================
# Plot 1: Boxplots
# ============================================================
ax1 = fig.add_subplot(2, 2, 1)

positions = [1, 2, 4, 5, 7, 8]
data_boxplot = [
    A_task, A_rest,
    k0_task, k0_rest,
    beta_task, beta_rest
]
bp = ax1.boxplot(data_boxplot, positions=positions, widths=0.6, patch_artist=True)

colors = ['#2ecc71', '#e74c3c'] * 3
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.set_xticks([2, 5, 8])
ax1.set_xticklabels(['Amplitude (A)', 'Midpoint (k0)', 'Steepness (b)'])
ax1.set_ylabel('Parameter Value')
ax1.set_title('Boxplot: Task vs Rest')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# ============================================================
# Plot 2: Paired line plots
# ============================================================
ax2 = fig.add_subplot(2, 2, 2)

for i in range(N):
    ax2.plot([0, 1], [A_task[i], A_rest[i]], 'o-', color='gray', alpha=0.5)
    ax2.plot([2, 3], [k0_task[i], k0_rest[i]], 'o-', color='gray', alpha=0.5)
    ax2.plot([4, 5], [beta_task[i], beta_rest[i]], 'o-', color='gray', alpha=0.5)

ax2.scatter([0]*N, A_task, c='#2ecc71', s=50, zorder=5, label='Task')
ax2.scatter([1]*N, A_rest, c='#e74c3c', s=50, zorder=5, label='Rest')
ax2.scatter([2]*N, k0_task, c='#2ecc71', s=50, zorder=5)
ax2.scatter([3]*N, k0_rest, c='#e74c3c', s=50, zorder=5)
ax2.scatter([4]*N, beta_task, c='#2ecc71', s=50, zorder=5)
ax2.scatter([5]*N, beta_rest, c='#e74c3c', s=50, zorder=5)

ax2.set_xticks([0, 1, 2, 3, 4, 5])
ax2.set_xticklabels(['A\ntask', 'A\nrest', 'k₀\ntask', 'k₀\nrest', 'β\ntask', 'β\nrest'])
ax2.set_ylabel('Parameter Value')
ax2.set_title('Paired Subject Changes')

# ============================================================
# Plot 3: Distribution of differences (Bootstrap)
# ============================================================
ax3 = fig.add_subplot(2, 2, 3)

np.random.seed(42)
n_bootstrap = 1000

for param, task_vals, rest_vals, color in [('A', A_task, A_rest, '#3498db'),
                                     ('k0', k0_task, k0_rest, '#9b59b6'),
                                     ('beta', beta_task, beta_rest, '#e67e22')]:
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, N, N)
        bootstrap_diffs.append(np.mean(task_vals[idx] - rest_vals[idx]))
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    ax3.hist(bootstrap_diffs, bins=30, alpha=0.5, label=param, color=color)

ax3.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax3.set_xlabel('Task - Rest (Difference)')
ax3.set_ylabel('Frequency')
ax3.set_title('Bootstrap Distribution of Differences')
ax3.legend()

# ============================================================
# Plot 4: Summary bar chart with error bars
# ============================================================
ax4 = fig.add_subplot(2, 2, 4)

params = ['A', 'k₀', 'β']
means = [np.mean(A_task - A_rest), np.mean(k0_task - k0_rest), np.mean(beta_task - beta_rest)]
sems = [np.std(A_task - A_rest, ddof=1)/np.sqrt(N),
        np.std(k0_task - k0_rest, ddof=1)/np.sqrt(N),
        np.std(beta_task - beta_rest, ddof=1)/np.sqrt(N)]

x_pos = np.arange(len(params))
bars = ax4.bar(x_pos, means, yerr=[1.96*s for s in sems], 
               color=['#3498db', '#9b59b6', '#e67e22'], alpha=0.7, capsize=5)

ax4.set_xticks(x_pos)
ax4.set_xticklabels(params)
ax4.set_ylabel('Task - Rest (Difference)')
ax4.set_title('Mean Difference ± 95% CI')
ax4.axhline(y=0, color='black', linestyle='--')

# Add significance markers
for i, (mean, p) in enumerate([(np.mean(A_task - A_rest), 0.0000),
                               (np.mean(k0_task - k0_rest), 0.0531),
                               (np.mean(beta_task - beta_rest), 0.0573)]):
    if p < 0.05:
        ax4.annotate('*', (i, mean + 0.1*abs(mean)), ha='center', fontsize=20, fontweight='bold')
    elif p < 0.1:
        ax4.annotate('†', (i, mean + 0.1*abs(mean)), ha='center', fontsize=20)

plt.tight_layout()
plt.savefig('/home/student/sgp-tribe3/figures/SFH_SGP_validation_plots.png', dpi=150, bbox_inches='tight')
plt.close()

print("Figure saved: /home/student/sgp-tribe3/figures/SFH_SGP_validation_plots.png")