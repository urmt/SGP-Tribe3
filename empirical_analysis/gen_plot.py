#!/usr/bin/env python3
"""Generate plots"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = "/home/student/sgp-tribe3/empirical_analysis/multitask_results/"
df = pd.read_csv(OUT+"auc_by_subject.csv")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax1 = axes[0]
means = df.groupby('load')['AUC'].mean()
stds = df.groupby('load')['AUC'].std()
x = range(1, 6)
ax1.bar(x, [means.get(i, 0) for i in x], yerr=[stds.get(i, 0) for i in x], capsize=5, alpha=0.7, color='steelblue')
ax1.set_xticks(x)
ax1.set_xticklabels(['rest', 'scap', 'bart', 'stop', 'switch'])
ax1.set_xlabel('Cognitive Load Level')
ax1.set_ylabel('Early AUC (D2+D4)/2')
ax1.set_title('Mean Dimensionality by Task')

ax2 = axes[1]
for subj in df['subject'].unique()[:10]:
    sd = df[df['subject'] == subj]
    ax2.plot(sd['load'], sd['AUC'], 'o-', alpha=0.5, label=subj)
ax2.set_xlabel('Load Level')
ax2.set_ylabel('AUC')
ax2.set_title('Individual Subject Profiles (first 10)')

ax3 = axes[2]
slope, intercept, r, p, se = stats.linregress(df['load'], df['AUC'])
x_line = np.linspace(1, 5, 100)
y_line = slope * x_line + intercept
ax3.scatter(df['load'], df['AUC'], alpha=0.3, s=20)
ax3.plot(x_line, y_line, 'r-', linewidth=2, label=f'r={r:.3f}, p={p:.3f}')
ax3.set_xlabel('Load Level')
ax3.set_ylabel('AUC')
ax3.set_title('Load Gradient Regression')
ax3.legend()

plt.tight_layout()
plt.savefig(OUT+"load_gradient_plot.png", dpi=150)
print(f"Saved plot")