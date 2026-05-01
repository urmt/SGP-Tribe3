import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

OUTPUT_DIR = "."
DATA_FILE = "../phase10a_training_dynamics/results.json"

with open(DATA_FILE) as f:
    data = json.load(f)

epochs = np.array([d['epoch'] for d in data])
acc = np.array([d['accuracy'] for d in data])
F_tot = np.array([d['F_total'] for d in data])
F_short = np.array([d['F_short'] for d in data])
F_long = np.array([d['F_long'] for d in data])

fig1, ax1 = plt.subplots(figsize=(8, 5))
color1 = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy', color=color1)
ax1.plot(epochs, acc, 'o-', color=color1, label='Accuracy', linewidth=2, markersize=6)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0.4, 1.05)

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('F_total', color=color2)
ax2.plot(epochs, F_tot, 's--', color=color2, label='F_total', linewidth=2, markersize=6)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0.3, 1.0)

plt.title('Training Trajectory: Accuracy vs F_total')
fig1.tight_layout()
fig1.savefig(f'{OUTPUT_DIR}/figure_training_trajectory.png', dpi=150)
plt.close()

fig2, ax = plt.subplots(figsize=(6, 5))
ax.scatter(F_tot, acc, c=epochs, cmap='viridis', s=80, edgecolor='black', linewidth=0.5)
slope, intercept, r, p, se = stats.linregress(F_tot, acc)
x_line = np.linspace(F_tot.min(), F_tot.max(), 100)
ax.plot(x_line, slope * x_line + intercept, 'r--', linewidth=2, label=f'r = {r:.3f}')
ax.set_xlabel('F_total')
ax.set_ylabel('Accuracy')
ax.set_title('F_total vs Accuracy')
ax.legend()
ax.set_ylim(0.4, 1.05)

cbar = plt.colorbar(ax.scatter(F_tot, acc, c=epochs, cmap='viridis', s=0), ax=ax)
cbar.set_label('Epoch')

fig2.tight_layout()
fig2.savefig(f'{OUTPUT_DIR}/figure_F_total_vs_accuracy.png', dpi=150)
plt.close()

fig3, ax = plt.subplots(figsize=(6, 5))
ax.scatter(F_long, acc, c=epochs, cmap='viridis', s=80, edgecolor='black', linewidth=0.5)
slope, intercept, r, p, se = stats.linregress(F_long, acc)
x_line = np.linspace(F_long.min(), F_long.max(), 100)
ax.plot(x_line, slope * x_line + intercept, 'r--', linewidth=2, label=f'r = {r:.3f}')
ax.set_xlabel('F_long')
ax.set_ylabel('Accuracy')
ax.set_title('F_long vs Accuracy')
ax.legend()
ax.set_ylim(0.4, 1.05)

cbar = plt.colorbar(ax.scatter(F_long, acc, c=epochs, cmap='viridis', s=0), ax=ax)
cbar.set_label('Epoch')

fig3.tight_layout()
fig3.savefig(f'{OUTPUT_DIR}/figure_F_long_vs_accuracy.png', dpi=150)
plt.close()

fig4, ax = plt.subplots(figsize=(6, 5))
ax.scatter(F_short, acc, c=epochs, cmap='viridis', s=80, edgecolor='black', linewidth=0.5)
slope, intercept, r, p, se = stats.linregress(F_short, acc)
x_line = np.linspace(F_short.min(), F_short.max(), 100)
ax.plot(x_line, slope * x_line + intercept, 'r--', linewidth=2, label=f'r = {r:.3f}')
ax.set_xlabel('F_short')
ax.set_ylabel('Accuracy')
ax.set_title('F_short vs Accuracy')
ax.legend()
ax.set_ylim(0.4, 1.05)

cbar = plt.colorbar(ax.scatter(F_short, acc, c=epochs, cmap='viridis', s=0), ax=ax)
cbar.set_label('Epoch')

fig4.tight_layout()
fig4.savefig(f'{OUTPUT_DIR}/figure_F_short_vs_accuracy.png', dpi=150)
plt.close()

fig5, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, F_tot, 'o-', label='F_total', linewidth=2, markersize=6)
ax.plot(epochs, F_long, 's-', label='F_long', linewidth=2, markersize=6)
ax.plot(epochs, F_short, '^-', label='F_short', linewidth=2, markersize=6)
ax.set_xlabel('Epoch')
ax.set_ylabel('F value')
ax.set_title('F Decomposition During Training')
ax.legend()
ax.set_ylim(0, 0.9)

fig5.tight_layout()
fig5.savefig(f'{OUTPUT_DIR}/figure_F_decomposition.png', dpi=150)
plt.close()

print(f"Saved 5 figures to {OUTPUT_DIR}/")