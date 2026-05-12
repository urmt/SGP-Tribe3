import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

base_F = 0.52
interventions = ['smooth', 'noise', 'shuffle', 'phase']
F_changes = [0.28, 0.04, 0.0, 0.78]
colors = ['green' if c < 0.1 else 'orange' for c in F_changes]

plt.subplot(1, 2, 1)
plt.bar([0], [base_F], color='blue', label='Base')
plt.xticks([0], ['Base'])
plt.ylabel('Fertility (F)')
plt.title('Base Recurrence')
plt.ylim(0, 1)

plt.subplot(1, 2, 2)
bars = plt.bar(interventions, F_changes, color=colors)
plt.ylabel('ΔF from Base')
plt.title('Intervention Effects')
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

for i, v in enumerate(F_changes):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('/home/student/sgp-tribe3/empirical_analysis/neural_networks/paper_neuronet_observability/figure_intervention.png', dpi=150)
plt.close()
print("Saved figure_intervention.png")