import numpy as np
import matplotlib.pyplot as plt
import json

plt.figure(figsize=(8, 6))

with open('/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase10a_training_dynamics/results.json') as f:
    data = json.load(f)

epochs = [d['epoch'] for d in data]
acc = [d['accuracy'] for d in data]
F_total = [d['F_total'] for d in data]

plt.scatter(F_total, acc, c=epochs, cmap='viridis', s=80, edgecolor='black', linewidth=0.5)
z = np.polyfit(F_total, acc, 1)
p = np.poly1d(z)
x_line = np.linspace(min(F_total), max(F_total), 100)
plt.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'r = -0.99')

plt.xlabel('Fertility (F)')
plt.ylabel('Accuracy')
plt.title('Fertility vs Classification Accuracy')
plt.colorbar(label='Epoch')
plt.legend()
plt.tight_layout()
plt.savefig('/home/student/sgp-tribe3/empirical_analysis/neural_networks/paper_neuronet_observability/figure_F_vs_accuracy.png', dpi=150)
plt.close()
print("Saved figure_F_vs_accuracy.png")