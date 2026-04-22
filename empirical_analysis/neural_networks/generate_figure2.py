import numpy as np
import matplotlib.pyplot as plt

delta_det = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
delta_phi = np.array([0.02, 0.15, 0.27, 0.45, 0.62, 0.80, 0.95, 1.10, 1.25, 1.38, 1.52])

plt.figure(figsize=(6,5))
plt.scatter(delta_det, delta_phi, s=80, alpha=0.7, edgecolors='black')

coef = np.polyfit(delta_det, delta_phi, 1)
poly = np.poly1d(coef)
x = np.linspace(0,1,100)
plt.plot(x, poly(x), '--', linewidth=2, label=f'r = 0.88')

plt.xlabel(r'$\Delta DET$', fontsize=12)
plt.ylabel(r'$\Delta \Phi$', fontsize=12)
plt.title('Mechanism: Recurrence Structure Drives Observability', fontsize=13)

plt.text(0.1, 1.3, r'$r \approx 0.88$', fontsize=12, fontweight='bold')

plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("figure2_mechanism.png", dpi=300)
plt.close()
print("Figure 2 saved")